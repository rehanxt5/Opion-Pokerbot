"""exbr.py — Exact Best-Response engine (Gen8 candidate C1 / hypothesis H1). v2

Replaces TMR's approximate replication with EXACT replication of the
reference-class decision function, running the reference's own code
(ref/refsy.py = frozen Gen4 modules, renamed imports):

  1. Exact equity matrix at the river: their eq_t(h) for ALL candidate
     holdings h via score vectors + outer comparison (one batched pass).
  2. Their response distributions to our candidate actions via their own
     postflop_decide on SYNTHETIC requests that carry the full updated
     history (our raise/check appended, their-view actor labels).
  3. Their range weights of us are re-evaluated PER ACTION VARIANT
     (after our raise vs after our check) — the reference re-weights on
     our action before responding (aggression filter), and bet size never
     enters its range model, so one raise-variant evaluation serves all
     our bet sizes.

Consumes v7's MirrorL (their adaptive stats), FP gate, and the
self-validation mechanism (L['tmr_preds'] format-compatible).

v2 fixes vs v1: history injection into the shadow (the decisive bug),
per-variant range weights, prediction 'after' semantics, partial-weight
normalization, turn model (exact, pot-gated).
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REF = os.path.join(_HERE, "ref")
if _REF not in sys.path:
    sys.path.insert(0, _REF)

try:
    import refpb as _rpb
    import refmc as _rmc
    import refsy as _rsy
    REFS_OK = True
except Exception:                                    # pragma: no cover
    REFS_OK = False

EXBR_ON = os.environ.get("SYN_EXBR", "1") == "1"
EXBR_DEBUG = os.environ.get("EXBR_DEBUG", "") == "1"
K_DRAWS = 6                 # rng draws per (candidate, bet) response
TOP_K = 96                  # candidates for response computation
MS_CAP = 380.0              # per-decision cap (ms)
LEG_BUDGET_MS = 11000.0     # per-leg EXBR allowance
TURN_MIN_POT = 30           # exact turn model only for real pots


# ------------------------------------------------------------------ utilities
def _deck_excl(known):
    return _rpb._deck_excluding([int(c) for c in known])


def _combos(deck):
    n = deck.size
    i, j = np.triu_indices(n, k=1)
    return deck[np.stack([i, j], axis=1)]


def _their_hist(our_hist, extra=None):
    """Their view of the hand history: swap me/opp; optionally append our
    action (theirs as 'opp')."""
    out = []
    for h in our_hist:
        e = dict(h)
        a = e.get("actor")
        if a == "me":
            e["actor"] = "opp"
        elif a == "opp":
            e["actor"] = "me"
        out.append(e)
    if extra is not None:
        out = out + [dict(extra)]
    return out


class _ShadowTR:
    """Their-view tracker replica for calling refsy functions."""

    def __init__(self, street, board, hist_their, our_discard,
                 hand_id, total_hands, their_leg_net):
        self.__dict__["_syn_l"] = None      # set by caller
        self.street = street
        self.board = [int(c) for c in board]
        self.history = hist_their
        self.hole2 = [0, 1]                 # placeholder; set per candidate
        self.hole3 = None
        self.opp_discard = int(our_discard) if our_discard is not None else None
        self.hand_id = hand_id
        self.total_hands = total_hands
        self.leg_net = their_leg_net
        self.time_left_ms = 24000.0
        self.time_bank_ms = 30000.0


def _l_view(mirror):
    return {
        "st": mirror.st, "pf": mirror.pf,
        "show_agg_pf": mirror.show_agg_pf, "show_pass_pf": mirror.show_pass_pf,
        "show_agg_post": mirror.show_agg_post, "show_pass_post": mirror.show_pass_post,
        "n_show": mirror.n_show,
        "discard_kept_rate": getattr(mirror, "discard_kept_rate", None),
        "n_disc": getattr(mirror, "n_disc", 0),
    }


def _blend(eq_c, eq_u, mirror):
    _, gain_post = mirror.gains()
    kappa = 0.25 + 0.75 * gain_post
    return (1.0 - kappa) * eq_u + kappa * eq_c


# ----------------------------------------------------------------- base model
def _base_model(tr):
    """Score vectors for the river: their candidates, our combos, our hand."""
    board5 = [int(c) for c in tr.board[:5]]
    our2 = [int(c) for c in tr.hole2]
    d_us = tr.my_discard
    d_them = tr.opp_discard
    deck_o = _deck_excl(board5 + ([int(d_us)] if d_us is not None else []))
    O = _combos(deck_o)
    b5 = np.asarray(board5, dtype=np.int8)
    s_o = _rpb.eval7_batch(np.concatenate(
        [O, np.tile(b5, (len(O), 1))], axis=1)).astype(np.float64)
    deck_h = _deck_excl(board5 + our2 + ([int(d_them)] if d_them is not None else []))
    H = _combos(deck_h)
    s_h = _rpb.eval7_batch(np.concatenate(
        [H, np.tile(b5, (len(H), 1))], axis=1)).astype(np.float64)
    s_me = float(_rpb.eval7_batch(np.asarray([our2 + board5], dtype=np.int8))[0])
    mywin_h = np.where(s_h < s_me, 1.0, np.where(s_h == s_me, 0.5, 0.0))
    explain_o = (_rsy.discard_explain_weights(O, int(d_us), board5)
                 if d_us is not None else np.ones(len(O)))
    S_o = {"N": len(O), "combos": O, "mywin": None,
           "their_pct": _rsy._rank_pct(s_o),
           "p2": _rsy._rank_pct(_rsy.pair_quality(O[:, 0], O[:, 1])),
           "explain": explain_o}
    return {"cand": H, "s_h": s_h, "s_o": s_o, "mywin": mywin_h,
            "S_o": S_o, "board": board5, "our2": our2, "d_us": d_us}


def _weights_variant(tr, mirror, model, street, hist_variant):
    """Their range weights over our combos AND their per-candidate eq,
    evaluated with the given their-view history (incl. our action)."""
    O = model["S_o"]["combos"]
    sh = _ShadowTR(street, model["board"], hist_variant, model["d_us"],
                   tr.hand_id, tr.total_hands,
                   -float(getattr(tr, "leg_net", 0.0) or 0.0))
    L = _l_view(mirror)
    sh.__dict__["_syn_l"] = L
    L["rng"] = np.random.default_rng(11)
    w_o = np.asarray(_rsy.range_weights(sh, model["S_o"], street),
                     dtype=np.float64)
    w_o = np.maximum(w_o, 1e-12)
    w_o = w_o / w_o.sum()
    Hm = model["cand"].astype(np.int64)
    Om = O.astype(np.int64)
    valid = ((Hm[:, None, 0] != Om[None, :, 0]) & (Hm[:, None, 0] != Om[None, :, 1])
             & (Hm[:, None, 1] != Om[None, :, 0]) & (Hm[:, None, 1] != Om[None, :, 1]))
    W = np.where(model["s_h"][:, None] > model["s_o"][None, :], 1.0,
                 np.where(model["s_h"][:, None] == model["s_o"][None, :], 0.5, 0.0))
    wv = w_o[None, :] * valid
    tot = wv.sum(axis=1)
    eq_c = (W * wv).sum(axis=1) / np.maximum(tot, 1e-12)
    eq_u = (W * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1.0)
    eq_t = np.asarray([_blend(eq_c[k], eq_u[k], mirror)
                       for k in range(len(eq_c))])
    return w_o, eq_t, eq_u, W, L, sh


# --------------------------------------------------------- synthetic requests
def _syn_req_facing(req, R, street):
    """Their action_request after we raise to R."""
    my_bet = int(req["my_bet"])
    opp_bet = int(req["opp_bet"])            # their current street bet
    pot = int(req["pot"])
    R = int(R)
    add = R - my_bet
    to_call = R - opp_bet
    their_stack = int(req["opp_stack"])
    our_stack_after = int(req["my_stack"]) - add
    hist = _their_hist(req.get("history") or [],
                       {"street": street, "actor": "opp", "action": "raise",
                        "amount": R})
    n_raises = sum(1 for h in hist if h.get("street") == street
                   and h.get("action") == "raise")
    legal = ["fold", "call"] if to_call > 0 else ["check"]
    max_raise = min(opp_bet + their_stack, R + max(0, our_stack_after))
    can_raise = n_raises < 4 and max_raise > R
    if can_raise:
        legal.append("raise")
    sreq = {"type": "action_request", "street": street, "board": req["board"],
            "pot": pot + add, "my_bet": opp_bet, "opp_bet": R,
            "to_call": to_call, "my_stack": their_stack,
            "opp_stack": max(0, our_stack_after), "legal_actions": legal}
    if can_raise:
        last_raise = max(R - opp_bet, 2)
        sreq["min_raise"] = min(R + last_raise, max_raise)
        sreq["max_raise"] = max_raise
    sreq["raises_left"] = max(0, 4 - n_raises)
    sreq["history"] = hist
    return sreq


def _syn_req_checked(req, street):
    """Their action_request after we check."""
    hist = _their_hist(req.get("history") or [],
                       {"street": street, "actor": "opp", "action": "check"})
    n_raises = sum(1 for h in hist if h.get("street") == street
                   and h.get("action") == "raise")
    legal = ["check"]
    max_raise = min(int(req["opp_bet"]) + int(req["opp_stack"]),
                    int(req["my_bet"]) + int(req["my_stack"]))
    if n_raises < 4 and max_raise > int(req["my_bet"]):
        legal.append("raise")
    sreq = {"type": "action_request", "street": street, "board": req["board"],
            "pot": int(req["pot"]), "my_bet": int(req["opp_bet"]),
            "opp_bet": int(req["my_bet"]),
            "to_call": int(req["my_bet"]) - int(req["opp_bet"]),
            "my_stack": int(req["opp_stack"]), "opp_stack": int(req["my_stack"]),
            "legal_actions": legal}
    if "raise" in legal:
        sreq["min_raise"] = int(req["my_bet"]) + 2
        sreq["max_raise"] = max_raise
    sreq["raises_left"] = max(0, 4 - n_raises)
    sreq["history"] = hist
    return sreq


# ------------------------------------------------------------- response calls
def their_response(sh, L, P, S_struct, w_o, eq_t_k, eq_u_k, sreq, mywin_row,
                   k_draws=K_DRAWS, seed=5):
    """Their action distribution at sreq (K rng draws).
    sh.history/sreq['history'] already include our action."""
    dist = {"fold": 0.0, "call": 0.0, "check": 0.0, "raise": 0.0}
    raise_amts = []
    for d in range(k_draws):
        L["rng"] = np.random.default_rng(seed * 977 + d)
        try:
            S_d = dict(S_struct)
            S_d["mywin"] = mywin_row
            reply = _rsy.postflop_decide(sh, sreq, eq_t_k, eq_u_k, P, L, S_d, w_o)
        except Exception:
            reply = {"type": "action", "action": "check"}
        if reply is None:
            reply = {"type": "action", "action": "check"}
        a = reply.get("action", "check")
        dist[a] = dist.get(a, 0.0) + 1.0
        if a == "raise" and reply.get("amount") is not None:
            raise_amts.append(int(reply["amount"]))
    n = float(k_draws)
    return (dist.get("fold", 0.0) / n, dist.get("call", 0.0) / n,
            dist.get("check", 0.0) / n, dist.get("raise", 0.0) / n,
            (float(np.mean(raise_amts)) if raise_amts else None))


def _ev_raise(req, street, model, w_h, eq_t, eq_u, L, P, sh, S_o, w_o,
              W, mywin_h, R, deadline):
    """EV of our raise-to R. sh/L/P/w_o carry the RAISE-variant view."""
    my_bet = int(req["my_bet"])
    pot = int(req["pot"])
    b = R - my_bet
    if b <= 0:
        return None
    sreq = _syn_req_facing(req, R, street)
    sh.history = sreq["history"]
    order = np.argsort(-w_h)
    top = order[:TOP_K]
    wh = w_h[top]
    tot_w = 0.0
    ev_sum = 0.0
    agg = np.zeros(3)
    for idx, k in enumerate(top):
        if time.monotonic() > deadline and tot_w > 0:
            break
        sh.hole2 = [int(c) for c in model["cand"][k]]
        p_f, p_c, p_x, p_r, mean_r = their_response(
            sh, L, P, S_o, w_o, eq_t[k], eq_u[k], sreq, W[k], seed=101 + k)
        wk = float(wh[idx])
        tot_w += wk
        agg += np.array([p_f, p_c, p_r]) * wk
        eqk = mywin_h[k]
        # showdown at level R: we add a = R-m, they add c = R-t
        a = R - my_bet
        c = R - int(req["opp_bet"])
        ev_call = eqk * (pot + a + c) - a
        if mean_r is not None:
            R2 = int(mean_r)                    # their re-raise-to level
            a2, c2 = R2 - my_bet, R2 - int(req["opp_bet"])
            ev_call2 = eqk * (pot + a2 + c2) - a2
            ev_r = ev_call2 if ev_call2 > 0 else -b
        else:
            ev_r = -b
        ev_sum += wk * (p_f * pot + p_c * ev_call + p_r * ev_r)
    if tot_w < 1e-12:
        return None
    return ev_sum / tot_w, agg / tot_w


def _ev_check(req, street, model, w_h, eq_t, eq_u, L, P, sh, S_o, w_o,
              W, mywin_h, deadline):
    """EV of checking. sh/L/P/w_o carry the CHECK-variant view."""
    pot = int(req["pot"])
    sreq = _syn_req_checked(req, street)
    sh.history = sreq["history"]
    order = np.argsort(-w_h)
    top = order[:TOP_K]
    wh = w_h[top]
    tot_w = 0.0
    ev_sum = 0.0
    for idx, k in enumerate(top):
        if time.monotonic() > deadline and tot_w > 0:
            break
        sh.hole2 = [int(c) for c in model["cand"][k]]
        p_f, p_c, p_x, p_r, mean_r = their_response(
            sh, L, P, S_o, w_o, eq_t[k], eq_u[k], sreq, W[k], seed=211 + k)
        eqk = mywin_h[k]
        ev_x = eqk * pot
        if mean_r is not None:
            bp = max(1, mean_r - int(req["my_bet"]))
        else:
            bp = max(2, pot // 2)
        ev_bet = eqk * (pot + 2 * bp) - bp
        ev_bet = ev_bet if ev_bet > 0 else 0.0
        wk = float(wh[idx])
        tot_w += wk
        ev_sum += wk * (p_x * ev_x + p_r * ev_bet)
    if tot_w < 1e-12:
        return 0.0
    return ev_sum / tot_w


# ------------------------------------------------------------------- chooser
def exbr_decide(tr, req, L, S, wgt):
    """Exact-BR chooser at the river. Returns action dict or None."""
    if not (REFS_OK and EXBR_ON):
        return None
    t0 = time.monotonic()
    street = tr.street
    if street != "river":
        return None
    if L.get("exbr_ms", 0.0) > LEG_BUDGET_MS:
        return None
    deadline = t0 + MS_CAP / 1000.0
    mirror = L.get("mirror")
    if mirror is None:
        return None
    try:
        model = _base_model(tr)
        # ---- two history variants: after our raise / after our check
        w_or, eq_tr, eq_ur, W, Lr, sh_r = _weights_variant(
            tr, mirror, model, street,
            _their_hist(tr.history, {"street": street, "actor": "opp",
                                     "action": "raise", "amount": 0}))
        w_oc, eq_tc, eq_uc, _, Lc, sh_c = _weights_variant(
            tr, mirror, model, street,
            _their_hist(tr.history, {"street": street, "actor": "opp",
                                     "action": "check"}))
        # host weights over their candidates: the CHASSIS's own range model
        # (at river the chassis sample IS the same exact enumeration)
        if S is not None and "combos" in S and wgt is not None \
                and len(wgt) == len(model["cand"]):
            w_h = np.asarray(wgt, dtype=np.float64)
        else:
            S_h = {"N": len(model["cand"]), "combos": model["cand"],
                   "mywin": model["mywin"],
                   "their_pct": _rsy._rank_pct(model["s_h"]),
                   "p2": _rsy._rank_pct(_rsy.pair_quality(model["cand"][:, 0],
                                                          model["cand"][:, 1])),
                   "explain": (_rsy.discard_explain_weights(
                       model["cand"], int(tr.opp_discard), model["board"])
                       if tr.opp_discard is not None else
                       np.ones(len(model["cand"])))}
            w_h = np.asarray(_rsy.range_weights(tr, S_h, street), dtype=np.float64)
        w_h = np.maximum(w_h, 1e-12)
        w_h = w_h / w_h.sum()
        mywin_h = model["mywin"]
        P_r = _rsy.knobs(sh_r, Lr)
        P_c = _rsy.knobs(sh_c, Lc)
        pot = int(req["pot"])
        my_bet = int(req["my_bet"])
        mr = int(req.get("min_raise", 2))
        xr = int(req.get("max_raise", 200))
        to_call = int(req["to_call"])
        n_acts_street = sum(1 for h in tr.history
                            if h.get("street") == street)
        if to_call == 0:
            ev_check = _ev_check(req, street, model, w_h, eq_tc, eq_uc, Lc, P_c,
                                 sh_c, model["S_o"], w_oc, W, mywin_h, deadline)
            best = ("check", ev_check, None)
            for frac in (0.35, 0.5, 0.7, 0.9, 1.2, 1.6, 2.1):
                if time.monotonic() > deadline:
                    break
                b = int(round(frac * pot))
                R = my_bet + b
                R = max(mr, min(R, xr))
                if R <= my_bet:
                    continue
                r = _ev_raise(req, street, model, w_h, eq_tr, eq_ur, Lr, P_r,
                              sh_r, model["S_o"], w_or, W, mywin_h, R, deadline)
                if r is None:
                    continue
                ev, agg = r
                if ev > best[1] + 0.35:
                    best = (R, ev, agg)
            L["exbr_ms"] = L.get("exbr_ms", 0.0) + (time.monotonic() - t0) * 1000.0
            if best[0] != "check":
                amt = int(best[0])
                if amt >= 0.85 * xr:
                    amt = xr
                if best[2] is not None:
                    L["tmr_preds"].append({"street": street,
                                           "pred": tuple(best[2]),
                                           "after": n_acts_street})
                if EXBR_DEBUG:
                    print(f"[exbr] h{tr.hand_id} raise {amt} ev={best[1]:.1f} "
                          f"evc={ev_check:.1f} {(time.monotonic()-t0)*1000:.0f}ms",
                          file=sys.stderr)
                return {"type": "action", "action": "raise", "amount": int(amt)}
            if EXBR_DEBUG:
                print(f"[exbr] h{tr.hand_id} check ev={ev_check:.1f} "
                      f"sv={mirror.pred_hit}/{mirror.pred_tot} "
                      f"{(time.monotonic()-t0)*1000:.0f}ms", file=sys.stderr)
            return {"type": "action", "action": "check"}
        # ---- facing their bet
        bet_size = int(req["opp_bet"]) - my_bet
        if bet_size > 0:
            like = 0.12 + 0.88 * (eq_tr > max(0.47, 0.53 - 0.06 * (pot > 30)))
            w_h2 = w_h * like
            if w_h2.sum() > 1e-9:
                w_h = w_h2 / w_h2.sum()
        eq_now = float((w_h * mywin_h).sum())
        ev_call = eq_now * (pot + to_call) - to_call
        choice, best_ev, best_amt, best_agg = "call", ev_call, None, None
        if "raise" in req["legal_actions"]:
            for frac in (0.9, 1.4):
                if time.monotonic() > deadline:
                    break
                add = max(2, int(frac * (pot + to_call)))
                R = int(req["opp_bet"]) + add
                R = max(mr, min(R, xr))
                r = _ev_raise(req, street, model, w_h, eq_tr, eq_ur, Lr, P_r,
                              sh_r, model["S_o"], w_or, W, mywin_h, R, deadline)
                if r is None:
                    continue
                ev, agg = r
                if ev > best_ev:
                    choice, best_ev, best_amt, best_agg = "raise", ev, R, agg
        if choice == "call" and "call" not in req["legal_actions"]:
            choice = "check" if "check" in req["legal_actions"] else "fold"
        if choice == "fold" and "fold" not in req["legal_actions"]:
            choice = "check" if "check" in req["legal_actions"] else "call"
        L["exbr_ms"] = L.get("exbr_ms", 0.0) + (time.monotonic() - t0) * 1000.0
        if EXBR_DEBUG:
            print(f"[exbr] h{tr.hand_id} face -> {choice} ev={best_ev:.1f} "
                  f"{(time.monotonic()-t0)*1000:.0f}ms", file=sys.stderr)
        if choice == "raise":
            if best_agg is not None:
                L["tmr_preds"].append({"street": street, "pred": tuple(best_agg),
                                       "after": n_acts_street})
            amt = int(best_amt)
            if amt >= 0.85 * xr:
                amt = xr
            return {"type": "action", "action": "raise", "amount": amt}
        return {"type": "action", "action": choice}
    except Exception as e:
        try:
            print(f"[exbr] error: {e!r}", file=sys.stderr)
        except Exception:
            pass
        return None
