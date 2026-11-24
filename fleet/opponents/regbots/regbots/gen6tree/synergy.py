"""synergy.py — Approach 6 "Synergy": real-time range-conditioned equity engine.

Base architecture = A3 (MC equity + opponent modeling), upgraded along layers
selected with the SYN_LEVEL env var (cumulative), so each layer is ablatable:

  L1  correctness + reuse (speed WITHOUT accuracy loss):
        - opponent's exposed discard card excluded from their sampling deck
        - one MC pass per street, cached: re-weighting for new information is a
          dot product, not a re-simulation
        - flop MC merged with the discard decision (same 6 keep-combos)
        - exact river enumeration replaces per-request Monte-Carlo
  L2  range conditioning (accuracy): importance weights over the cached sample
        - preflop-action bands (limp / caller / open / 3bet / 4bet)
        - discard-explain synergy model (they kept the cards that fit the board)
        - street-aggression strength filter (exp-likelihood form)
        - showdown calibration: the filter strength (gain) is driven by the
          OBSERVED correlation between their actions and shown hands, so vs an
          uncorrelated (random) aggressor the model flattens back to uniform —
          conditioning can only help, never confuse
  L3  (folded into L2 build: exact river + per-combo strength percentiles)
  L4  policy upgrades: conditioned-equity value ladder, fold-EV pure bluffs,
        street-aware semi-bluffs, texture sizing, street-resolved exploit knobs
  L5  point-max overlay: late-leg variance shaping (protect a lead / chase a
        deficit), scaled by hands-left and leg_net

Time discipline: every reply's full wall time is charged to the leg time bank
(30 s Stage 1 / 75 s Stage 2, 5 s hard cap per reply). Sample sizes adapt to
the live bank fraction; all builders take a deadline; guards everywhere.
"""
import sys, os, time, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbcore as pb
import mcbase as mc

try:
    import bot_om as om   # level-0 parity mode only
except Exception:
    om = None

LEVEL = int(os.environ.get("SYN_LEVEL", "4"))
if LEVEL > 4:
    LEVEL = 4          # 5 kept as an alias for the full stack
if not pb.PREFLOP.ok:
    LEVEL = min(LEVEL, 1)     # no table -> no percentile conditioning
DEBUG = os.environ.get("SYN_DEBUG", "") == "1"
NEUTRAL_ONLY = os.environ.get("SYN_NEUTRAL", "") == "1"     # bisection: fixed params
NO_EXCLUDE = os.environ.get("SYN_NOEXCLUDE", "") == "1"     # bisection: sample their range including their discard
OLD_DISCARD = os.environ.get("SYN_OLDDISC", "") == "1"       # bisection: mcbase standalone discard MC

# ---- gen5 variant flags (each independently testable, default OFF = syn4) ----
H3 = os.environ.get("SYN_H3", "") == "1"    # maturity-gated, hard-capped adaptation
H4 = os.environ.get("SYN_H4", "1") != "0"   # river overbet value option (gen5: on)
H7 = os.environ.get("SYN_H7", "") == "1"    # MC sample rebalance flop->turn
H9 = os.environ.get("SYN_H9", "") == "1"    # cap-keyed late-leg overlay
H10 = os.environ.get("SYN_H10", "1") != "0" # preflop open-raise size mix (gen5: on)
H5 = os.environ.get("SYN_H5", "1") != "0"   # bet-sizing tell conditioning (gen5: on)

# ---- gen6 hypothesis flags (each testable on the gen5 baseline) ----
G6_A = os.environ.get("SYN_G6A", "1") != "0" # pot-odds defend vs raises (gen6: on)
G6_B = os.environ.get("SYN_G6B", "") == "1"  # flop c-bet boost as preflop aggressor
G6_C = os.environ.get("SYN_G6C", "") == "1"  # threebet size mix 16/20
G6_D = os.environ.get("SYN_G6D", "") == "1"  # turn probe bet vs check-heavy opponents
G6_F = os.environ.get("SYN_G6F", "1") != "0" # river-complete turn sampling (gen6: on)


def _was_pf_aggressor(tr):
    return any(h.get("street") == "preflop" and h.get("actor") == "me"
               and h.get("action") == "raise" for h in tr.history)


def _opp_checked_street(tr, s):
    return any(h.get("street") == s and h.get("actor") == "opp"
               and h.get("action") == "check" for h in tr.history)
H1 = os.environ.get("SYN_H1", "") == "1"    # range-conditioned preflop defend

# ------------------------------------------------------------------ fast pct
_SORTED_EQ = np.sort(pb.PREFLOP.equity) if pb.PREFLOP.ok else None


def _pct_of_eq(v):
    return float(np.searchsorted(_SORTED_EQ, v) / max(1, len(_SORTED_EQ)))


def preflop_pct3(cards3):
    if _SORTED_EQ is None:
        return 0.5
    cid = pb.PREFLOP.class_id(cards3)
    if cid is None:
        return 0.5
    return _pct_of_eq(float(pb.PREFLOP.equity[cid]))


def preflop_pct3_batch(rows):
    """rows: (N,3) int array -> (N,) float percentile of each 3-card hand."""
    if _SORTED_EQ is None:
        return np.full(len(rows), 0.5)
    lut = pb.PREFLOP.class_of_hand
    eq = pb.PREFLOP.equity
    out = np.empty(len(rows), dtype=np.float64)
    for i in range(len(rows)):
        r = rows[i]
        key = tuple(sorted((int(r[0]), int(r[1]), int(r[2]))))
        cid = lut.get(key)
        out[i] = eq[cid] if cid is not None else 0.5
    return np.searchsorted(_SORTED_EQ, out) / max(1, len(_SORTED_EQ))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


# ------------------------------------------------------------- range weights
BANDS = {
    "limped":   (0.02, 0.58),
    "caller":   (0.18, 0.90),
    "open":     (0.50, 1.02),
    "threebet": (0.70, 1.02),
    "fourbet":  (0.78, 1.02),
    None:       (0.02, 1.02),
}


def classify_preflop(history):
    """Classify the opponent's preflop behavior this hand from full history."""
    pf = [h for h in history if h.get("street") == "preflop"]
    if not pf:
        return None
    opp = [h for h in pf if h.get("actor") == "opp"]
    raises = [i for i, h in enumerate(pf) if h.get("action") == "raise"]
    opp_raise_pos = {i for i, h in enumerate(pf)
                     if h.get("actor") == "opp" and h.get("action") == "raise"}
    if not raises:
        return "limped"
    if not opp_raise_pos:
        # opponent never raised: caller (limp-call or blind-call of an open)
        return "caller"
    opp_raises = len(opp_raise_pos)
    first_raiser_is_opp = raises[0] in opp_raise_pos
    if opp_raises >= 2:
        return "fourbet"
    if first_raiser_is_opp:
        return "open"
    return "threebet"


def band_weights(pct3, lo, hi, gain=1.0, sigma=0.14):
    """Soft plateau weight on the percentile band, scaled by evidence gain."""
    w = _sig((pct3 - lo) / sigma) * _sig((hi - pct3) / sigma)
    return 1.0 + gain * (w - 1.0)


# ------------------------------------------------------- discard-explain
def _synergy_score(a, b, branks_mask, flush_suit):
    """Cheap 'why keep these two cards' score. a, b: (M,) card int arrays."""
    ra, rb = (a >> 2).astype(np.int64), (b >> 2).astype(np.int64)
    sa, sb = (a & 3).astype(np.int64), (b & 3).astype(np.int64)
    s = np.zeros(len(a), dtype=np.float64)
    s += branks_mask[ra] * 1.0 + branks_mask[rb] * 1.0     # pair with board
    s += (ra == rb) * 1.2                                  # own pair
    s += (sa == sb) * 0.4                                  # suited
    if flush_suit is not None:
        s += (sa == flush_suit) * 0.7 + (sb == flush_suit) * 0.7
    s += (np.maximum(ra, rb) >= 9) * 0.3 + ((ra >= 9) & (rb >= 9)) * 0.5
    s += (np.abs(ra - rb) <= 2) * 0.25                     # connected
    return s


def discard_explain_weights(combos2, d, board):
    """Weight opp kept-2 candidates by 'would they have discarded d?'."""
    if d is None:
        return np.ones(len(combos2))
    board_r = [c >> 2 for c in board]
    branks_mask = np.zeros(13, dtype=bool)
    for r in board_r:
        branks_mask[r] = True
    suits = np.bincount([c & 3 for c in board], minlength=4)
    flush_suit = int(np.argmax(suits)) if suits.max() >= 2 else None
    a, b = combos2[:, 0].astype(np.int64), combos2[:, 1].astype(np.int64)
    dd = np.full(len(a), int(d), dtype=np.int64)
    s_ab = _synergy_score(a, b, branks_mask, flush_suit)
    s_ad = _synergy_score(a, dd, branks_mask, flush_suit)
    s_bd = _synergy_score(b, dd, branks_mask, flush_suit)
    kept = s_ab >= np.maximum(s_ad, s_bd) - 0.05
    w = np.where(kept, 1.0, 0.35)
    dr = d >> 2
    if dr >= 9:      # threw away a broadway -> kept something that fits
        w *= 0.80 + 0.40 * (s_ab / (s_ab.max() + 1e-9))
    elif dr <= 5:    # threw away junk -> kept real cards
        hi = np.maximum(a >> 2, b >> 2)
        w *= 0.85 + 0.30 * ((hi >= 9) | ((a >> 2) == (b >> 2)))
    return w * 0.85   # mild damping: model is approximate


# ----------------------------------------------------------- street samples
def _rank_pct(scores):
    n = len(scores)
    if n < 2:
        return np.full(max(n, 0), 0.5)
    return np.argsort(np.argsort(scores)).astype(np.float64) / (n - 1)


def pair_quality(a, b):
    """Cheap preflop-ish quality of a 2-card holding (M,) arrays of card ints.
    Used to range-condition on turn/river, where their third card (the exposed
    discard) is fixed and a 3-card percentile cannot discriminate candidates."""
    ra, rb = (a >> 2).astype(np.float64), (b >> 2).astype(np.float64)
    hi = np.maximum(ra, rb)
    q = hi                                            # high card
    q = q + (ra == rb) * (hi + 6.0)                   # pocket pair
    q = q + ((a & 3) == (b & 3)) * 2.0                # suited
    q = q + np.clip(4.0 - np.abs(ra - rb), 0.0, 4.0) * 0.5   # connected
    return q


def build_flop_sample(hole3, board3, N, rng):
    """One MC pass: opp3 + runout samples; all 6 keep-option scores.

    Produces:
      w_best   our per-runout best-keep win vs their per-runout best (clairvoyant
                both sides — mcbase parity semantics for BETTING decisions)
      w_keep   our fixed-keep wins vs their REALISTIC fixed keep (they discard by
                flop synergy) — the DISCARD objective (true fixed-keep EV)
      their_pct / pct3 for range weighting
    """
    hole3 = list(hole3)
    board3 = list(board3)
    deck = pb._deck_excluding(hole3 + board3)
    nxt = pb._sample_cards(deck, 5, N, rng)          # (N,5): opp3, turn, river
    opp3, run = nxt[:, :3], nxt[:, 3:]
    b3 = np.asarray(board3, dtype=np.int8)
    board5 = np.concatenate([np.tile(b3, (N, 1)), run], axis=1)
    h = np.asarray(hole3, dtype=np.int8)
    keeps = ((0, 1), (0, 2), (1, 2))
    rows = np.empty((6 * N, 7), dtype=np.int8)
    for k, keep in enumerate(keeps):
        rows[k * N:(k + 1) * N, 0:2] = h[list(keep)]
        rows[k * N:(k + 1) * N, 2:7] = board5
    for k, keep in enumerate(keeps):
        rows[(3 + k) * N:(4 + k) * N, 0:2] = opp3[:, list(keep)]
        rows[(3 + k) * N:(4 + k) * N, 2:7] = board5
    scores = pb.eval7_batch(rows)
    obest = np.maximum(np.maximum(scores[3 * N:4 * N], scores[4 * N:5 * N]),
                       scores[5 * N:6 * N])
    # their REALISTIC keep: the pair with the best flop synergy (fixed pre-turn)
    branks_mask = np.zeros(13, dtype=bool)
    for c in board3:
        branks_mask[c >> 2] = True
    suits = np.bincount([c & 3 for c in board3], minlength=4)
    fs = int(np.argmax(suits)) if suits.max() >= 2 else None
    o0, o1, o2 = opp3[:, 0].astype(np.int64), opp3[:, 1].astype(np.int64), opp3[:, 2].astype(np.int64)
    sy = np.stack([_synergy_score(o0, o1, branks_mask, fs),
                   _synergy_score(o0, o2, branks_mask, fs),
                   _synergy_score(o1, o2, branks_mask, fs)], axis=1)   # (N,3)
    pick = np.argmax(sy, axis=1)
    obest_fix = np.choose(pick, [scores[3 * N:4 * N], scores[4 * N:5 * N],
                               scores[5 * N:6 * N]])
    w_keep = np.empty((3, N), dtype=np.float64)
    for k in range(3):
        s = scores[k * N:(k + 1) * N]
        w_keep[k] = np.where(s > obest_fix, 1.0, np.where(s == obest_fix, 0.5, 0.0))
    mybest = np.maximum(np.maximum(scores[0:N], scores[N:2 * N]), scores[2 * N:3 * N])
    w_best = np.where(mybest > obest, 1.0, np.where(mybest == obest, 0.5, 0.0))
    return {
        "N": N,
        "w_keep": w_keep,            # (3,N) fixed-keep wins vs realistic opp keep
        "w_best": w_best,            # (N,) per-runout best-keep wins (betting metric)
        "their_pct": _rank_pct(obest),
        "pct3": preflop_pct3_batch(opp3) if LEVEL >= 2 else np.full(N, 0.5),
    }


def build_turn_sample(hole2, board4, d, N, rng):
    hole2 = list(hole2)
    board4 = list(board4)
    known = hole2 + board4 + ([int(d)] if d is not None else [])
    deck = pb._deck_excluding(known)                  # excludes their discard!
    if G6_F:
        # river-complete: enumerate every river card, sample opp holdings
        # per river. Same eval count as plain MC but zero river variance.
        n = deck.size
        per = max(1, N // n)
        opp_l, riv_l = [], []
        for r in deck:
            dk = deck[deck != r]
            opp_l.append(pb._sample_cards(dk, 2, per, rng))
            riv_l.append(np.full((per, 1), r, dtype=np.int8))
        opp2 = np.concatenate(opp_l, axis=0)
        river = np.concatenate(riv_l, axis=0)
        N = opp2.shape[0]
    else:
        nxt = pb._sample_cards(deck, 3, N, rng)           # opp2 + river
        opp2, river = nxt[:, :2], nxt[:, 2:3]
    b4 = np.asarray(board4, dtype=np.int8)
    board5 = np.concatenate([np.tile(b4, (N, 1)), river], axis=1)
    h2 = np.asarray(hole2, dtype=np.int8)
    mine = np.concatenate([np.tile(h2, (N, 1)), board5], axis=1)
    theirs = np.concatenate([opp2, board5], axis=1)
    s_m = pb.eval7_batch(mine)
    s_t = pb.eval7_batch(theirs)
    mywin = np.where(s_m > s_t, 1.0, np.where(s_m == s_t, 0.5, 0.0))
    if LEVEL >= 2 and d is not None:
        explain = discard_explain_weights(opp2, d, board4)
    else:
        explain = np.ones(N)
    p2 = _rank_pct(pair_quality(opp2[:, 0], opp2[:, 1]))
    return {
        "N": N,
        "combos": opp2,
        "mywin": mywin,
        "their_pct": _rank_pct(s_t),
        "p2": p2,
        "explain": explain,
    }


def build_river_sample(hole2, board5, d):
    """Exact enumeration over all C(45,2) opponent holdings."""
    hole2 = list(hole2)
    board5 = list(board5)
    known = hole2 + board5 + ([int(d)] if d is not None else [])
    deck = pb._deck_excluding(known)
    n = deck.size
    if n < 2:
        return None
    i, j = np.triu_indices(n, k=1)
    combos = deck[np.stack([i, j], axis=1)]           # (M,2)
    M = combos.shape[0]
    b5 = np.asarray(board5, dtype=np.int8)
    mine = np.tile(np.asarray(hole2, dtype=np.int8), (M, 1))
    mine = np.concatenate([mine, np.tile(b5, (M, 1))], axis=1)
    theirs = np.concatenate([combos, np.tile(b5, (M, 1))], axis=1)
    s_m = pb.eval7_batch(mine)
    s_t = pb.eval7_batch(theirs)
    mywin = np.where(s_m > s_t, 1.0, np.where(s_m == s_t, 0.5, 0.0))
    if LEVEL >= 2 and d is not None:
        explain = discard_explain_weights(combos, d, board5)
    else:
        explain = np.ones(M)
    p2 = _rank_pct(pair_quality(combos[:, 0], combos[:, 1]))
    return {
        "N": M,
        "combos": combos,
        "mywin": mywin,
        "their_pct": _rank_pct(s_t),
        "p2": p2,
        "explain": explain,
    }


# ----------------------------------------------------------------- leg state
STREETS = ("flop", "turn", "river")


def _new_leg_ctx():
    return {
        "rng": np.random.default_rng(),
        # per-street opp behavior stats (Beta-smoothed when read)
        "st": {s: {"folds": 0, "calls": 0, "raises": 0, "checks": 0, "bets": 0,
                   "faced": 0, "opts": 0} for s in STREETS},
        "pf": {"raises": 0, "opts": 0},
        # calibration EWMAs (None until first sample)
        "show_agg_pf": None, "show_pass_pf": None,
        "show_agg_post": None, "show_pass_post": None,
        "n_show": 0,
        "discard_kept_rate": None,    # fraction of shown hands where their
        "n_disc": 0,                  # discard was the low-synergy card
        # instrumentation
        "replies": [], "last_report": 0, "warns": 0,
        # H5: opponent bet-sizing tell (EWMA of raise-to amount / 200)
        "size_ewma": None, "n_sizes": 0,
    }


def _ewma(old, x, a=0.30):
    return x if old is None else (1.0 - a) * old + a * x


def leg_ctx(tr):
    L = tr.__dict__.get("_syn_l")
    if L is None:
        L = _new_leg_ctx()
        tr._syn_l = L
    return L


def hand_ctx(tr, L):
    S = tr.__dict__.get("_syn_h")
    if S is None or S.get("hand_id") != tr.hand_id:
        S = {"hand_id": tr.hand_id, "flop": None, "turn": None, "river": None}
        tr._syn_h = S
    return S


def on_result(tr, msg):
    """hand_result hook: fold opp-action stats + showdown calibration."""
    L = leg_ctx(tr)
    hist = msg.get("history") or []
    # ---- per-street stats (opponent actions only)
    for s in STREETS:
        street_acts = [h for h in hist if h.get("street") == s]
        if not street_acts:
            continue
        seen_aggr = False
        for h in street_acts:
            if h.get("actor") != "opp":
                if h.get("action") == "raise":
                    seen_aggr = True
                continue
            a = h.get("action")
            d = L["st"][s]
            if a == "fold":
                d["folds"] += 1
                d["faced"] += 1
            elif a == "call":
                d["calls"] += 1
                d["faced"] += 1
            elif a == "raise":
                d["raises"] += 1
                if seen_aggr:
                    d["faced"] += 1
                else:
                    d["bets"] += 1
                    d["opts"] += 1
                seen_aggr = True
                # H5: record their sizing (raise-to total; 200-chip stacks)
                amt = h.get("amount")
                if amt is not None:
                    L["size_ewma"] = _ewma(L["size_ewma"], float(amt) / 200.0, 0.25)
                    L["n_sizes"] += 1
            elif a == "check":
                d["checks"] += 1
                d["opts"] += 1
    # preflop raise rate
    pf = [h for h in hist if h.get("street") == "preflop"]
    opp_pf = [h for h in pf if h.get("actor") == "opp"]
    if opp_pf:
        L["pf"]["opts"] += 1
        if any(h.get("action") == "raise" for h in opp_pf):
            L["pf"]["raises"] += 1
    # ---- showdown calibration
    shown = msg.get("opp_hole_cards")
    if shown and msg.get("opp_discard"):
        L["n_show"] += 1
        try:
            cards = sorted([pb.c2i(c) for c in shown] + [pb.c2i(msg["opp_discard"])])
            p3 = preflop_pct3(cards)
            pf_cls = classify_preflop(hist)
            agg_pf = pf_cls in ("open", "threebet", "fourbet")
            # their shown MADE-HAND strength percentile, free from the cached
            # river sample (correct calibration variable for gain_post)
            strength = None
            S = tr.__dict__.get("_syn_h")
            if S and S.get("river") is not None and "combos" in S["river"]:
                try:
                    ss = sorted(pb.c2i(c) for c in shown)
                    cb = S["river"]["combos"]
                    m = np.where((cb[:, 0] == ss[0]) & (cb[:, 1] == ss[1]))[0]
                    if len(m):
                        strength = float(S["river"]["their_pct"][m[0]])
                except Exception:
                    strength = None
            agg_post = any(h.get("actor") == "opp" and h.get("action") == "raise"
                           and h.get("street") in STREETS for h in hist)
            if agg_pf:
                L["show_agg_pf"] = _ewma(L["show_agg_pf"], p3)
            else:
                L["show_pass_pf"] = _ewma(L["show_pass_pf"], p3)
            if strength is not None:
                if agg_post:
                    L["show_agg_post"] = _ewma(L["show_agg_post"], strength)
                else:
                    L["show_pass_post"] = _ewma(L["show_pass_post"], strength)
        except Exception:
            pass
    # ---- discard-behavior calibration: did they throw away the low-synergy
    # card? (random discarders fail this ~2/3 of the time)
    if shown and msg.get("opp_discard") and msg.get("board"):
        try:
            flop = [pb.c2i(c) for c in msg["board"][:3]]
            kept = [pb.c2i(c) for c in shown]
            dcard = pb.c2i(msg["opp_discard"])
            branks = np.zeros(13, dtype=bool)
            for c in flop:
                branks[c >> 2] = True
            suits = np.bincount([c & 3 for c in flop], minlength=4)
            fs = int(np.argmax(suits)) if suits.max() >= 2 else None
            a = np.int64(kept[0]); b = np.int64(kept[1]); dd = np.int64(dcard)
            s_ab = float(_synergy_score(np.array([a]), np.array([b]), branks, fs)[0])
            s_ad = float(_synergy_score(np.array([a]), np.array([dd]), branks, fs)[0])
            s_bd = float(_synergy_score(np.array([b]), np.array([dd]), branks, fs)[0])
            kept_best = s_ab >= max(s_ad, s_bd) - 0.05
            L["discard_kept_rate"] = _ewma(L["discard_kept_rate"],
                                           1.0 if kept_best else 0.0, a=0.20)
            L["n_disc"] += 1
        except Exception:
            pass


def gains(L):
    """Evidence-driven conditioning strength (0=flat/uniform, 1=full)."""
    if LEVEL < 2:
        return 0.0, 0.0
    a_pf = L["show_agg_pf"]
    p_pf = L["show_pass_pf"]
    a_post = L["show_agg_post"]
    p_post = L["show_pass_post"]
    n = L["n_show"]
    if n >= 4 and a_pf is not None and p_pf is not None:
        corr_pf = a_pf - p_pf
        gain_pf = min(1.0, max(0.0, corr_pf / 0.20))
    else:
        gain_pf = 0.5
    if n >= 4 and a_post is not None and p_post is not None:
        corr_post = a_post - p_post
        gain_post = min(1.0, max(0.0, corr_post / 0.22))
    else:
        gain_post = 0.5
    return gain_pf, gain_post


def explain_gain(L):
    """Evidence that their discard choice is synergy-driven (vs random)."""
    if LEVEL < 2:
        return 0.0
    r = L["discard_kept_rate"]
    if L["n_disc"] >= 5 and r is not None:
        return min(1.0, max(0.0, (r - 0.40) / 0.45))
    return 0.6


def fold_to_bet(L, street):
    d = L["st"][street]
    return (d["folds"] + 1.5) / (d["faced"] + 6.0)      # Beta(1.5,4.5) prior


def bet_freq(L, street):
    d = L["st"][street]
    return (d["bets"] + 1.0) / (d["opts"] + 4.0)


def aggr_stat(L):
    r = c = 0
    for s in STREETS:
        d = L["st"][s]
        r += d["raises"] + d["bets"]
        c += d["calls"] + d["checks"]
    return (r + 2.0) / (r + c + 6.0)


# -------------------------------------------------------------- range weights
def street_theta(L, street):
    base = {"flop": 0.60, "turn": 0.58, "river": 0.55}[street]
    theta = base - 0.40 * (aggr_stat(L) - 0.33)
    a_post = L["show_agg_post"]
    if L["n_show"] >= 4 and a_post is not None:
        theta = 0.5 * theta + 0.5 * a_post
    return min(0.80, max(0.28, theta))


def range_weights(tr, S, street):
    """Importance weights over the cached sample for the CURRENT information."""
    L = leg_ctx(tr)
    gain_pf, gain_post = gains(L)
    # preflop band variable: 3-card percentile on the flop, kept-2 quality on
    # turn/river (their exposed discard fixes the third card)
    pv = S["p2"] if "p2" in S else S["pct3"]
    w = np.ones(S["N"])
    # preflop band
    cls = classify_preflop(tr.history)
    lo, hi = BANDS[cls]
    pf_rate = (L["pf"]["raises"] + 2.0) / (L["pf"]["opts"] + 6.0)
    if cls == "open":
        lo = min(1.0, max(0.20, lo + 0.8 * (0.45 - pf_rate)))
    w = w * band_weights(pv, lo, hi, gain=gain_pf)
    # discard explain (turn/river samples only), evidence-gated
    if "explain" in S:
        eg = explain_gain(L)
        w = w * (1.0 + eg * (S["explain"] - 1.0))
    # street aggression filter: one bet is ~a 3x likelihood ratio, sustained
    # aggression compounds; tau stays modest because the explain factor already
    # carries part of the board-connection evidence (avoid double counting)
    n_aggr = sum(1 for h in tr.history
                 if h.get("actor") == "opp" and h.get("action") == "raise"
                 and h.get("street") in STREETS)
    theta = street_theta(L, street)
    if n_aggr >= 1:
        tau_base = {"flop": 1.6, "turn": 2.0, "river": 2.2}[street]
        tau = tau_base * (0.6 + 0.4 * min(n_aggr, 3))
        # trap read: aggression AFTER passive acts (check/calls before their
        # first bet) is value-heavy — e.g. check-call flop/turn, then leads river
        first_aggr = None
        for i, h in enumerate(tr.history):
            if h.get("actor") == "opp" and h.get("action") == "raise" \
                    and h.get("street") in STREETS:
                first_aggr = i
                break
        if first_aggr is not None:
            n_pass_prior = sum(1 for h in tr.history[:first_aggr]
                               if h.get("actor") == "opp"
                               and h.get("action") in ("check", "call"))
            if n_aggr == 1 and n_pass_prior >= 2:
                tau *= 1.4
        f = np.exp(gain_post * tau * (S["their_pct"] - theta))
        w = w * f
    else:
        n_pass_post = sum(1 for h in tr.history
                          if h.get("actor") == "opp" and h.get("street") in STREETS
                          and h.get("action") in ("check", "call"))
        if n_pass_post > 0:   # only if they actually acted passively
            center = L["show_pass_post"]
            center = 0.42 if center is None or L["n_show"] < 4 else center
            w = w * np.exp(gain_post * 1.2 * (center - S["their_pct"]))
        # else: no postflop information yet -> preflop band only
    tot = w.sum()
    if not np.isfinite(tot) or tot < 1e-9:
        return np.ones(S["N"])
    return w


def blend_equity(eq_c, eq_u, L):
    _, gain_post = gains(L)
    kappa = 0.25 + 0.75 * gain_post
    return (1.0 - kappa) * eq_u + kappa * eq_c


# ------------------------------------------------------------------- budgets
def sample_size(L, tr, kind):
    """Adaptive sample count from live bank fraction."""
    bank0 = max(1.0, float(tr.time_bank_ms))
    f = max(0.0, min(1.0, tr.time_left_ms / bank0))
    if H7:
        # H7: shift ~25% of the flop budget to the turn (bigger pots there)
        table = {
            "flop":  (1200, 980, 720, 510, 300),
            "turn":  (2400, 1900, 1460, 990, 560),
            "disc":  (900, 720, 600, 450, 280),
        }[kind]
    else:
        table = {
            "flop":  (1600, 1300, 960, 680, 400),
            "turn":  (1900, 1500, 1160, 780, 440),
            "disc":  (900, 720, 600, 450, 280),
        }[kind]
    if f > 0.70:
        return table[0]
    if f > 0.48:
        return table[1]
    if f > 0.30:
        return table[2]
    if f > 0.12:
        return table[3]
    return table[4]


def call_vs_bet(L, street):
    d = L["st"][street]
    return (d["calls"] + 2.0) / (d["faced"] + 5.0)      # Beta(2,3) prior ~0.4


def eq_vs_continue(S, w, q):
    """Our equity against the strongest weighted q-fraction of their range
    (i.e. the hands that continue vs a bet). Falls back to overall eq."""
    try:
        order = np.argsort(-S["their_pct"])
        cw = np.cumsum(w[order])
        total = cw[-1]
        if total <= 1e-9:
            return None
        cutoff = total * min(0.95, max(0.05, q))
        sel = np.zeros(len(w), dtype=bool)
        sel[order] = cw <= cutoff
        if sel.sum() < 8:
            return None
        w2 = w * sel
        if "mywin" in S:
            return float((S["mywin"] * w2).sum() / w2.sum())
        # flop: best fixed keep vs their continue range
        return float(((S["w_keep"] * w2).sum(axis=1) / w2.sum()).max())
    except Exception:
        return None


# --------------------------------------------------------------------- equity
def street_equity(tr, req, S, L):
    """(eq_conditioned_blended, eq_uniform, sample, weights)."""
    street = tr.street
    rng = L["rng"]
    if street == "flop":
        if S["flop"] is None:
            N = sample_size(L, tr, "flop")
            S["flop"] = build_flop_sample(tr.hole3, tr.board[:3], N, rng)
        F = S["flop"]
        if LEVEL >= 2:
            w = range_weights(tr, F, "flop")
            # betting metric: symmetric per-runout best-keep (mcbase semantics);
            # the honest fixed-keep means (w_keep) drive the DISCARD choice only
            eq_c = float((F["w_best"] * w).sum() / w.sum())
        else:
            w = np.ones(F["N"])
            eq_c = float(F["w_best"].mean())
        eq_u = float(F["w_best"].mean())
        return blend_equity(eq_c, eq_u, L), eq_u, F, w
    if street == "turn":
        if S["turn"] is None:
            N = sample_size(L, tr, "turn")
            d_use = None if NO_EXCLUDE else tr.opp_discard
            S["turn"] = build_turn_sample(tr.hole2, tr.board[:4], d_use,
                                          N, rng)
        T = S["turn"]
        if LEVEL >= 2:
            w = range_weights(tr, T, "turn")
            eq_c = float((T["mywin"] * w).sum() / w.sum())
        else:
            w = np.ones(T["N"])
            eq_c = float(T["mywin"].mean())
        eq_u = float(T["mywin"].mean())
        return blend_equity(eq_c, eq_u, L), eq_u, T, w
    if street == "river":
        if S["river"] is None:
            d_use = None if NO_EXCLUDE else tr.opp_discard
            if tr.time_left_ms / max(1.0, float(tr.time_bank_ms)) > 0.06:
                S["river"] = build_river_sample(tr.hole2, tr.board[:5], d_use)
            if S["river"] is None:                    # bank nearly dead
                N = 300
                S["river"] = build_turn_sample(tr.hole2, tr.board[:5],
                                               tr.opp_discard, N, L["rng"])
        R = S["river"]
        if LEVEL >= 2:
            w = range_weights(tr, R, "river")
            eq_c = float((R["mywin"] * w).sum() / w.sum())
        else:
            w = np.ones(R["N"])
            eq_c = float(R["mywin"].mean())
        eq_u = float(R["mywin"].mean())
        return blend_equity(eq_c, eq_u, L), eq_u, R, w
    return 0.5, 0.5, None, None


# ---------------------------------------------------------------------- knobs
def knobs(tr, L):
    """Exploit parameters from street-resolved stats + late-leg overlay."""
    if NEUTRAL_ONLY:
        return dict(mc.NEUTRAL)
    P = dict(mc.NEUTRAL)
    fold_all = sum(fold_to_bet(L, s) for s in STREETS) / 3.0
    aggr = aggr_stat(L)
    P["bluff_mult"] = max(0.10, min(2.4, 0.45 + 2.1 * fold_all))
    P["size_mult"] = max(0.80, min(1.35, 0.78 + 0.75 * fold_all))
    P["call_mult"] = max(0.78, min(1.15, 1.14 - 0.45 * aggr))
    P["value_thin"] = max(-0.06, min(0.08, 0.10 * (fold_all - 0.38)))
    P["defend_pct"] = max(0.30, min(0.52, 0.38 + 0.25 * (fold_all - 0.40)))
    P["threebet_pct"] = max(0.78, min(0.94, 0.86 + 0.15 * (aggr - 0.33)))
    pfr = (L["pf"]["raises"] + 2.0) / (L["pf"]["opts"] + 6.0)
    open_w = min(0.50, max(0.30, 0.42 + 0.25 * (0.35 - pfr)))
    P["rfi_pct"] = 1.0 - open_w
    # discard-conditioned read (kept a broadway -> respect)
    if tr.opp_discard is not None and (tr.opp_discard >> 2) >= 8:
        P["bluff_mult"] *= 0.75
    # L3+: late-leg point-max overlay
    if LEVEL >= 3:
        hands_left = max(1, tr.total_hands - tr.hand_id + 1)
        if H9:
            # H9: key the overlay to the capped-delta C, not raw chips
            cap = 400 if tr.total_hands <= 400 else 1000
            lead = float(tr.leg_net)
            if abs(lead) > 0.75 * cap and hands_left <= 30:
                # near-lock: never bluff, pot-control, bank the point
                P["bluff_mult"] = 0.05
                P["value_thin"] += 0.02
                P["call_mult"] *= 1.05
                P["rfi_pct"] = max(0.30, P["rfi_pct"] - 0.10)
            elif abs(lead) < 0.55 * cap and hands_left <= 45:
                scale = min(1.0, (46 - hands_left) / 30.0) * min(2.5, abs(lead) / 120.0)
                if scale > 0.02:
                    if lead < -35:      # behind: raise variance
                        P["bluff_mult"] *= (1.0 + 0.55 * scale)
                        P["size_mult"] *= (1.0 + 0.22 * scale)
                        P["call_mult"] *= (1.0 - 0.08 * scale)
                        P["rfi_pct"] = max(0.30, P["rfi_pct"] - 0.06 * scale)
                    elif lead > 35:     # ahead: damp variance
                        P["bluff_mult"] *= (1.0 - 0.55 * scale)
                        P["size_mult"] *= (1.0 - 0.15 * scale)
                        P["call_mult"] *= (1.0 + 0.05 * scale)
                        P["value_thin"] += 0.02 * scale
        else:
            scale0 = min(1.0, (46 - hands_left) / 30.0) if hands_left <= 45 else 0.0
            lead0 = float(tr.leg_net)
            scale = scale0 * min(2.5, abs(lead0) / 120.0)
            if scale > 0.02:
                if lead0 < -35:      # behind: raise variance
                    P["bluff_mult"] *= (1.0 + 0.55 * scale)
                    P["size_mult"] *= (1.0 + 0.22 * scale)
                    P["call_mult"] *= (1.0 - 0.08 * scale)
                    P["rfi_pct"] = max(0.30, P["rfi_pct"] - 0.06 * scale)
                elif lead0 > 35:     # ahead: damp variance
                    P["bluff_mult"] *= (1.0 - 0.55 * scale)
                    P["size_mult"] *= (1.0 - 0.15 * scale)
                    P["call_mult"] *= (1.0 + 0.05 * scale)
                    P["value_thin"] += 0.02 * scale
    for k in ("bluff_mult", "size_mult", "call_mult"):
        P[k] = float(min(3.0, max(0.05, P[k])))
    if H5 and L["n_sizes"] >= 12 and L["size_ewma"] is not None:
        # H5: small sizings = wide ranges (call looser); overbets = polarized
        # (fold marginal). Maturity-gated at 12 observed sizings.
        f = L["size_ewma"]
        if f < 0.25:
            P["call_mult"] = min(1.10, P["call_mult"] * 1.08)
        elif f > 0.75:
            P["call_mult"] = max(0.85, P["call_mult"] * 0.90)
    if H3:
        # H3: adaptation only with evidence; hard caps against value-burn
        ev = 0
        for s in STREETS:
            d = L["st"][s]
            ev += d["faced"] + d["opts"]
        m = min(1.0, ev / 15.0)
        base = dict(mc.NEUTRAL)
        for k, v0 in list(P.items()):
            if k in base and isinstance(v0, (int, float)) \
                    and isinstance(base[k], (int, float)):
                P[k] = base[k] + (v0 - base[k]) * m
        P["bluff_mult"] = max(0.5, min(1.8, P["bluff_mult"]))
        P["size_mult"] = max(0.85, min(1.25, P["size_mult"]))
        P["call_mult"] = max(0.85, min(1.10, P["call_mult"]))
    return P


# -------------------------------------------------------------------- actions
def _act(req, action, amount=None):
    legal = req["legal_actions"]
    if action not in legal:
        return None
    if action == "raise":
        mr = int(req.get("min_raise", 2))
        xr = int(req.get("max_raise", 200))
        amt = int(round(float(amount)))
        amt = max(mr, min(amt, xr))
        return {"type": "action", "action": "raise", "amount": amt}
    return {"type": "action", "action": action}


def _raise_to(req, target):
    legal = req["legal_actions"]
    if "raise" not in legal:
        return None
    mr = int(req.get("min_raise", 2))
    xr = int(req.get("max_raise", 200))
    amt = int(round(float(target)))
    if amt >= 0.85 * xr:
        amt = xr                                   # clean all-in
    amt = max(mr, min(amt, xr))
    return {"type": "action", "action": "raise", "amount": amt}


def flop_texture_mult(board3):
    ranks = sorted(c >> 2 for c in board3)
    suits = [c & 3 for c in board3]
    sc = np.bincount(suits, minlength=4)
    wet = 0
    if sc.max() >= 2:
        wet += 1
    if len(set(ranks)) < 3:
        wet += 1
    if ranks[-1] - ranks[0] <= 4:
        wet += 1
    return 1.0 + 0.08 * wet


def _count_raises(history, street, actor=None):
    n = 0
    for h in history:
        if h.get("street") == street and h.get("action") == "raise":
            if actor is None or h.get("actor") == actor:
                n += 1
    return n


# ------------------------------------------------------------------ preflop
def preflop_decide(tr, req, P):
    hole3 = tr.hole3
    pct = preflop_pct3(hole3)
    n_raises = sum(1 for h in tr.history if h["action"] == "raise")
    legal = req["legal_actions"]
    to_call = req["to_call"]
    # H1: facing one raise, judge the hand against their RAISING range
    # (top-40% classes). The within-range percentile is mapped back to the
    # uniform percentile of the equally-ranked range hand, so all existing
    # thresholds keep their calibration.
    if H1 and n_raises == 1 and pb.PREFLOP.ok2:
        q = pb.PREFLOP.pct_in_range(hole3)
        if q is not None:
            pct = pb.PREFLOP.range_equiv_pct(q)

    def r6():
        if H10 and rng.random() < 0.35:
            return _raise_to(req, 8)          # H10: mixed open sizes
        return _raise_to(req, 6)

    rng = leg_ctx(tr)["rng"]
    # open cutoff: mcbase parity (0.58) for L1-3; stat-adapted width for L4+
    open_cut = 0.58 if LEVEL <= 3 else (1.0 - P["rfi_pct"])
    bb_opt = 0.45
    if n_raises == 0:
        if to_call == 0:            # BB option after a limp
            if pct > bb_opt and "raise" in legal:
                r = r6()
                if r:
                    return r
            return _act(req, "check")
        if to_call <= 1:            # button facing the big blind
            if pct > open_cut and "raise" in legal:
                r = r6()
                if r:
                    return r
            if pct > 0.30 and "call" in legal:
                return _act(req, "call")
            if "call" in legal and pct > 0.22 and rng.random() < 0.3:
                return _act(req, "call")
            return _act(req, "fold")
    if n_raises == 1:
        if pct > P["threebet_pct"] and "raise" in legal and req.get("max_raise", 0) >= 16:
            if G6_C and rng.random() < 0.5:
                r = _raise_to(req, 20)
                if r:
                    return r
            r = _raise_to(req, 16)
            if r:
                return r
        if G6_A and pb.PREFLOP.ok2:
            # defend vs a raise by direct chip-EV: our equity vs the raising
            # range must beat the pot odds (scaled by call_mult), with a
            # uniform-strength floor so absolute trash never defends
            cid = pb.PREFLOP.class_id(hole3)
            if cid is not None:
                vseq = float(pb.PREFLOP.vseq[cid])
                pot_i = max(1, int(req.get("pot", 3)))
                odds = to_call / max(1.0, pot_i + to_call)
                if vseq > odds * P["call_mult"] and pct > 0.28                         and "call" in legal:
                    return _act(req, "call")
            return _act(req, "fold")
        if pct > 1 - P["defend_pct"] * P["call_mult"]:
            if "call" in legal:
                return _act(req, "call")
        return _act(req, "fold")
    need = 0.72 if req.get("max_raise", 200) < 80 else 0.90
    if pct > 1 - need * P["call_mult"]:
        if "call" in legal:
            return _act(req, "call")
        if "check" in legal:
            return _act(req, "check")
    return _act(req, "fold")


# ----------------------------------------------------------------- postflop
def _mcbase_style_postflop(tr, req, eq, P, L):
    """mcbase parity ladder (A2/A3 policy) on the new engine — L1..L3."""
    legal = req["legal_actions"]
    pot = int(req["pot"])
    to_call = int(req["to_call"])
    rng = L["rng"]
    min_r, max_r = req.get("min_raise", 2), req.get("max_raise", 200)

    def raise_to(frac):
        target = max(req["my_bet"], req["opp_bet"]) + max(2, int(round(frac * (pot + to_call))))
        amt = max(min_r, min(int(target), max_r))
        if "raise" not in legal or amt < min_r or amt > max_r:
            return None
        return {"type": "action", "action": "raise", "amount": amt}

    if to_call == 0:
        bluffs = 0.16 * P["bluff_mult"]
        if eq > 0.68:
            r = raise_to(0.75 * P["size_mult"])
            return r or _act(req, "check")
        if eq > 0.55:
            r = raise_to(0.55 * P["size_mult"])
            return r or _act(req, "check")
        if eq > 0.42 and rng.random() < 0.45 * P["size_mult"]:
            r = raise_to(0.5 * P["size_mult"])
            if r:
                return r
            return _act(req, "check")
        if rng.random() < bluffs:
            r = raise_to(0.6 * P["size_mult"])
            if r:
                return r
        return _act(req, "check")
    required = to_call / max(1.0, pot + to_call)
    need = required * P["call_mult"]
    if eq > 0.82:
        r = raise_to(1.0 * P["size_mult"])
        if r:
            return r
        return _act(req, "call")
    if eq > max(need, 0.36):
        return _act(req, "call")
    if rng.random() < 0.08 * P["bluff_mult"]:
        r = raise_to(1.25 * P["size_mult"])
        if r:
            return r
    return _act(req, "fold") if "fold" in legal else _act(req, "check")


def postflop_decide(tr, req, eq, eq_u, P, L, S, w):
    if LEVEL <= 3:
        return _mcbase_style_postflop(tr, req, eq, P, L)
    street = tr.street
    pot = int(req["pot"])
    to_call = int(req["to_call"])
    rng = L["rng"]
    my_raises = _count_raises(tr.history, street, "me")
    n_raises_street = _count_raises(tr.history, street)
    value_thin = P["value_thin"]
    size_mult = P["size_mult"]

    # value metric: equity vs their CONTINUE range when we have the stats
    # (a bet only has value if it wins vs the hands that call/raise it)
    eq_cont = None
    if S is not None and w is not None:
        q_call = call_vs_bet(L, street)
        eq_cont = eq_vs_continue(S, w, q_call)
    v = eq_cont if eq_cont is not None else eq

    if to_call == 0:
        # -------- we may check or bet
        tex = flop_texture_mult(tr.board[:3]) if street == "flop" else 1.0
        size = size_mult * tex
        if my_raises >= 2 and eq < 0.85:
            return _act(req, "check")                      # cap raising wars
        if eq > 0.90:
            r = _raise_to(req, max(2, int(1.0 * size * pot)))
            return r or _act(req, "check")
        if H4 and street == "river" and my_raises == 0 and v > 0.72 \
                and eq > 0.62 and rng.random() < 0.5:
            r = _raise_to(req, max(2, int(1.3 * pot)))     # H4: river overbet
            if r:
                return r
        if v > 0.60:
            r = _raise_to(req, max(2, int(0.82 * size * pot)))
            return r or _act(req, "check")
        if v > 0.53 and eq > 0.45:
            r = _raise_to(req, max(2, int(0.60 * size * pot)))
            if r:
                return r
            return _act(req, "check")
        if v > 0.505 + value_thin and eq > 0.47:
            p_bet = 1.0 if street != "flop" else 0.75
            if rng.random() < p_bet:
                r = _raise_to(req, max(2, int(0.45 * size * pot)))
                if r:
                    return r
            return _act(req, "check")
        if G6_D and street == "turn" and 0.30 < eq <= 0.40 \
                and _opp_checked_street(tr, "flop") \
                and not any(h.get("actor") == "opp" and h.get("street") == "turn"
                            for h in tr.history) \
                and rng.random() < 0.45:
            r = _raise_to(req, max(2, int(0.40 * pot)))   # G6_D: turn probe
            if r:
                return r
        if eq > 0.40:
            p = {"flop": 0.50, "turn": 0.30, "river": 0.12}[street]
            if fold_to_bet(L, street) > 0.25:
                p *= P["bluff_mult"]
            if G6_B and street == "flop" and _was_pf_aggressor(tr) \
                    and not any(h.get("actor") == "opp" and h.get("street") == "flop"
                                for h in tr.history):
                p = min(0.85, p * 1.35)               # G6_B: c-bet the caller band
            if rng.random() < min(0.8, p):
                r = _raise_to(req, max(2, int(0.50 * size * pot)))
                if r:
                    return r
            return _act(req, "check")
        # pure bluff: fold-EV gated
        fe = fold_to_bet(L, street)
        s = max(2, int(0.60 * size * pot))
        ev_bluff = fe * pot + (1 - fe) * (eq * (pot + 2 * s) - s)
        ev_check = 0.45 * eq * pot
        if ev_bluff > ev_check + 0.5 and fe > 0.22:
            freq = min(0.35, 1.2 * fe) * P["bluff_mult"]
            if rng.random() < freq:
                r = _raise_to(req, s)
                if r:
                    return r
        return _act(req, "check")

    # -------- facing a bet
    need = to_call / max(1.0, pot + to_call)
    floor = {"flop": 0.34, "turn": 0.36, "river": 0.30}[street]
    if eq > 0.80 and n_raises_street <= 2 and "raise" in req["legal_actions"]:
        f = 1.35 if eq > 0.90 else 0.90
        target = req["opp_bet"] + max(2, int(f * size_mult * (pot + to_call)))
        r = _raise_to(req, target)
        if r:
            return r
        return _act(req, "call")
    thresh = max(need * P["call_mult"], floor)
    if n_raises_street >= 3:                    # raising war: tighten
        thresh = max(thresh, 0.60 if street != "river" else 0.55)
        if eq < thresh and eq < 0.72:
            return _act(req, "fold") if "fold" in req["legal_actions"] else _act(req, "check")
    if eq >= thresh:
        return _act(req, "call") if "call" in req["legal_actions"] else _act(req, "check")
    # rare bluff-raise vs foldy opponents
    legal = req["legal_actions"]
    if (street == "river" and eq > 0.28 and rng.random() < 0.05 * P["bluff_mult"]
            and n_raises_street <= 1):
        r = _raise_to(req, req["opp_bet"] + max(2, int(1.25 * pot)))
        if r:
            return r
    return _act(req, "fold") if "fold" in legal else _act(req, "check")


# -------------------------------------------------------------------- discard
def choose_discard(tr, req):
    if LEVEL == 0 or OLD_DISCARD:
        return mc.choose_discard(tr, req)
    L = leg_ctx(tr)
    S = hand_ctx(tr, L)
    rng = L["rng"]
    if S["flop"] is None:
        try:
            N = sample_size(L, tr, "disc")
            S["flop"] = build_flop_sample(tr.hole3, tr.board[:3], N, rng)
        except Exception as e:
            _warn(L, f"discard build failed: {e!r}")
            return mc.choose_discard(tr, req)
    F = S["flop"]
    try:
        if LEVEL >= 2:
            w = range_weights(tr, F, "flop")
            means = (F["w_keep"] * w).sum(axis=1) / w.sum()
        else:
            means = F["w_keep"].mean(axis=1)
    except Exception:
        means = F["w_keep"].mean(axis=1)
    # the honest per-keep means are the discard objective (fixed keep, true EV)
    hole3 = tr.hole3
    order = sorted(range(3), key=lambda k: -means[k])
    # means index KEEP-PAIRS ((0,1),(0,2),(1,2)); keep-pair j discards card 2-j
    best_pair = order[0]
    pick = 2 - best_pair                       # index of the card to DISCARD
    if len(order) > 1 and means[order[1]] > means[order[0]] - 0.020 \
            and rng.random() < 0.40:
        pick = 2 - order[1]
    elif len(order) > 2 and means[order[2]] > means[order[0]] - 0.010 \
            and rng.random() < 0.15:
        pick = 2 - order[2]
    return {"type": "discard", "card": pb.i2c(hole3[pick])}


# -------------------------------------------------------------- instrumentation
def _warn(L, msg):
    if L["warns"] < 25:
        L["warns"] += 1
        try:
            print(f"[synergy] {msg}", file=sys.stderr)
        except Exception:
            pass


def _record_time(L, dt_ms, tr):
    L["replies"].append(dt_ms)
    if len(L["replies"]) >= 50:
        arr = np.array(L["replies"])
        L["replies"] = []
        L["last_report"] += 50
        try:
            print(f"[synergy] t50 hand={tr.hand_id} mean={arr.mean():.1f}ms "
                  f"p95={np.percentile(arr, 95):.0f}ms max={arr.max():.0f}ms "
                  f"bank={tr.time_left_ms / 1000:.1f}s", file=sys.stderr)
        except Exception:
            pass


# --------------------------------------------------------------------- decide
def decide(tr, req):
    t0 = time.monotonic()
    L = leg_ctx(tr)
    try:
        if LEVEL == 0:
            if om is not None:
                return om.decide(tr, req)
            return mc.decide(tr, req, dict(mc.NEUTRAL))
        street = tr.street
        if street == "preflop":
            P = knobs(tr, L)
            reply = preflop_decide(tr, req, P)
        else:
            S = hand_ctx(tr, L)
            eq, eq_u, Smp, wgt = street_equity(tr, req, S, L)
            P = knobs(tr, L)
            if DEBUG and Smp is not None:
                try:
                    n_aggr = sum(1 for h in tr.history
                                 if h.get("actor") == "opp" and h.get("action") == "raise"
                                 and h.get("street") in STREETS)
                    gpf, gpo = gains(L)
                    ws = wgt / max(1e-9, wgt.sum())
                    top = float(ws[np.argsort(-Smp["their_pct"])[:max(1, Smp["N"] // 10)]].sum())
                    print(f"[dbg] h{tr.hand_id} {tr.street} eq={eq:.3f} u={eq_u:.3f} "
                          f"cls={classify_preflop(tr.history)} nag={n_aggr} "
                          f"g=({gpf:.2f},{gpo:.2f}) top10%mass={top:.2f} "
                          f"to_call={req['to_call']}", file=sys.stderr)
                except Exception as e:
                    print(f"[dbg] err {e!r}", file=sys.stderr)
            reply = postflop_decide(tr, req, eq, eq_u, P, L, Smp, wgt)
        if reply is None:
            reply = pb.fallback_action(req)
    except SystemExit:
        raise
    except Exception as e:
        _warn(L, f"decide error: {e!r}")
        try:
            reply = mc.decide(tr, req, dict(mc.NEUTRAL))
        except Exception:
            reply = pb.fallback_action(req)
    _record_time(L, (time.monotonic() - t0) * 1000.0, tr)
    return reply


def discard(tr, req):
    t0 = time.monotonic()
    L = leg_ctx(tr)
    try:
        reply = choose_discard(tr, req)
        if reply is None:
            reply = {"type": "discard", "card": pb.default_discard_str(tr.hole3)}
    except SystemExit:
        raise
    except Exception as e:
        _warn(L, f"discard error: {e!r}")
        reply = {"type": "discard", "card": pb.default_discard_str(tr.hole3)}
    _record_time(L, (time.monotonic() - t0) * 1000.0, tr)
    return reply
