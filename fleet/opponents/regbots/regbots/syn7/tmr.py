"""tmr.py — Gen7 white-box module: Their-Model Replication + EDI + EV chooser.

Measured basis (probe experiments M1-M5, ~2000 hands vs the reference
ladder-bot class):
  * reference postflop decisions = scalar eq (P(win vs its importance-weighted
    model of OUR range)) compared to pot-odds thresholds; bet size never
    enters its range model; floors {flop .34, turn .36, river .30};
    raise branch at eq > .80 (size .9/1.35x); 5% river bluff-raises;
  * reference preflop: never 4bets; calls 3bets 134/134 and jams 36/36;
  * reference discard: argmax keep-EV, 869/869 showdown-consistent.

Module contents:
  MirrorL      — replica of the reference leg-stats, public inputs only
  EDI          — exact Bayesian inversion of the discard policy
  build_model  — their-candidate posterior + THEIR eq estimate per candidate
  predict      — P(fold/call/raise) for a bet we are considering
  choose_river / choose_turn — EV chooser over our candidate actions
  preflop_exploit — measured counter-policy vs the reference preflop holes
  FP           — fingerprint tracker (is the opponent reference-class?)

Stdlib + NumPy only. All public-information based.
"""
import os
import numpy as np

import pbcore as pb

# ---------------- reference-class constants (g4-lineage, measured) ----------
REF_BANDS = {
    "limped":   (0.02, 0.58),
    "caller":   (0.18, 0.90),
    "open":     (0.50, 1.02),
    "threebet": (0.70, 1.02),
    "fourbet":  (0.78, 1.02),
    None:       (0.02, 1.02),
}
REF_FLOOR = {"flop": 0.34, "turn": 0.36, "river": 0.30}
EDI_TIE1, EDI_TIE2 = 0.020, 0.010
EDI_MIX1, EDI_MIX2 = 0.40, 0.15

DEBUG = os.environ.get("TMR_DEBUG", "") == "1"


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _rank_pct(scores):
    n = len(scores)
    if n < 2:
        return np.full(max(n, 0), 0.5)
    return np.argsort(np.argsort(scores)).astype(np.float64) / (n - 1)


# ============================ reference replicas ============================

def _synergy_score(a, b, branks_mask, flush_suit):
    ra, rb = (a >> 2).astype(np.int64), (b >> 2).astype(np.int64)
    sa, sb = (a & 3).astype(np.int64), (b & 3).astype(np.int64)
    s = np.zeros(len(a), dtype=np.float64)
    s += branks_mask[ra] * 1.0 + branks_mask[rb] * 1.0
    s += (ra == rb) * 1.2
    s += (sa == sb) * 0.4
    if flush_suit is not None:
        s += (sa == flush_suit) * 0.7 + (sb == flush_suit) * 0.7
    s += (np.maximum(ra, rb) >= 9) * 0.3 + ((ra >= 9) & (rb >= 9)) * 0.5
    s += (np.abs(ra - rb) <= 2) * 0.25
    return s


def ref_explain_weights(combos2, d, board):
    """Replica of the reference discard-explain factor."""
    if d is None or len(combos2) == 0:
        return np.ones(max(1, len(combos2)))
    branks_mask = np.zeros(13, dtype=bool)
    for c in board:
        branks_mask[c >> 2] = True
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
    if dr >= 9:
        w *= 0.80 + 0.40 * (s_ab / (s_ab.max() + 1e-9))
    elif dr <= 5:
        hi = np.maximum(a >> 2, b >> 2)
        w *= 0.85 + 0.30 * ((hi >= 9) | ((a >> 2) == (b >> 2)))
    return w * 0.85


def ref_band_weights(pct, lo, hi, gain=1.0, sigma=0.14):
    w = _sig((pct - lo) / sigma) * _sig((hi - pct) / sigma)
    return 1.0 + gain * (w - 1.0)


def classify_pf(history, who):
    """Classify `who`'s preflop line ('me' or 'opp') from a full history."""
    pf = [h for h in history if h.get("street") == "preflop"]
    if not pf:
        return None
    raises = [i for i, h in enumerate(pf) if h.get("action") == "raise"]
    who_raise = {i for i, h in enumerate(pf)
                 if h.get("actor") == who and h.get("action") == "raise"}
    if not raises:
        return "limped"
    if not who_raise:
        return "caller"
    if len(who_raise) >= 2:
        return "fourbet"
    if raises[0] in who_raise:
        return "open"
    return "threebet"


_SORTED_EQ = np.sort(pb.PREFLOP.equity) if pb.PREFLOP.ok else None


def pct3_of(three):
    """Percentile of one 3-card holding (list of ints)."""
    if _SORTED_EQ is None:
        return 0.5
    cid = pb.PREFLOP.class_id(sorted(int(x) for x in three))
    if cid is None:
        return 0.5
    return float(np.searchsorted(_SORTED_EQ, pb.PREFLOP.equity[cid]) /
                 max(1, len(_SORTED_EQ)))


def pct3_batch(three):
    if _SORTED_EQ is None:
        return np.full(len(three), 0.5)
    lut = pb.PREFLOP.class_of_hand
    eq = pb.PREFLOP.equity
    out = np.empty(len(three), dtype=np.float64)
    for i in range(len(three)):
        r = three[i]
        key = tuple(sorted((int(r[0]), int(r[1]), int(r[2]))))
        cid = lut.get(key)
        out[i] = eq[cid] if cid is not None else 0.5
    return np.searchsorted(_SORTED_EQ, out) / max(1, len(_SORTED_EQ))


# ============================== MirrorL ====================================

class MirrorL:
    """Replica of the reference bot's leg statistics OF US (public inputs).
    Used to replicate their eq estimate and their adaptive knobs."""

    def __init__(self):
        self.st = {s: {"folds": 0, "calls": 0, "raises": 0, "checks": 0,
                       "bets": 0, "faced": 0, "opts": 0}
                   for s in ("flop", "turn", "river")}
        self.pf = {"raises": 0, "opts": 0}
        self.show_agg_pf = None
        self.show_pass_pf = None
        self.show_agg_post = None
        self.show_pass_post = None
        self.n_show = 0
        self.pred_hit = 0
        self.pred_tot = 0

    @staticmethod
    def _ewma(old, x, a=0.30):
        return x if old is None else (1.0 - a) * old + a * x

    def observe_showdown(self, history, our_shown2, our_discard, board5):
        """Update their showdown calibration when WE show cards.
        our_shown2: our 2 kept cards (ints); our_discard: our exposed discard.
        """
        try:
            if our_shown2 is None or len(our_shown2) < 2:
                return
            if our_discard is not None:
                cards = sorted([int(c) for c in our_shown2] + [int(our_discard)])
                p3 = pct3_of(cards)
            else:
                p3 = 0.5
            agg_pf = classify_pf(history, "me") in ("open", "threebet", "fourbet")
            # strength percentile of our shown hand vs a uniform candidate
            # grid (their river sample is the exact uniform enumeration)
            excl = [int(c) for c in board5] + [int(c) for c in our_shown2]
            deck = pb._deck_excluding(excl)
            n = deck.size
            if n >= 2:
                ii, jj = np.triu_indices(n, k=1)
                cand = deck[np.stack([ii, jj], axis=1)]
                b5 = np.asarray(board5, dtype=np.int8)
                rows = np.concatenate([cand, np.tile(b5, (len(cand), 1))], axis=1)
                s_c = pb.eval7_batch(rows).astype(np.float64)
                ours = np.concatenate(
                    [np.tile(np.asarray([int(c) for c in our_shown2],
                                        dtype=np.int8), (len(cand), 1)),
                     np.tile(b5, (len(cand), 1))], axis=1)
                s_o = pb.eval7_batch(ours).astype(np.float64)[0]
                strength = float((s_c < s_o).mean())
            else:
                strength = 0.5
            agg_post = any(h.get("actor") == "me" and h.get("action") == "raise"
                           and h.get("street") in ("flop", "turn", "river")
                           for h in history)
            self.n_show += 1
            if agg_pf:
                self.show_agg_pf = self._ewma(self.show_agg_pf, p3)
            else:
                self.show_pass_pf = self._ewma(self.show_pass_pf, p3)
            if agg_post:
                self.show_agg_post = self._ewma(self.show_agg_post, strength)
            else:
                self.show_pass_post = self._ewma(self.show_pass_post, strength)
        except Exception:
            pass

    def gains(self):
        """Replica of their evidence-driven conditioning strengths."""
        n = self.n_show
        if n >= 4 and self.show_agg_pf is not None and self.show_pass_pf is not None:
            gain_pf = min(1.0, max(0.0, (self.show_agg_pf - self.show_pass_pf) / 0.20))
        else:
            gain_pf = 0.5
        if n >= 4 and self.show_agg_post is not None and self.show_pass_post is not None:
            gain_post = min(1.0, max(0.0,
                                     (self.show_agg_post - self.show_pass_post) / 0.22))
        else:
            gain_post = 0.5
        return gain_pf, gain_post

    def blend_kappa(self):
        return 0.25 + 0.75 * self.gains()[1]

    def observe_hand(self, history):
        """Update stats of US (actor 'me') from one completed hand history."""
        for s in ("flop", "turn", "river"):
            acts = [h for h in history if h.get("street") == s]
            if not acts:
                continue
            seen_aggr = False
            for h in acts:
                if h.get("actor") != "me":
                    if h.get("action") == "raise":
                        seen_aggr = True
                    continue
                a = h.get("action")
                d = self.st[s]
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
                elif a == "check":
                    d["checks"] += 1
                    d["opts"] += 1
        pf = [h for h in history if h.get("street") == "preflop"]
        my_pf = [h for h in pf if h.get("actor") == "me"]
        if my_pf:
            self.pf["opts"] += 1
            if any(h.get("action") == "raise" for h in my_pf):
                self.pf["raises"] += 1

    def fold_to_bet(self, street):
        d = self.st[street]
        return (d["folds"] + 1.5) / (d["faced"] + 6.0)

    def call_vs_bet(self, street):
        d = self.st[street]
        return (d["calls"] + 2.0) / (d["faced"] + 5.0)

    def bet_freq(self, street):
        d = self.st[street]
        return (d["bets"] + 1.0) / (d["opts"] + 4.0)

    def aggr_stat(self):
        r = c = 0
        for s in ("flop", "turn", "river"):
            d = self.st[s]
            r += d["raises"] + d["bets"]
            c += d["calls"] + d["checks"]
        return (r + 2.0) / (r + c + 6.0)

    def call_mult(self):
        return min(1.15, max(0.78, 1.14 - 0.45 * self.aggr_stat()))

    def bluff_mult(self):
        fold_all = sum(self.fold_to_bet(s) for s in ("flop", "turn", "river")) / 3.0
        return max(0.10, min(2.4, 0.45 + 2.1 * fold_all))

    def size_mult(self):
        fold_all = sum(self.fold_to_bet(s) for s in ("flop", "turn", "river")) / 3.0
        return max(0.80, min(1.35, 0.78 + 0.75 * fold_all))

    def pred_acc(self):
        return self.pred_hit / max(1, self.pred_tot)


# ================================ EDI ======================================

def edi_weights(cand, d, board3, our_hole3, rng, n_run=8):
    """P(they would discard d | their 3 cards = (cand_i, d)) — exact policy
    inversion via shared-runout replication of their keep-EV argmax.

    cand: (M,2) their candidate kept-pairs (int8 card ints)
    d: their exposed discard (int)
    board3: flop (3 ints)
    our_hole3: OUR 3 cards at flop time (ints) — for deck exclusion
    """
    try:
        M = len(cand)
        if M == 0 or d is None:
            return None
        known = list(board3) + [int(d)] + [int(c) for c in our_hole3]
        deck = pb._deck_excluding(known)
        m = deck.size
        if m < 5:
            return None
        # shared runouts: our3-proxy + turn + river  (5 cards each)
        nxt = pb._sample_cards(deck, 5, n_run, rng)
        our3p, run = nxt[:, 0:3], nxt[:, 3:5]
        b3 = np.asarray(board3, dtype=np.int8)
        # our realistic keep per runout: argmax synergy pair of our3-proxy
        branks_mask = np.zeros(13, dtype=bool)
        for c in board3:
            branks_mask[c >> 2] = True
        suits = np.bincount([c & 3 for c in board3], minlength=4)
        fs = int(np.argmax(suits)) if suits.max() >= 2 else None
        o0, o1, o2 = (our3p[:, 0].astype(np.int64), our3p[:, 1].astype(np.int64),
                      our3p[:, 2].astype(np.int64))
        sy = np.stack([_synergy_score(o0, o1, branks_mask, fs),
                       _synergy_score(o0, o2, branks_mask, fs),
                       _synergy_score(o1, o2, branks_mask, fs)], axis=1)
        pick = np.argmax(sy, axis=1)
        R = n_run
        keep_pairs = ((0, 1), (0, 2), (1, 2))
        our_keep = np.empty((R, 2), dtype=our3p.dtype)
        for j, (c1, c2) in enumerate(keep_pairs):
            mrow = pick == j
            our_keep[mrow, 0] = our3p[mrow, c1]
            our_keep[mrow, 1] = our3p[mrow, c2]

        # board5 per runout
        board5 = np.concatenate([np.tile(b3, (n_run, 1)), run], axis=1)

        # their keep options: keep(cand), keep(cand0,d), keep(cand1,d)
        a = cand[:, 0].astype(np.int64)
        b = cand[:, 1].astype(np.int64)
        dd = np.full(M, int(d), dtype=np.int64)
        opts = [(a, b), (a, dd), (b, dd)]
        evs = np.empty((3, M), dtype=np.float64)
        for k, (x, y) in enumerate(opts):
            rows = np.empty((M * R, 7), dtype=np.int8)
            for r in range(R):
                blk = rows[r * M:(r + 1) * M, :]
                blk[:, 0] = x
                blk[:, 1] = y
                blk[:, 2:7] = board5[r]
            s_t = pb.eval7_batch(rows)
            # vs our keep over the same runouts
            rows2 = np.empty((R, 7), dtype=np.int8)
            rows2[:, 0:2] = our_keep
            rows2[:, 2:7] = board5
            s_o = pb.eval7_batch(rows2)
            s_t = s_t.reshape(M, R)
            win = np.where(s_t > s_o[None, :], 1.0,
                           np.where(s_t == s_o[None, :], 0.5, 0.0))
            evs[k] = win.mean(axis=1)
        # their pick logic: order by keep-EV; near-tie mixing
        e0, e1, e2 = evs[0], evs[1], evs[2]      # e0 = keep(cand)
        order = np.argsort(-evs, axis=0)          # (3, M) indices
        best = order[0]
        second = order[1]
        third = order[2]
        gap12 = evs[best, np.arange(M)] - evs[second, np.arange(M)]
        gap13 = evs[best, np.arange(M)] - evs[third, np.arange(M)]
        w = np.zeros(M, dtype=np.float64)
        is_best = best == 0
        is_second = second == 0
        is_third = third == 0
        w[is_best] = np.where(gap12[is_best] <= EDI_TIE1, 0.60,
                              np.where(gap13[is_best] <= EDI_TIE2, 0.85, 1.0))
        w[is_second] = np.where(gap12[is_second] <= EDI_TIE1, EDI_MIX1, 0.0)
        w[is_third] = np.where((gap12[is_third] > EDI_TIE1) &
                               (gap13[is_third] <= EDI_TIE2), EDI_MIX2, 0.0)
        return np.maximum(w, 1e-3)
    except Exception:
        return None


# ======================== their-eq replication =============================

def apparent_weights(tr, mirror, g, s_g, street, board, d_us):
    """Replicate their range_weights over OUR candidate grid g (public).

    g: (GN,2) our apparent 2-card combos; s_g: their eval strengths.
    """
    try:
        GN = len(g)
        w = np.ones(GN)
        gain_pf, gain_post = mirror.gains()
        # preflop band on our 2-card pair-quality rank (their p2 approach)
        p2 = _rank_pct(ref_pair_quality(g[:, 0], g[:, 1]))
        cls = classify_pf(tr.history, "me")
        lo, hi = REF_BANDS.get(cls, REF_BANDS[None])
        w = w * ref_band_weights(p2, lo, hi, gain=gain_pf)
        # discard explain on OUR discard (evidence-gated like the reference)
        if d_us is not None:
            eg = 0.6
            w = w * (1.0 + eg * (ref_explain_weights(g, int(d_us), board) - 1.0))
        # aggression filter: OUR postflop raises this hand, scaled by THEIR
        # measured evidence gain (calibration!)
        n_aggr = sum(1 for h in tr.history
                     if h.get("actor") == "me" and h.get("action") == "raise"
                     and h.get("street") in ("flop", "turn", "river"))
        theta = 0.55 - 0.40 * (mirror.aggr_stat() - 0.33)
        theta = min(0.80, max(0.28, theta))
        their_pct = _rank_pct(s_g)
        if n_aggr >= 1:
            tau = {"flop": 1.6, "turn": 2.0, "river": 2.2}[street]
            tau = tau * (0.6 + 0.4 * min(n_aggr, 3))
            w = w * np.exp(gain_post * tau * (their_pct - theta))
        else:
            n_pass = sum(1 for h in tr.history
                         if h.get("actor") == "me"
                         and h.get("street") in ("flop", "turn", "river")
                         and h.get("action") in ("check", "call"))
            if n_pass > 0:
                w = w * np.exp(gain_post * 1.2 * (0.42 - their_pct))
        tot = w.sum()
        if not np.isfinite(tot) or tot < 1e-9:
            return np.ones(GN)
        return w / tot
    except Exception:
        return None


def ref_pair_quality(a, b):
    ra, rb = (a >> 2).astype(np.float64), (b >> 2).astype(np.float64)
    hi = np.maximum(ra, rb)
    q = hi + (ra == rb) * (hi + 6.0) + ((a & 3) == (b & 3)) * 2.0 \
        + np.clip(4.0 - np.abs(ra - rb), 0.0, 4.0) * 0.5
    return q


def eq_theirs_grid(tr, mirror, L, street, board, hole2, d_us, cand, rng):
    """THEIR eq estimate for each of THEIR candidates (M,) — replication.

    eq(h) = (1-k) * uniform-CDF + k * weighted-CDF of our apparent-range
    strengths at h's strength.  River: exact; turn: mean over sampled rivers.
    """
    try:
        excl = list(board) + list(hole2)
        if d_us is not None:
            excl.append(int(d_us))
        deck = pb._deck_excluding(excl)
        n = deck.size
        if n < 2 or len(cand) == 0:
            return None
        GN = min(150, n * (n - 1) // 2)
        idx = rng.integers(0, n, size=(int(GN * 1.3), 2))
        idx = idx[idx[:, 0] != idx[:, 1]][:GN]
        if len(idx) < 40:
            return None
        g = deck[idx]

        def cdf_eq(s_g_sorted, w_sorted, s_h):
            """weighted CDF of grid strengths at candidate strengths."""
            cum = np.cumsum(w_sorted)
            tot = cum[-1] if len(cum) else 0.0
            if tot <= 1e-12:
                return np.full(np.shape(s_h), 0.5)
            pos = np.searchsorted(s_g_sorted, s_h, side="right")
            lo = np.searchsorted(s_g_sorted, s_h, side="left")
            below = np.where(pos > 0,
                             cum[np.minimum(pos - 1, len(cum) - 1)], 0.0)
            tie_lo = np.where(lo > 0, cum[np.minimum(lo - 1, len(cum) - 1)], 0.0)
            tie_mass = np.maximum(0.0, below - tie_lo)
            return np.clip((below + 0.5 * tie_mass) / tot, 0.0, 1.0)

        if street == "river":
            b5 = np.asarray(board, dtype=np.int8)
            rows_g = np.concatenate([g, np.tile(b5, (len(g), 1))], axis=1)
            rows_h = np.concatenate([cand, np.tile(b5, (len(cand), 1))], axis=1)
            s_g = pb.eval7_batch(rows_g).astype(np.float64)
            s_h = pb.eval7_batch(rows_h).astype(np.float64)
            w_g = apparent_weights(tr, mirror, g, s_g, street, board, d_us)
            if w_g is None:
                return None
            o = np.argsort(s_g)
            eq_c = cdf_eq(s_g[o], w_g[o], s_h)
            eq_u = cdf_eq(s_g[o], np.ones(len(s_g))[o], s_h)
            k = mirror.blend_kappa()
            return (1 - k) * eq_u + k * eq_c

        # ---------------- turn: average over sampled rivers -------------
        K = 4
        b4 = np.asarray(board[:4], dtype=np.int8)
        ridx = rng.integers(0, n, size=(K,))
        rivers = deck[ridx]
        eqs = []
        for rv in rivers:
            b5 = np.concatenate([np.tile(b4, (len(g), 1)),
                                 np.full((len(g), 1), rv, dtype=np.int8)], axis=1)
            rows_g = np.concatenate([g, b5], axis=1)
            b5h = np.concatenate([np.tile(b4, (len(cand), 1)),
                                  np.full((len(cand), 1), rv, dtype=np.int8)], axis=1)
            rows_h = np.concatenate([cand, b5h], axis=1)
            s_g = pb.eval7_batch(rows_g).astype(np.float64)
            s_h = pb.eval7_batch(rows_h).astype(np.float64)
            w_g = apparent_weights(tr, mirror, g, s_g, street, board[:4], d_us)
            if w_g is None:
                continue
            o = np.argsort(s_g)
            eqs.append(cdf_eq(s_g[o], w_g[o], s_h) * mirror.blend_kappa() +
                       cdf_eq(s_g[o], np.ones(len(s_g))[o], s_h) *
                       (1 - mirror.blend_kappa()))
        if not eqs:
            return None
        return np.mean(eqs, axis=0)
    except Exception:
        return None


# ===================== response prediction + EV ============================

def predict_response(model, mirror, bet, pot, n_raises_street, street):
    """P(their fold/call/raise) if we bet `bet` chips into `pot`.

    bet: our bet size in chips (raise-to minus our current street bet).
    pot: pot INCLUDING both players' current street commitments.
    """
    eq_t = model["eq_t"]
    w_h = model["w_h"]
    cm = mirror.call_mult()
    bm = mirror.bluff_mult()
    need = bet / max(1.0, pot + bet)
    thresh = max(need * cm, REF_FLOOR[street])
    p_fold_raw = float((w_h * (eq_t < thresh)).sum())
    p_raise_hi = float((w_h * (eq_t > 0.80)).sum())
    br = 0.05 * bm
    p_bluffraise = min(0.25, br) * float((w_h * ((eq_t > 0.28) &
                                                  (eq_t < thresh))).sum())
    p_f = max(0.0, p_fold_raw - p_bluffraise)
    p_r = min(0.95, p_raise_hi + p_bluffraise)
    p_c = max(0.0, 1.0 - p_f - p_r)
    return p_f, p_c, p_r


def _cond_eq(model, mask):
    w = model["w_h"] * mask
    tot = w.sum()
    if tot < 1e-9:
        return 0.5
    return float((w * model["mywin"]).sum() / tot)


def bet_ev(model, mirror, bet, pot, street, n_raises_street, my_bet,
           max_raise, min_raise):
    """EV of betting `bet` chips (bluff or value; our hand enters via
    model['mywin']).  Returns (ev, p_f, p_c, p_r) or None."""
    eq_t, w_h = model["eq_t"], model["w_h"]
    cm = mirror.call_mult()
    need = bet / max(1.0, pot + bet)
    thresh = max(need * cm, REF_FLOOR[street])
    p_f, p_c, p_r = predict_response(model, mirror, bet, pot,
                                     n_raises_street, street)
    # our equity vs their calling band
    call_mask = (eq_t >= thresh) & (eq_t <= 0.80)
    eq_call = _cond_eq(model, call_mask)
    # their raise: we fold bluffs, continue with strong hands
    raise_mask = eq_t > 0.80
    eq_raise = _cond_eq(model, raise_mask)
    # their raise-to estimate: opp_bet(=bet) + 0.9 * size_mult * (pot+bet)
    sm = mirror.size_mult()
    R = bet + max(2, int(0.9 * sm * (pot + bet)))
    R = min(R, 200)
    if eq_raise > R / max(1.0, pot + 2 * R):        # +EV call of their raise
        ev_r = eq_raise * (pot + 2 * R) - R
    else:
        ev_r = -bet                                  # fold to the raise
    ev = (p_f * pot + p_c * (eq_call * (pot + 2 * bet) - bet)
          + p_r * ev_r)
    return ev, p_f, p_c, p_r


def check_ev(model, mirror, pot, street):
    """EV of checking (showdown value + facing their probable bet)."""
    eq_t, w_h = model["eq_t"], model["w_h"]
    eq_full = _cond_eq(model, np.ones_like(eq_t))
    # their betting probability when checked to (v-ladder approx)
    beta = float((w_h * (eq_t > 0.53)).sum())
    beta = min(0.92, beta + 0.10)
    bet_mask = eq_t > 0.53
    eq_bet = _cond_eq(model, bet_mask)
    b = 0.6 * pot
    if eq_bet * (pot + 2 * b) - b > 0:
        ev_face = eq_bet * (pot + 2 * b) - b
    else:
        ev_face = 0.0
    return (1 - beta) * eq_full * pot + beta * ev_face, eq_full


def choose_bet(model, mirror, req, street):
    """EV chooser when to_call == 0.  Returns an action dict or None."""
    pot = int(req["pot"])
    my_bet = int(req["my_bet"])
    legal = req["legal_actions"]
    if "raise" not in legal:
        return None
    mr = int(req.get("min_raise", 2))
    xr = int(req.get("max_raise", 200))
    n_raises_street = sum(1 for h in (req.get("history") or [])
                          if h.get("street") == street
                          and h.get("action") == "raise")
    ev_check, eq_full = check_ev(model, mirror, pot, street)
    best = ("check", ev_check)
    cands = []
    for frac in (0.35, 0.5, 0.7, 0.9, 1.2, 1.6, 2.1):
        b = int(round(frac * pot))
        b = max(2, min(b, xr - my_bet))
        if b < mr - my_bet:
            continue
        r = bet_ev(model, mirror, b, pot, street, n_raises_street,
                   my_bet, xr, mr)
        if r is None:
            continue
        ev, p_f, p_c, p_r = r
        cands.append((b, ev, p_f, p_c, p_r))
    for b, ev, p_f, p_c, p_r in cands:
        if ev > best[1] + 0.35:                 # hysteresis toward checking
            best = (b, ev)
    if best[0] == "check":
        return {"type": "action", "action": "check"}
    b = int(best[0])
    amt = my_bet + b
    amt = max(mr, min(amt, xr))
    if amt >= 0.85 * xr:
        amt = xr
    return {"type": "action", "action": "raise", "amount": int(amt)}


def apply_bet_likelihood(model, mirror, req, street):
    """Update their posterior by the likelihood that they would make the
    observed bet, under the reference ladder (value-heavy betting policy).

    Their bet branch: bets with v>0.505..0.60+ (prob ~1), semi-bluffs with
    eq>0.40 (p~0.12 river / 0.3 turn), pure bluffs EV-gated (p<=0.35*bm).
    """
    try:
        pot = int(req["pot"])
        my_bet = int(req["my_bet"])
        opp_bet = int(req["opp_bet"])
        to_call = int(req["to_call"])
        if to_call <= 0 or opp_bet <= my_bet:
            return None
        eq_t = model["eq_t"]
        w_h = np.asarray(model["w_h"], dtype=np.float64).copy()
        bet_size = opp_bet - my_bet
        bm = mirror.bluff_mult()
        semi = 0.12 if street == "river" else 0.30
        bluff_p = min(0.35, 0.5 * bm) * 0.5
        like = np.where(eq_t > 0.505,
                        1.0,
                        np.where(eq_t > 0.40, semi, bluff_p))
        like = np.maximum(like, 0.05)
        w_h = w_h * like
        tot = w_h.sum()
        if tot < 1e-9:
            return None
        m2 = dict(model)
        m2["w_h"] = w_h / tot
        return m2
    except Exception:
        return None


def choose_facing(model, mirror, req, street):
    """EV chooser when facing their bet: fold / call / raise."""
    pot = int(req["pot"])
    to_call = int(req["to_call"])
    my_bet = int(req["my_bet"])
    opp_bet = int(req["opp_bet"])
    legal = req["legal_actions"]
    # --- update their posterior by their bet (policy likelihood)
    eq_t = model["eq_t"]
    w_h = model["w_h"].copy()
    bet_size = opp_bet - my_bet          # their bet in chips (approx)
    if bet_size > 0:
        need_them = bet_size / max(1.0, pot)     # their pot odds when betting
        # they bet with v>0.505..0.60+ mostly value; weak eq rarely bets
        like = 0.12 + 0.88 * (eq_t > max(0.47, 0.53 - 0.06 * (pot > 30)))
        w_h = w_h * like
        tot = w_h.sum()
        if tot < 1e-9:
            w_h = model["w_h"].copy()
        else:
            w_h = w_h / tot
    model2 = dict(model)
    model2["w_h"] = w_h
    eq_now = _cond_eq(model2, np.ones_like(eq_t))
    ev_call = eq_now * (pot + to_call) - to_call
    ev_fold = 0.0
    # --- raise option (value-heavy): TMR response to our raise
    ev_raise = None
    if "raise" in legal:
        mr = int(req.get("min_raise", 2))
        xr = int(req.get("max_raise", 200))
        n_raises_street = sum(1 for h in (req.get("history") or [])
                              if h.get("street") == street
                              and h.get("action") == "raise")
        for frac in (0.9, 1.4):
            add = max(2, int(frac * (pot + to_call)))
            amt = opp_bet + add
            amt = max(mr, min(amt, xr))
            bet = amt - my_bet
            r = bet_ev(model2, mirror, bet, pot, street, n_raises_street,
                       my_bet, xr, mr)
            if r is None:
                continue
            if ev_raise is None or r[0] > ev_raise:
                ev_raise = r[0]
                best_raise_amt = amt
    opts = [("call", ev_call), ("fold", ev_fold)]
    if ev_raise is not None and "raise" in legal:
        opts.append(("raise", ev_raise))
    opts.sort(key=lambda x: -x[1])
    choice = opts[0][0]
    if choice == "call" and "call" not in legal:
        choice = "check" if "check" in legal else "fold"
    if choice == "fold" and "fold" not in legal:
        choice = "check" if "check" in legal else "call"
    if choice == "raise":
        return {"type": "action", "action": "raise",
                "amount": int(best_raise_amt)}
    return {"type": "action", "action": choice}


# ============================ preflop exploit ==============================

def preflop_exploit(tr, req, pct):
    """Counter-policy vs the measured reference preflop holes.

    Reference behavior (measured): never 4bets; calls 3bets 100%;
    calls jams after 3betting 100%; folds 61% vs button opens;
    3bets ~18-22% of opens; opens ~58-70% on button.
    """
    legal = req["legal_actions"]
    to_call = req["to_call"]
    n_raises = sum(1 for h in tr.history if h["action"] == "raise")
    mr = int(req.get("min_raise", 4))
    xr = int(req.get("max_raise", 200))

    def _raise(amt):
        amt = max(mr, min(int(amt), xr))
        if "raise" not in legal or amt < mr or amt > xr:
            return None
        return {"type": "action", "action": "raise", "amount": amt}

    if n_raises == 0:
        if to_call == 0:
            # BB option after their limp: raise top 45% (their limp range is
            # capped at their 0.58 open cut; value-attack it)
            if pct > 0.55:
                r = _raise(6)
                if r:
                    return r
            return {"type": "action", "action": "check"}
        # button facing the blind: open 100% (they fold 61%, and opens set
        # up the jam line vs their 3bets)
        r = _raise(6)
        if r:
            return r
        return {"type": "action", "action": "call"}

    if n_raises == 1:
        if tr.seat == "big_blind":
            # BB vs their button open: 3bet wide-value (they call 100% of
            # their opens; our top-38% has a large equity edge)
            if pct > 0.62:
                r = _raise(16)
                if r:
                    return r
            if pct > 0.28 and "call" in legal:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}
        # button, our limp raised over: call wide (pot odds) or fold
        if pct > 0.50 and "call" in legal:
            return {"type": "action", "action": "call"}
        return {"type": "action", "action": "fold"}

    # facing 3bet+ (we opened on button): they never fold after 3betting
    if tr.seat == "button":
        if pct > 0.86:
            r = _raise(xr)             # value jam: they call 100% (36/36)
            if r:
                return r
        if pct > 0.42 and "call" in legal:
            return {"type": "action", "action": "call"}
        return {"type": "action", "action": "fold"}
    # BB facing 4bet (rare): tight call
    if pct > 0.80 and "call" in legal:
        return {"type": "action", "action": "call"}
    return {"type": "action", "action": "fold"}


# ============================ fingerprint ===================================

class FP:
    """Reference-class fingerprint from public preflop behavior."""

    def __init__(self):
        self.open6 = 0
        self.opens = 0
        self.t3b16 = 0
        self.t3b = 0
        self.call3 = 0
        self.fold3 = 0
        self.n4bet = 0

    def observe(self, history):
        pf = [h for h in history if h.get("street") == "preflop"]
        if not pf:
            return
        raises = [h for h in pf if h.get("action") == "raise"]
        opp_raises = [h for h in raises if h.get("actor") == "opp"]
        me_raises = [h for h in raises if h.get("actor") == "me"]
        if not raises:
            return
        first = raises[0]
        if first["actor"] == "opp":
            self.opens += 1
            if first.get("amount") == 6:
                self.open6 += 1
        if len(opp_raises) >= 1 and len(me_raises) >= 1:
            # we raised, they re-raised: their 3bet
            self.t3b += 1
            if opp_raises[-1].get("amount") == 16:
                self.t3b16 += 1
            # their response to OUR later raise (4bet check)
            if len(opp_raises) >= 2 and len(me_raises) >= 2:
                self.n4bet += 1
        if len(opp_raises) >= 1 and me_raises and not any(
                h.get("actor") == "opp" for h in pf[pf.index(opp_raises[-1]) + 1:]):
            pass
        # their call/fold vs our 3bet: our raise, then their action
        if me_raises and len(opp_raises) == 1 and opp_raises[0] is raises[0]:
            after = pf[pf.index(me_raises[0]) + 1:]
            their = [h for h in after if h.get("actor") == "opp"]
            if their:
                a = their[0]["action"]
                if a == "call":
                    self.call3 += 1
                elif a == "fold":
                    self.fold3 += 1

    def score(self):
        s = 0.0
        wsum = 0.0
        if self.opens >= 5:
            s += 0.30 * (self.open6 / self.opens >= 0.8)
            wsum += 0.30
        if self.t3b >= 2:
            s += 0.25 * (self.t3b16 / self.t3b >= 0.8)
            wsum += 0.25
        n3 = self.call3 + self.fold3
        if n3 >= 3:
            s += 0.35 * (self.call3 / n3 >= 0.8)
            wsum += 0.35
        if self.opens + self.t3b >= 8:
            s += 0.10 * (self.n4bet == 0)
            wsum += 0.10
        if wsum <= 0.2:
            return 0.5            # cold start: unknown
        return s / wsum
