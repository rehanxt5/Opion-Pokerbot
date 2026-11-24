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

# g17: white-box exploit modules (their-model replication, exact discard
# inference, measured preflop counter-policy, fingerprint gating)
try:
    import tmr as TMR
except Exception:
    TMR = None
G17_TMR = os.environ.get("SYN_TMR", "1") == "1" and TMR is not None
G17_PF = os.environ.get("SYN_PF", "1") == "1" and TMR is not None
# g18a: exact best-response engine (EXBR) — river-first, TMR fallback
try:
    import exbr as EXBR
except Exception:
    EXBR = None
G18_EXBR = os.environ.get("SYN_EXBR", "0") == "1" and EXBR is not None
# g18b: fingerprint-breaking (deny white-box opponents our reference pattern)
G18_FBRK = int(os.environ.get("SYN_FBRK", "2"))     # 0 off, 1 v1, 2 +call3-break
# g18c: MSU — Phi-based variance shaping from measured per-hand mu/sigma
G18_MSU = os.environ.get("SYN_MSU", "0") == "1"
MSU_W = int(os.environ.get("SYN_MSUW", "60"))          # hands-remaining window
MSU_SD_PRIOR = 45.0                                     # per-hand chips (measured ~45-55)


def _norm_pdf(z):
    import math
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _msu_sd(L):
    """Measured per-hand SD with shrinkage to the prior while immature."""
    n = L.get("msu_n") or 0
    if n < 3:
        return MSU_SD_PRIOR
    sd = (L["msu_m2"] / (n - 1)) ** 0.5
    w = min(1.0, n / 60.0)              # shrinkage weight of the measurement
    return max(18.0, (w * sd + (1.0 - w) * MSU_SD_PRIOR))


# g18c RCF — response-curve farming (H2): architecture-agnostic Bayesian model
# of THEIR response to OUR bet size, P(fold/call/raise | street, size bucket),
# with hierarchical shrinkage (cell -> street pool -> structural prior), used
# to pick EV-optimal bet sizes on turn/river. DBR-style safety: deviate from
# the core size only with enough effective data and a clear EV margin.
G18_RCF = os.environ.get("SYN_RCF", "1") == "1"
RCF_FRACS = (0.45, 0.85, 1.4, 99.0)     # bucket edges on bet/pot fraction
RCF_REPS = (0.30, 0.62, 1.10, 1.90)     # representative fraction per bucket
RCF_PRIOR = (0.40, 0.46, 0.14)          # structural prior (F, C, R)
RCF_POOL_A = 8.0                        # pseudo-counts: pool -> prior
RCF_CELL_A = 4.0                        # pseudo-counts: cell -> pool
RCF_MIN_EFF = 8.0                       # min effective samples to act


def _rcf_sb(frac):
    for i, e in enumerate(RCF_FRACS):
        if frac < e:
            return i
    return len(RCF_FRACS) - 1


def _rcf_curve(L, street, frac):
    """Posterior (F, C, R) response probabilities at bet fraction `frac`, with
    hierarchical shrinkage; returns (F, C, R, n_eff) or None if too little
    street-level data."""
    cells = L.get("rcf_cells") or {}
    pool = [0.0, 0.0, 0.0]
    for sb in range(4):
        c = cells.get((street, sb))
        if c:
            for j in range(3):
                pool[j] += c[j]
    n_pool = sum(pool)
    if n_pool < RCF_MIN_EFF:
        return None
    # pool shrunk to structural prior
    wp = n_pool / (n_pool + RCF_POOL_A)
    pr = [wp * pool[j] / n_pool + (1.0 - wp) * RCF_PRIOR[j] for j in range(3)]
    # blend the two buckets whose representative fractions are NEAREST to frac
    # (by log distance, either direction) -> a smooth curve for argmax
    import math as _m
    sb = _rcf_sb(frac)
    cand = sorted({max(0, sb - 1), sb, min(len(RCF_REPS) - 1, sb + 1)},
                  key=lambda i: abs(_m.log(frac + 0.05) - _m.log(RCF_REPS[i] + 0.05)))
    ws = {}
    for i in cand[:2]:
        d = abs(_m.log(frac + 0.05) - _m.log(RCF_REPS[i] + 0.05))
        ws[i] = 1.0 / (d + 0.25)
    tot_w = sum(ws.values())
    F = C = R = 0.0
    for i, w in ws.items():
        c = cells.get((street, i))
        if c and sum(c) > 0:
            nc = sum(c)
            wc = nc / (nc + RCF_CELL_A)
            rates = [wc * c[j] / nc + (1.0 - wc) * pr[j] for j in range(3)]
        else:
            rates = pr
        F += w / tot_w * rates[0]
        C += w / tot_w * rates[1]
        R += w / tot_w * rates[2]
    s = F + C + R
    return F / s, C / s, R / s, n_pool


def _rcf_ev(F, C, R, pot, b, v):
    """EV of betting `b` into pot `pot` with continue-equity `v`."""
    return (F * pot + C * (v * (pot + 2 * b) - b)
            + R * (v * (pot + 2 * b) - 1.35 * b))


def _g18_rcf_resize(tr, req, L, S, w, eq, reply):
    """Post-process a ladder BET (to_call == 0): re-pick its size against the
    measured response curve. Solver replies and TMR replies are excluded."""
    if (not G18_RCF or reply is None or reply.get("action") != "raise"
            or int(req.get("to_call", 0)) != 0
            or tr.street not in ("turn", "river")
            or L.get("rs_last") == (tr.hand_id, tr.street)):
        return reply
    try:
        pot = int(req["pot"])
        if pot < 6:
            return reply
        my_bet = int(req.get("my_bet", 0))
        b_core = int(reply.get("amount", 0)) - my_bet
        if b_core < 2:
            return reply
        # continue-equity (vs their calling range) like the ladder uses
        v = eq
        try:
            if S is not None and w is not None:
                q_call = call_vs_bet(L, tr.street)
                v = eq_vs_continue(S, w, q_call) or eq
        except Exception:
            pass
        if not (v >= 0.58 or v <= 0.40):        # thin-value region: core size
            return reply
        mr = int(req.get("min_raise", 2))
        xr = int(req.get("max_raise", 200))
        rng = L["rng"]
        best, best_ev, core_ev = None, None, None
        for m in (0.7, 1.0, 1.35):
            b = int(round(m * b_core))
            amt = my_bet + b
            if amt < mr or amt > xr or b < 2:
                continue
            if amt >= 0.85 * xr and m != 1.0:
                continue                             # don't create clean jams
            frac = b / float(pot)
            cur = _rcf_curve(L, tr.street, frac)
            if cur is None:
                return reply
            F, C, R, _ = cur
            ev = _rcf_ev(F, C, R, pot, b, v)
            if m == 1.0:
                core_ev = ev
            if best_ev is None or ev > best_ev:
                best, best_ev = amt, ev
        if (best is not None and core_ev is not None and best != int(reply["amount"])
                and best_ev - core_ev >= max(1.5, 0.03 * pot)):
            out = dict(reply)
            out["amount"] = int(best)
            return out
        return reply
    except Exception:
        return reply


def _g18_fbrk_preflop(tr, req, L, reply):
    """Entropy injection at near-zero EV cost:
    - open sizes {5,6,7} (was always 6)  -> breaks FP component 'open6>=0.8'
    - 3bet sizes {14,16,18} (was 16)     -> breaks FP 't3b16>=0.8'
    - occasional value 4bet jam (top hole percentile) -> breaks 'never 4bet'
    v2 (FBRK>=2): also break the call3 signature — when OUR open gets 3bet,
    fold weak/marginal opens with a probability schedule so our
    call-rate-vs-3bet lands ~0.55-0.70 instead of the reference >=0.8.
    Folding weak opens vs a 3bet is near-zero-EV (marginal odds anyway) and
    directly cuts cold-start jam-3bet exploitation by white-box opponents.
    """
    if not G18_FBRK or reply is None:
        return reply
    try:
        rng = L["rng"]
        pf = [h for h in tr.history if h.get("street") == "preflop"]
        my_raises = [h for h in pf if h.get("actor") == "me"
                     and h.get("action") == "raise"]
        opp_raises = [h for h in pf if h.get("actor") == "opp"
                      and h.get("action") == "raise"]
        # ---- v2: call3-breaking (we opened, they 3bet, we have not yet acted)
        if (G18_FBRK >= 2 and my_raises and opp_raises
                and len(my_raises) == 1 and len(opp_raises) == 1
                and pf and pf[-1].get("actor") == "opp"
                and pf.index(my_raises[0]) < pf.index(opp_raises[0])
                and int(req.get("to_call", 0)) > 0
                and tr.hole3 is not None):
            try:
                pct = preflop_pct3(tr.hole3)
            except Exception:
                pct = 0.5
            if pct < 0.30:
                p_fold = 0.65
            elif pct < 0.45:
                p_fold = 0.45
            elif pct < 0.60:
                p_fold = 0.25
            else:
                p_fold = 0.05
            if rng.random() < p_fold and "fold" in req["legal_actions"]:
                return {"type": "action", "action": "fold"}
        if reply.get("action") == "raise":
            amt = reply.get("amount")
            if amt == 6 and rng.random() < 0.55:
                reply = dict(reply)
                reply["amount"] = int(5 if rng.random() < 0.5 else 7)
            elif amt == 16 and rng.random() < 0.55:
                reply = dict(reply)
                reply["amount"] = int(14 if rng.random() < 0.5 else 18)
        # value-weighted 4bet: facing exactly two raises with a top hand
        n_raises = sum(1 for h in tr.history if h.get("action") == "raise")
        if (n_raises == 2 and int(req.get("to_call", 0)) > 0
                and tr.hole3 is not None and "raise" in req["legal_actions"]):
            try:
                pct = preflop_pct3(tr.hole3)
            except Exception:
                pct = 0.0
            if pct > 0.965 and rng.random() < 0.5:
                reply = {"type": "action", "action": "raise",
                         "amount": int(req.get("max_raise", 200))}
        return reply
    except Exception:
        return reply

LEVEL = int(os.environ.get("SYN_LEVEL", "4"))
if LEVEL > 4:
    LEVEL = 4          # 5 kept as an alias for the full stack
if not pb.PREFLOP.ok:
    LEVEL = min(LEVEL, 1)     # no table -> no percentile conditioning
DEBUG = os.environ.get("SYN_DEBUG", "") == "1"
NEUTRAL_ONLY = os.environ.get("SYN_NEUTRAL", "") == "1"     # bisection: fixed params
NO_EXCLUDE = os.environ.get("SYN_NOEXCLUDE", "") == "1"     # bisection: sample their range including their discard
OLD_DISCARD = os.environ.get("SYN_OLDDISC", "") == "1"       # bisection: mcbase standalone discard MC
MYDISC = os.environ.get("SYN_MYDISC", "1") == "1"          # g10-HA: exclude OUR discard from their deck
RS_MODE = int(os.environ.get("SYN_RS", "2"))                # g10-HB: 0 off, 1 facing-bets, 2 +tilted initiative, 3 all
BR_MODE = int(os.environ.get("SYN_BR", "1"))                # g11-HC: empirical best-response blend in the solver
BRCAP = float(os.environ.get("SYN_BRCAP", "0.75"))          # g12: max empirical blend (shrinkage toward equilibrium)
TS_MODE = int(os.environ.get("SYN_TS", "3"))                # g12: street-solver allowance cap (1 river, 2 +turn, 3 +flop)
ITQ = float(os.environ.get("SYN_ITQ", "1.0"))              # g14: global iteration-quality multiplier
ITQ_R = float(os.environ.get("SYN_ITQ_R", "1.0"))           # g15: per-street quality (river full)
ITQ_T = float(os.environ.get("SYN_ITQ_T", "0.75"))          # g15: turn
ITQ_F = float(os.environ.get("SYN_ITQ_F", "0.5"))           # g15: flop
GATE = os.environ.get("SYN_GATE", "1") == "1"                # g16: exploitability gate for the solver
ROLL_A = 0.055                                                # g16: EWMA alpha for realized per-hand net
FLOPIN = os.environ.get("SYN_FLOPIN", "1") == "1"           # g14: flop initiative solving on/off
SKIPM = int(os.environ.get("SYN_SKIP", "1"))                # g14: clear-spot skip (0 none, 1 strict, 2 loose)

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


_PF_KEYS = None
_PF_EQ = None


def _pf_lut():
    """Build sorted (key, equity) arrays for O(log n) batch lookups."""
    global _PF_KEYS, _PF_EQ
    if _PF_KEYS is not None or not pb.PREFLOP.ok:
        return _PF_KEYS is not None
    items = [(k0 * 2704 + k1 * 52 + k2, float(pb.PREFLOP.equity[cid]))
             for (k0, k1, k2), cid in pb.PREFLOP.class_of_hand.items()]
    items.sort()
    _PF_KEYS = np.array([k for k, _ in items], dtype=np.int64)
    _PF_EQ = np.array([e for _, e in items], dtype=np.float64)
    return True


def preflop_pct3_batch(rows):
    """rows: (N,3) int array -> (N,) float percentile of each 3-card hand.
    Vectorized: sort each row, pack to one int key, searchsorted."""
    if _SORTED_EQ is None or not _pf_lut():
        return np.full(len(rows), 0.5)
    r = np.sort(np.asarray(rows, dtype=np.int64), axis=1)
    keys = r[:, 0] * 2704 + r[:, 1] * 52 + r[:, 2]
    pos = np.clip(np.searchsorted(_PF_KEYS, keys), 0, len(_PF_KEYS) - 1)
    hit = _PF_KEYS[pos] == keys
    out = np.where(hit, _PF_EQ[pos], 0.5)
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


def build_turn_sample(hole2, board4, d, N, rng, my_d=None):
    hole2 = list(hole2)
    board4 = list(board4)
    known = hole2 + board4 + ([int(d)] if d is not None else [])
    if my_d is not None and MYDISC:
        known = known + [int(my_d)]                # our own discard: they can never hold it
    deck = pb._deck_excluding(known)                  # excludes their discard!
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


def build_river_sample(hole2, board5, d, my_d=None):
    """Exact enumeration over all opponent holdings (our discard excluded)."""
    hole2 = list(hole2)
    board5 = list(board5)
    known = hole2 + board5 + ([int(d)] if d is not None else [])
    if my_d is not None and MYDISC:
        known = known + [int(my_d)]                # our own discard: they can never hold it
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
        # g16: realized-edge governor state
        "roll_net": None,             # EWMA of our per-hand net
        "hands_seen": 0,              # hands observed this leg
        "gate_open": False,           # sticky solver-gate state (hysteresis)
        # g17: white-box state
        "mirror": TMR.MirrorL() if TMR is not None else None,
        "fp": TMR.FP() if TMR is not None else None,
        "tmr_rng": np.random.default_rng(),
        "edi_cache": None,            # (hand_id, weights)
        "tmr_preds": [],              # pending predictions for validation
        "tmr_hand_done": False,
        # g18c: MSU per-hand net stats (Welford, RAW net — roll_net clamps at 80)
        "msu_n": 0, "msu_mean": None, "msu_m2": 0.0,
        # g18c: RCF response-curve cells + pending observations
        "rcf_cells": {},                 # (street, size-bucket) -> [fold, call, raise]
        "rcf_pend": [],                  # our bets awaiting their response
        "rs_last": None,                 # (hand_id, street) when solver replied,
        # instrumentation
        "replies": [], "last_report": 0, "warns": 0,
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
    # ---- g17: white-box state updates (mirror of their stats, fingerprint,
    # prediction self-validation) — all from PUBLIC hand data
    try:
        if TMR is not None:
            hist = msg.get("history") or []
            m = L["mirror"]
            fp = L["fp"]
            m.observe_hand(hist)
            # our own shown cards at showdown (they see them) — from the msg
            if msg.get("result") == "showdown" and msg.get("my_hole_cards") \
                    and tr.my_discard is not None and len(msg.get("board") or []) >= 5:
                m.observe_showdown(hist,
                                   [pb.c2i(c) for c in msg["my_hole_cards"]],
                                   int(tr.my_discard),
                                   [pb.c2i(c) for c in msg["board"]])
            fp.observe(hist)
            # score pending TMR predictions (our bet -> their response)
            for pend in L["tmr_preds"]:
                street = pend.get("street")
                pred = pend.get("pred")          # (F, C, R)
                after = pend.get("after", 0)
                acts = [h for h in hist if h.get("street") == street]
                resp = next((h for h in acts[after:] if h.get("actor") == "opp"), None)
                if resp is not None and pred is not None:
                    act = resp.get("action")
                    if act in ("fold", "call", "raise"):
                        m.pred_tot += 1
                        probs = {"fold": pred[0], "call": pred[1], "raise": pred[2]}
                        if max(probs, key=probs.get) == act:
                            m.pred_hit += 1
            L["tmr_preds"] = []
            L["tmr_hand_done"] = False
    except Exception:
        pass
    # g18c RCF: resolve our pending bets -> their measured response
    try:
        if L.get("rcf_pend"):
            hist = msg.get("history") or []
            for pend in L["rcf_pend"]:
                st = pend.get("street")
                acts = [h for h in hist if h.get("street") == st]
                resp = next((h for h in acts[pend.get("after", 0):]
                             if h.get("actor") == "opp"), None)
                if resp is None:
                    continue
                a = resp.get("action")
                if a not in ("fold", "call", "raise"):
                    continue          # check (we were checked to) — no response info
                j = {"fold": 0, "call": 1, "raise": 2}[a]
                key = (st, pend["sb"])
                cell = L["rcf_cells"].get(key)
                if cell is None:
                    cell = [0, 0, 0]
                    L["rcf_cells"][key] = cell
                cell[j] += 1
            L["rcf_pend"] = []
    except Exception:
        pass
    # g16: realized-edge tracker for the exploitability gate
    L["hands_seen"] += 1
    n = max(-80.0, min(80.0, float(msg.get("net", 0))))   # clamp: one lucky
    L["roll_net"] = _ewma(L["roll_net"], n, ROLL_A)        # stack-win must not
    # g18c: MSU raw-net Welford update
    try:
        n_raw = float(msg.get("net", 0))
        if L.get("msu_mean") is None:
            L["msu_mean"], L["msu_m2"], L["msu_n"] = n_raw, 0.0, 1
        else:
            L["msu_n"] += 1
            d0 = n_raw - L["msu_mean"]
            L["msu_mean"] += d0 / L["msu_n"]
            L["msu_m2"] += d0 * (n_raw - L["msu_mean"])
    except Exception:
        pass
    hist = msg.get("history") or []                        # reopen the gate
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


# ------------------------------------------------------- g16: solve gate
def _mean3(L, fn):
    return sum(fn(L, s) for s in STREETS) / 3.0


def extremity(L):
    """g16: how far the opponent's measured frequencies sit outside a
    solid/balanced region.  Large values mark statically exploitable
    profiles (calling stations, over-folders); ~0 marks solid play.
    Calibrated on measured solver value per archetype:
      station +6700 (open), over-folder +656 (open), maniac -664 (CLOSE),
      solid equity/CFR bots -500..-900 (CLOSE).  Aggression is deliberately
    a weak signal: vs maniacs the ladder's call-more response already wins
    and the solver's bet-more tilt loses."""
    fb = _mean3(L, fold_to_bet)          # solid ~0.30-0.45
    cv = _mean3(L, call_vs_bet)          # solid ~0.35-0.55
    ag = aggr_stat(L)                    # solid ~0.25-0.45
    e = 0.0
    e += max(0.0, cv - 0.60) * 2.0       # calling station
    e += max(0.0, fb - 0.48) * 2.0       # over-folder
    e += max(0.0, 0.15 - fb) * 1.0       # never folds (station-family)
    e += max(0.0, ag - 0.65) * 0.5       # hyper-aggressive (weak signal)
    return e


def solver_gate(tr, L):
    """g16: enable the real-time solver ONLY on measured static extremity
    (stations, over-folders).  vs solid ADAPTIVE opponents a best-response-
    to-estimated-range gets counter-exploited over a long match (measured:
    -750 chips/400 hands vs the v4 peer; win-rate alone is NOT a safe signal
    — bot_cfr loses to our ladder yet punishes our solver), so realized
    edge never opens the gate.  Hysteresis: open at ex>0.22, close when
    their extremity decays below 0.18 (they adapted)."""
    if not GATE:
        return True
    hands = L["hands_seen"]
    roll = L["roll_net"] if L["roll_net"] is not None else 0.0
    ex = extremity(L)
    if DEBUG:
        try:
            import sys as _s
            print(f"[gdbg] h={hands} roll={roll:+.1f} ex={ex:.3f} "
                  f"fb={_mean3(L, fold_to_bet):.2f} cv={_mean3(L, call_vs_bet):.2f} "
                  f"ag={aggr_stat(L):.2f}", file=_s.stderr)
        except Exception:
            pass
    open_ = bool(L.get("gate_open", False))
    if hands < 14:
        want = ex > 0.18                    # cold start: blatant profiles
    elif open_:
        want = ex > 0.18                    # close when they stop being extreme
    else:
        want = ex > 0.22                    # open on clear static extremity
    L["gate_open"] = want
    return want



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
def _g17_exploit_mode(L):
    """True when the white-box layer is engaged for this opponent (leg-level:
    fingerprint matched and replication self-validated).  While the exploit
    engine drives turn/river decisions, our own ladder samples can be smaller
    (variance-aware); vs unknown opponents the full g12 sample counts keep
    the solid core at v6 quality."""
    if TMR is None or L.get("mirror") is None or L.get("fp") is None:
        return False
    try:
        if _g17_fp_score(L) < 0.45:
            return False
        m = L["mirror"]
        if m.pred_tot >= 40 and m.pred_acc() < 0.45:
            return False
        return True
    except Exception:
        return False


def sample_size(L, tr, kind):
    """Adaptive sample count from live bank fraction."""
    bank0 = max(1.0, float(tr.time_bank_ms))
    f = max(0.0, min(1.0, tr.time_left_ms / bank0))
    if _g17_exploit_mode(L):
        # g17 exploit mode: the TMR layer drives turn/river; the flop ladder
        # thresholds are coarse (0.34-0.68) and paired runouts keep keep-option
        # DIFFERENCES accurate — 40% of the g12 sample count retains decision
        # quality at ~40% of the latency (measured: flop reply 20ms -> ~11ms)
        table = {
            "flop":  (640, 520, 420, 300, 190),
            "turn":  (760, 640, 500, 360, 210),
            "disc":  (500, 400, 330, 250, 160),
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
                                          N, rng, my_d=tr.my_discard)
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
                S["river"] = build_river_sample(tr.hole2, tr.board[:5], d_use,
                                                my_d=tr.my_discard)
            if S["river"] is None:                    # bank nearly dead
                N = 300
                S["river"] = build_turn_sample(tr.hole2, tr.board[:5],
                                               tr.opp_discard, N, L["rng"],
                                               my_d=tr.my_discard)
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
    # g18c MSU: Phi-based variance shaping (replaces the fixed-scale L5 overlay).
    # P(finish leg ahead) = Phi(z), z = (net + mu*R) / (sigma*sqrt(R)) with
    # measured per-hand mu/sigma; the value of extra variance is |z|*phi(z)
    # (peak at |z|=1, ~0 when safe/hopeless) — tilt strength follows it, so
    # noise-sized leads (|z|<0.15) are correctly NOT chased.
    if LEVEL >= 3:
        if G18_MSU:
            R = max(1, tr.total_hands - tr.hand_id + 1)
            if R <= MSU_W:
                net = float(tr.leg_net)
                mu = L.get("msu_mean") if L.get("msu_n") else 0.0
                sd = _msu_sd(L)
                z = (net + (mu or 0.0) * R) / max(1.0, sd * (R ** 0.5))
                vv = abs(z) * _norm_pdf(z) / 0.24197       # 0..1 value of variance
                # |z| >= 0.4: near z=0 variance is ~free but its utility gain is
                # first-order zero while the tilt's chip-EV (mu) cost is real.
                if vv > 0.03 and abs(z) >= 0.40:
                    s = min(1.0, 1.7 * vv)
                    if z < 0:          # projected behind: raise variance
                        P["bluff_mult"] *= (1.0 + 0.55 * s)
                        P["size_mult"] *= (1.0 + 0.22 * s)
                        P["call_mult"] *= (1.0 - 0.08 * s)
                        P["rfi_pct"] = max(0.30, P["rfi_pct"] - 0.06 * s)
                        P["msu"] = ("up", round(z, 2))
                    else:              # projected ahead: damp variance
                        P["bluff_mult"] *= (1.0 - 0.55 * s)
                        P["size_mult"] *= (1.0 - 0.15 * s)
                        P["call_mult"] *= (1.0 + 0.05 * s)
                        P["value_thin"] += 0.02 * s
                        P["msu"] = ("down", round(z, 2))
        else:
            # legacy L5 fixed-scale overlay (ablation arm)
            hands_left = max(1, tr.total_hands - tr.hand_id + 1)
            if hands_left <= 45:
                lead = float(tr.leg_net)
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
    for k in ("bluff_mult", "size_mult", "call_mult"):
        P[k] = float(min(3.0, max(0.05, P[k])))
    return P


# ------------------------------------------------------------------ river solve
def _solver_tilt(strat, eq, P, L, street):
    """Exploit tilt on the solved strategy: equilibrium vs their hand RANGE is
    not equilibrium vs their ACTION frequencies.  Scale the aggressive
    branches with our measured reads:
      - weak hand (bluff region): raise-mass x bluff_mult (folders -> bluff
        more, callers -> bluff less)
      - strong hand (value region): raise-mass x call-driven factor (stations
        -> bet more/thinner)
    Non-raise mass keeps its internal ratio; total renormalized to 1.
    """
    try:
        rs = sum(p for (k, a), p in strat.items() if k == "raise")
        if rs <= 1e-9 or rs >= 0.999:
            return strat
        if eq < 0.45:
            bm = float(P.get("bluff_mult", 1.0))
            f = max(0.25, min(2.2, 0.55 + 0.55 * bm))       # bm 0.45..2.4 -> 0.80..1.87
        elif eq > 0.55:
            cv = call_vs_bet(L, street)
            f = max(0.85, min(1.60, 0.75 + 1.35 * cv))      # cv 0.4..0.7 -> 1.29..1.70
        else:
            f = 1.0
        if abs(f - 1.0) < 1e-6:
            return strat
        new_rs = min(0.92, rs * f)
        rest = 1.0 - new_rs
        old_rest = 1.0 - rs
        out = {}
        for a, p in strat.items():
            if a[0] == "raise":
                out[a] = p * (new_rs / rs)
            else:
                out[a] = p * (rest / old_rest) if old_rest > 1e-9 else 0.0
        return out
    except Exception:
        return strat


def _empirical_opp_fn(L, street):
    """g11-HC: opponent action-frequency model from street-resolved stats.

    Returns (fn, maturity). fn maps a solver action set to probabilities:
      - facing a bet: Dirichlet(2,3,1)-smoothed (fold, call, raise), size-
        conditioned (overbets fold more, small bets fold less)
      - initiative (check-or-bet): Dirichlet(2.5,1.5) over (check, bet)
    maturity in [0,1] from observed event counts; blend it into the solver so
    early hands stay close to equilibrium and exploitation ramps up with
    evidence.
    """
    d = L["st"][street]
    faced = d["folds"] + d["calls"] + d["raises"]
    p_fold = (d["folds"] + 2.0) / (faced + 6.0)
    p_call = (d["calls"] + 3.0) / (faced + 6.0)
    p_raise = (d["raises"] + 1.0) / (faced + 6.0)
    n_opts = d["checks"] + d["bets"]
    p_check = (d["checks"] + 2.5) / (n_opts + 4.0)
    p_bet = (d["bets"] + 1.5) / (n_opts + 4.0)
    maturity = min(BRCAP, (faced + n_opts) / 30.0)

    def fn(acts, to_call, pot_now):
        out = {}
        raises = [a for a in acts if a[0] == "raise"]
        if to_call > 0:
            frac = to_call / max(1.0, pot_now)
            fs = max(0.60, min(1.60, 1.0 + 0.6 * (frac - 0.33)))
            pf, pc, pr = p_fold * fs, p_call, p_raise / max(0.40, fs)
            tot = pf + pc + pr
            if tot <= 1e-9:
                return None
            pf, pc, pr = pf / tot, pc / tot, pr / tot
            for a in acts:
                if a[0] == "fold":
                    out[a] = pf
                elif a[0] == "call":
                    out[a] = pc
                elif raises:
                    out[a] = pr / len(raises)
        else:
            for a in acts:
                if a[0] == "check":
                    out[a] = p_check
                elif raises:
                    out[a] = p_bet / len(raises)
        tot = sum(out.values())
        if tot <= 1e-9:
            return None
        return {a: p / tot for a, p in out.items()}
    return fn, maturity


def solve_allowance(tr):
    """g12/g16: bank-aware governor for real-time solving.
    Returns 0..3: how much solving we can afford per decision on average.
    Street ranks: river=1 (always first), turn=2, flop=3.

    g16 fix: the old slack-only rule reacted too late when the gate stays
    open all match (station crush: measured 61-71 s vs the 60 s Stage-1
    bank).  Now the bank FRACTION downgrades solving first (<55% -> turn+
    river, <35% -> river only, <18% -> off), with the per-hand slack as a
    second condition, so a full-leg crush cannot drain the bank."""
    if TS_MODE <= 0:
        return 0
    hands_left = max(1, tr.total_hands - tr.hand_id + 1)
    per_hand = tr.time_left_ms / hands_left
    frac = tr.time_left_ms / max(1.0, float(tr.time_bank_ms))
    if frac < 0.18:
        return 0
    if frac < 0.35:
        return min(1, TS_MODE)          # river only
    if frac < 0.55 or per_hand < 70:
        return min(2, TS_MODE)          # turn + river
    if per_hand < 110:
        return min(2, TS_MODE)
    return min(3, TS_MODE)


STREET_RANK = {"river": 1, "turn": 2, "flop": 3}


def river_solver_action(tr, req, eq, P, L, S, w):
    """Real-time CFR+ street subgame vs the Bayesian-weighted range.

    g10: river (exact combos). g12: turn and flop too, via rollout-value
    worlds (each sampled world's win value already integrates the runout,
    i.e. depth-limited solving with rollout terminal values).
    """
    if RS_MODE <= 0 or S is None:
        return None
    if not solver_gate(tr, L):
        return None                     # g16: opponent not measurably exploitable
    street = tr.street
    if street not in STREET_RANK:
        return None
    if street != "flop" and "combos" not in S:
        return None
    if solve_allowance(tr) < STREET_RANK[street]:
        return None
    legal = req["legal_actions"]
    to_call = int(req["to_call"])
    pot = int(req["pot"])
    my_b, opp_b = int(req["my_bet"]), int(req["opp_bet"])
    # mode gating: solver when facing a bet; initiative only in the marginal
    # equity band (clear value-bet / give-up spots stay with the fast ladder)
    if to_call == 0 and RS_MODE < 2:
        return None
    if to_call == 0 and RS_MODE == 2 and not (0.34 < eq < 0.78):
        return None
    bank = tr.time_left_ms / max(1.0, float(tr.time_bank_ms))
    if bank < 0.12:
        return None                                   # protect the time bank
    B = pot - my_b - opp_b                             # pot before this street
    if B < 0:
        B = pot
    opp_acted = any(h.get("street") == street and h.get("actor") == "opp"
                    for h in tr.history)
    iters_tbl = {"river": (320, 180, 90), "turn": (240, 140, 70), "flop": (200, 120, 60)}
    it3 = iters_tbl[street]
    iters = it3[0] if bank > 0.5 else (it3[1] if bank > 0.30 else it3[2])
    q = {"river": ITQ_R, "turn": ITQ_T, "flop": ITQ_F}.get(street, 1.0) * ITQ
    iters = max(40, int(round(iters * q)))
    # g14: skip the solve only in truly clear spots (ladder is already right)
    if SKIPM > 0 and to_call > 0:
        need = to_call / max(1.0, pot + to_call)
        if eq > (0.95 if SKIPM == 1 else 0.92) or eq < (0.08 if SKIPM == 1 else 0.12) \
                or (SKIPM == 2 and need > 0.55 and eq < 0.40):
            return None
    if not FLOPIN and to_call == 0 and street == "flop":
        return None          # flop initiative stays with the ladder+tilt
    opp_fn, opp_blend = None, 0.0
    if BR_MODE > 0:
        opp_fn, mat = _empirical_opp_fn(L, street)
        opp_blend = mat * (1.0 if BR_MODE == 1 else 0.6)
    win_override = None
    combos = S.get("combos")
    if street == "turn":
        win_override = S["mywin"]                      # per-world runout value
    elif street == "flop":
        win_override = S["w_best"]                     # per-runout best-keep value
        if combos is None and "w_best" in S:
            combos = np.zeros((S["N"], 2), dtype=np.int8)   # shape placeholder
    if combos is None:
        return None
    hole2 = tr.hole2 if tr.hole2 is not None else list(tr.hole3[:2])
    board5 = tr.board[:5] if len(tr.board) >= 5 else list(tr.board)
    try:
        solver = pb.RiverSolver(hole2, board5, B, my_b, opp_b,
                                int(req["my_stack"]), req.get("raises_left", 4),
                                "me", combos, iters=iters,
                                root_acted=opp_acted, w=w,
                                opp_fn=opp_fn, opp_blend=opp_blend,
                                win_override=win_override, street=street)
        strat = solver.solve()
    except Exception:
        return None
    if not strat:
        return None
    if to_call == 0 and opp_blend < 0.5:
        strat = _solver_tilt(strat, eq, P, L, street)
    # map solver actions onto legal replies and sample
    rng = L["rng"]
    mr = int(req.get("min_raise", 2))
    xr = int(req.get("max_raise", 200))
    items = []
    for (kind, amt), p in strat.items():
        if p <= 1e-6:
            continue
        if kind == "fold" and "fold" not in legal:
            continue
        if kind == "check" and "check" not in legal:
            continue
        if kind == "call" and "call" not in legal:
            continue
        if kind == "raise":
            if "raise" not in legal:
                continue
            amt = int(max(mr, min(int(amt), xr)))
            items.append((p, {"type": "action", "action": "raise", "amount": amt}))
        else:
            items.append((p, {"type": "action", "action": kind}))
    tot = sum(p for p, _ in items)
    if tot <= 1e-9:
        return None
    r = rng.random() * tot
    acc = 0.0
    for p, reply in items:
        acc += p
        if r <= acc:
            return reply
    return items[-1][1]


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

    def r6():
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
            r = _raise_to(req, 16)
            if r:
                return r
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

    # g10-HB / g12: real-time CFR+ street subgames vs the Bayesian-weighted range
    rs_reply = river_solver_action(tr, req, eq, P, L, S, w)
    if rs_reply is not None:
        L["rs_last"] = (tr.hand_id, street)   # g18c: RCF must not resize solver output
        return rs_reply

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
        if eq > 0.40:
            p = {"flop": 0.50, "turn": 0.30, "river": 0.12}[street]
            if fold_to_bet(L, street) > 0.25:
                p *= P["bluff_mult"]
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
    # g17: widened near-tie mixing — our discards must not be exactly
    # argmax-predictable (the reference class is 100% predictable and we
    # exploit that; deny the same read of us). Small keep-EV cost buys
    # information protection.
    if len(order) > 1 and means[order[1]] > means[order[0]] - 0.032 \
            and rng.random() < 0.42:
        pick = 2 - order[1]
    elif len(order) > 2 and means[order[2]] > means[order[0]] - 0.016 \
            and rng.random() < 0.18:
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



# ---- calibration instrumentation (local measurement only) ----
import os as _os, json as _json
_CAL = _os.environ.get("SYN_CAL_LOG", "")
def _cal_log(tr, Smp, wgt, req=None, reply=None):
    if not _CAL or tr.street not in ("turn", "river"):
        return
    try:
        cp = tr.__dict__.get("_syn_l", {}).get("cal_p")
        if cp is not None and cp[0] == tr.hand_id:
            p = cp[1]
        elif Smp is not None and "mywin" in Smp and wgt is not None:
            w = np.asarray(wgt, dtype=np.float64)
            tot = w.sum()
            if tot > 1e-9:
                p = float((np.asarray(Smp["mywin"], dtype=np.float64) * w).sum() / tot)
            else:
                return
        else:
            return
        rec = {"t": "d", "hand": tr.hand_id,
               "street": tr.street, "p": float(p),
               "pot": int(req.get("pot", 0)) if req else None,
               "to_call": int(req.get("to_call", 0)) if req else None,
               "my_bet": int(req.get("my_bet", 0)) if req else None,
               "opp_bet": int(req.get("opp_bet", 0)) if req else None,
               "hole": [pb.i2c(int(c)) for c in (tr.hole2 or [])],
               "board": [pb.i2c(int(c)) for c in tr.board[:5]],
               "act": (reply or {}).get("action"),
               "amt": (reply or {}).get("amount")}
        with open(_CAL, "a") as f:
            f.write(_json.dumps(rec) + "\n")
    except Exception:
        pass

def _cal_result_hook(tr, msg):
    if not _CAL or msg.get("result") != "showdown":
        return
    try:
        with open(_CAL, "a") as f:
            f.write(_json.dumps({"t": "r", "hand": tr.hand_id,
                                 "net": int(msg.get("net", 0)),
                                 "opp_hole": msg.get("opp_hole_cards")}) + "\n")
        # (range-LL logging removed with EDI ablation)
    except Exception:
        pass

_orig_on_result_g17 = on_result
def on_result(tr, msg):     # type: ignore
    _orig_on_result_g17(tr, msg)
    _cal_result_hook(tr, msg)

# --------------------------------------------------------------------- decide
# g17: white-box layer activation ------------------------------------------
def _g17_fp_score(L):
    try:
        if L["fp"] is not None:
            return float(L["fp"].score())
    except Exception:
        pass
    return 0.0


def _g17_tmr_active(tr, L):
    """Postflop white-box chooser active? Requires:
    - TMR module loaded and enabled
    - opponent fingerprint matches the reference class (or too few obs)
    - the replication self-validates (modal prediction accuracy) once we
      have enough predictions; vs solver-style opponents (v5-like) the
      replication drifts and we fall back to the solid ladder
    - time budget sane
    """
    if not G17_TMR or TMR is None:
        return False
    m = L["mirror"]
    if m is None:
        return False
    bank = tr.time_left_ms / max(1.0, float(tr.time_bank_ms))
    if bank < 0.12:
        return False
    fp = _g17_fp_score(L)
    if fp < 0.45:
        return False
    # self-validation: after 40 scored predictions require >= 45% modal acc
    if m.pred_tot >= 40 and m.pred_acc() < 0.45:
        return False
    return True


def _g17_preflop_active(L):
    if not G17_PF or TMR is None:
        return False
    # cached per hand (fp only changes at hand_result)
    c = L.get("_pf_active_cache")
    if c is None or c[0] != L.get("hands_seen"):
        c = (L.get("hands_seen"), _g17_fp_score(L) >= 0.45)
        L["_pf_active_cache"] = c
    return c[1]


def _g17_tmr_decide(tr, req, L, S, wgt):
    """Build the TMR model and run the EV chooser at turn/river.
    Returns an action dict or None (fallback to the ladder)."""
    try:
        if TMR is None or S is None or "combos" not in S or wgt is None:
            return None
        street = tr.street
        if street not in ("turn", "river"):
            return None
        if street == "turn" and int(req.get("pot", 0)) < 10:
            return None        # tiny pots: the ladder is already near-optimal
        mirror = L["mirror"]
        rng = L["tmr_rng"]
        # g18a: EXACT replication first (river); TMR approximation as fallback
        if G18_EXBR and street == "river":
            try:
                r_exbr = EXBR.exbr_decide(tr, req, L, S, wgt)
                if r_exbr is not None:
                    return r_exbr
            except Exception:
                pass
        # g17 measured ablations: exact discard inversion (EDI) and the
        # true-3-card preflop band BOTH failed validation (EDI discriminates
        # only ~1.0x vs random pairs; band3 reduced range-model LL to 1.37x
        # from the host's 1.56x) -> the posterior stays the host's
        # battle-tested range weights.  The white-box edge comes from the
        # RESPONSE-CURVE exploitation, not from sharper ranges.
        w_h = np.asarray(wgt, dtype=np.float64).copy()
        tot = w_h.sum()
        if not np.isfinite(tot) or tot < 1e-9:
            w_h = np.ones(len(w_h))
        else:
            w_h = w_h / tot
        # their eq estimate per candidate
        board = [int(c) for c in (tr.board[:5] if street == "river" else tr.board[:4])]
        hole2 = [int(c) for c in tr.hole2]
        d_us = int(tr.my_discard) if tr.my_discard is not None else None
        eq_t = TMR.eq_theirs_grid(tr, mirror, L, street, board, hole2, d_us,
                                  S["combos"], rng)
        if eq_t is None:
            return None
        model = {"cand": S["combos"], "w_h": w_h, "eq_t": eq_t,
                 "mywin": S.get("mywin"), "street": street}
        to_call = int(req["to_call"])
        n_acts_street = sum(1 for h in tr.history
                            if h.get("street") == street)
        if to_call == 0:
            reply = TMR.choose_bet(model, mirror, req, street)
        else:
            # conditional update: THEIR bet is informative (value-heavy
            # ladder) — fold this into the posterior before deciding AND
            # before logging calibration
            m2 = TMR.apply_bet_likelihood(model, mirror, req, street)
            if m2 is not None:
                model = m2
            reply = TMR.choose_facing(model, mirror, req, street)
        if _CAL and model.get("mywin") is not None:
            try:
                ww = model["w_h"]
                p_cal = float((np.asarray(model["mywin"], dtype=np.float64)
                               * ww).sum() / max(1e-9, float(ww.sum())))
                L["cal_p"] = (tr.hand_id, p_cal)
            except Exception:
                pass
        # register the prediction for self-validation when we bet
        if reply is not None and reply.get("action") == "raise":
            try:
                bet = int(reply.get("amount", 0)) - int(req["my_bet"])
                pot = int(req["pot"])
                p_f, p_c, p_r = TMR.predict_response(model, mirror, bet,
                                                      pot + bet, 0, street)
                L["tmr_preds"].append({"street": street,
                                       "pred": (p_f, p_c, p_r),
                                       "after": n_acts_street})
            except Exception:
                pass
        return reply
    except Exception as e:
        try:
            _warn(L, f"tmr decide error: {e!r}")
        except Exception:
            pass
        return None


def decide(tr, req):
    t0 = time.monotonic()
    L = leg_ctx(tr)
    try:
        if LEVEL == 0:
            if om is not None:
                return om.decide(tr, req)
            return mc.decide(tr, req, dict(mc.NEUTRAL))
        street = tr.street
        kc = L.get("_knob_cache")
        if kc is None or kc[0] != tr.hand_id:
            kc = (tr.hand_id, knobs(tr, L))
            L["_knob_cache"] = kc
        P = kc[1]
        if street == "preflop":
            reply = None
            # g17: measured preflop counter-policy (fingerprint-gated)
            if _g17_preflop_active(L) and tr.hole3 is not None:
                try:
                    pct = preflop_pct3(tr.hole3)
                    reply = TMR.preflop_exploit(tr, req, pct)
                except Exception:
                    reply = None
            if reply is None:
                reply = preflop_decide(tr, req, P)
            reply = _g18_fbrk_preflop(tr, req, L, reply)
        else:
            S = hand_ctx(tr, L)
            eq, eq_u, Smp, wgt = street_equity(tr, req, S, L)
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
            # g17: white-box EV chooser at turn/river (self-validating)
            reply = None
            if _g17_tmr_active(tr, L):
                reply = _g17_tmr_decide(tr, req, L, Smp, wgt)
            if reply is None:
                reply = postflop_decide(tr, req, eq, eq_u, P, L, Smp, wgt)
                # g18c RCF: EV-optimal size vs the measured response curve
                reply = _g18_rcf_resize(tr, req, L, Smp, wgt, eq, reply)
                # g18c RCF: register our bet for curve measurement
                if (G18_RCF and reply is not None and reply.get("action") == "raise"
                        and int(req.get("to_call", 0)) == 0
                        and tr.street in STREETS):
                    try:
                        b = int(reply["amount"]) - int(req.get("my_bet", 0))
                        L["rcf_pend"].append({
                            "street": tr.street,
                            "sb": _rcf_sb(b / max(1, int(req["pot"]))),
                            "after": sum(1 for h in tr.history
                                         if h.get("street") == tr.street),
                        })
                    except Exception:
                        pass
            _cal_log(tr, Smp, wgt, req, reply)
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
    dt = (time.monotonic() - t0) * 1000.0
    _record_time(L, dt, tr)
    if _CAL and os.environ.get("SYN_TIME_LOG", "") == "1":
        try:
            with open(_CAL + ".times", "a") as f:
                f.write(_json.dumps({"h": tr.hand_id, "s": tr.street,
                                     "ms": round(dt, 2)}) + "\n")
        except Exception:
            pass
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
