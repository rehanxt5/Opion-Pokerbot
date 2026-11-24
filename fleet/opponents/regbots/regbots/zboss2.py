#!/usr/bin/env python3
"""
Pokerbots submission: Discard Hold'em bot (heads-up no-limit, 3 hole cards,
simultaneous exposed discard after the flop).

Architecture (see README):
  1. Protocol shell      - stdin/stdout JSON, never crashes, never gets a notice
  2. Vectorized evaluator- numpy batch 5-8 card evaluation (~2 us/hand)
  3. Preflop table       - embedded equity table for all canonical 3-card classes
  4. Equity engine       - discard-aware Monte-Carlo on every street
  5. Strategy            - percentile chart preflop, EV tiers postflop,
                           equity-max discard, exploitative adjustments
  6. Opponent model      - per-leg statistics from showdowns and action history

Python 3.11 compatible. Uses only the standard library and numpy.
"""

import sys
import json
import math
import random
import base64
import zlib

import numpy as np

# ============================================================ 0. constants

RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_VALUE = {r: i for i, r in enumerate(RANKS)}          # 2 -> 0 ... A -> 12
SUIT_ORDER = {s: i for i, s in enumerate(SUITS)}

STARTING_STACK = 200
SMALL_BLIND = 1
BIG_BLIND = 2

DEBUG = False                     # stderr debug logging (64 KB/leg cap!)
STDERR_CAP = 60 * 1024

# strategy constants (all tunable; see README "Tuning")
CFG = {
    # preflop, thresholds are percentiles p in [0,1] (fraction of hands weaker)
    "open_thr": 0.25,             # BTN opens top 65%
    "open_size": 4,
    "fourbet_thr": 0.90,          # BTN 4-bets top 10%
    "fourbet_size": 2.3,          # x of opponent's 3-bet
    "call3bet_thr": 0.66,         # BTN calls a 3-bet with top 34%
    "vs_limp_raise_thr": 0.80,    # BB raises a BTN limp with top 20%
    "vs_limp_size": 6,
    "def3bet_thr": 0.9,          # BB 3-bets an open with top 16%
    "def3bet_size": 3.8,          # x of the open
    "defcall_thr": 0.52,          # BB calls an open with top 56%
    "vs4bet_call_thr": 0.88,      # facing a 4-bet: call top 12%
    # postflop tiers (equity vs uniform, adjusted by opponent range width)
    "val1": 0.74,                 # bet big above this
    "val2": 0.65,                 # bet medium above this
    "val3": 0.49,                 # bet small with this prob
    "val3_prob": 0.25,
    "bluff_hi": 0.44,             # bluff below this equity
    "bluff_base": 0.32,           # base bluff probability
    "bet_big": 0.80,
    "bet_med": 0.7,
    "bet_small": 0.45,
    "raise_eq": 0.78,             # raise a bet with equity above this
    "raise_size": 3.0,            # x of the bet faced
    "raise_prob": 0.95,
    # river
    "riv_value": 0.58,
    "riv_value2": 0.55,
    "riv_bluff": 0.25,
    "riv_big": 0.75,
    "riv_med": 0.50,
    # calling margins (required equity multiplier)
    "margin_flop": 1.18,
    "margin_turn": 1.12,
    "margin_river": 1.08,
    "margin_allin": 1.00,
    "face_raise_margin": 1.14,    # extra margin when opp raised THIS street
    # polarized-overbet guard: big prices demand a hard equity floor
    "ob_need": 0.36,              # pot-odds price above which a bet is treated as polar overbet
    "ob_call_thr": 0.68,          # min equity to call such bets (bluff-catchers fold)
    # value-raise gates (postflop)
    "val_raise": 0.8,            # equity above which value raises are automatic
    "val_shove": 0.97,            # equity above which value raises go all-in
    # opponent range fraction by this-hand preflop context
    "f_open": 0.45,               # opponent opened
    "f_limp": 0.80,               # opponent limped
    "f_call": 0.55,               # opponent called my open
    "f_3bet": 0.18,               # opponent 3-bet
    "f_4bet": 0.10,               # opponent 4-bet
    "f_check": 0.60,              # unknown / checked
    # Monte-Carlo sample counts (scaled by the time manager)
    "B_flop": 1400,
    "B_discard": 1200,
    "B_turn": 3000,
    "B_river": 3500,
    "B_min": 200,
    "reserve_s": 2.5,             # time-bank reserve
    "tie_eps": 0.004,             # discard tie randomization window
}

# ============================================================ 1. cards

RANK_IDX = np.arange(13, dtype=np.int64)
POW2 = (1 << RANK_IDX).astype(np.int32)                    # 1 << rank index


def card_to_idx(c):
    return RANK_VALUE[c[0]] * 4 + SUIT_ORDER[c[1]]


def idx_to_card(i):
    return RANKS[i // 4] + SUITS[i % 4]


def cards_to_idx(cards):
    return np.array([card_to_idx(c) for c in cards], dtype=np.int64)


# ============================================================ 2. evaluator
# Batch evaluator for K cards per hand, K in 5..8. Returns a comparable
# int64 score: (category << 26) | tiebreak. Validated against brute force
# in tools/validate_eval.py.

# straight patterns, highest first; wheel last (5-high)
_STRAIGHTS = []
for _hi in range(12, 3, -1):
    _m = 0b11111 << (_hi - 4)
    _STRAIGHTS.append((_m, _hi))
_STRAIGHTS.append(((1 << 12) | 0b1111, 3))                 # A-2-3-4-5

# tiebreak slots used per category (best-5 composition)
# cat: 0 high, 1 pair, 2 two pair, 3 trips, 4 straight, 5 flush, 6 boat,
#      7 quads, 8 straight flush
_NSLOTS = np.array([5, 4, 3, 3, 1, 5, 2, 2, 1], dtype=np.int64)

_SLOT_SHIFT = (4 * np.arange(4, -1, -1)).astype(np.int64)  # 16,12,8,4,0


def _straight_hi(mask):
    """mask: (N,) int32 rank bitmask with ace-low in bit 13. Returns (N,) hi idx (0 if none)."""
    hi = np.zeros(mask.shape[0], dtype=np.int64)
    for m, h in _STRAIGHTS:
        hit = (mask & np.int32(m)) == np.int32(m)
        hi = np.where(hit & (hi == 0), h, hi)
    return hi


def eval_hands(cards):
    """cards: (N,K) int64 card indices. Returns (N,) int64 scores."""
    N, K = cards.shape
    r = cards // 4
    s = cards % 4
    rows = np.repeat(np.arange(N, dtype=np.int64), K)

    counts = np.bincount(rows * 13 + r.ravel(), minlength=N * 13).reshape(N, 13)
    scounts = np.bincount(rows * 4 + s.ravel(), minlength=N * 4).reshape(N, 4)

    flush_suit = scounts.argmax(axis=1)
    has_flush = scounts.max(axis=1) >= 5
    fsel = s == flush_suit[:, None]

    # rank bitmasks built from counts (no arithmetic carry on duplicate ranks)
    present = (counts > 0).astype(np.int32)
    rm_all = (present * POW2[None, :]).sum(axis=1).astype(np.int32)
    rm_all |= ((rm_all >> 12) & 1) << 13                   # ace-low bit

    # flush-suit rank counts, then bitmask
    fmask_c = np.zeros((N, 13), dtype=np.int32)
    sel_rows = rows[fsel.ravel()]
    sel_ranks = r.ravel()[fsel.ravel()]
    if sel_rows.size:
        fmask_c = np.bincount(sel_rows * 13 + sel_ranks,
                              minlength=N * 13).reshape(N, 13).astype(np.int32)
    rm_fl = (fmask_c > 0) * POW2[None, :]
    rm_fl = rm_fl.sum(axis=1).astype(np.int32)
    rm_fl |= ((rm_fl >> 12) & 1) << 13

    sf_hi = _straight_hi(rm_fl)
    st_hi = _straight_hi(rm_all)

    # group shape
    n4 = (counts == 4)
    n3 = (counts == 3)
    n2 = (counts == 2)
    n_trips = n3.sum(axis=1)
    n_pairs = n2.sum(axis=1)

    cat8 = sf_hi > 0
    cat7 = n4.any(axis=1)
    cat6 = (n_trips >= 2) | ((n_trips == 1) & (n_pairs >= 1))
    cat5 = has_flush
    cat4 = st_hi > 0
    cat3 = n_trips >= 1
    cat2 = n_pairs >= 2
    cat1 = n_pairs == 1

    cat = np.select(
        [cat8, cat7, cat6, cat5, cat4, cat3, cat2, cat1],
        [8, 7, 6, 5, 4, 3, 2, 1], default=0)

    # count-order ranks (importance desc): count desc, then rank desc
    w = counts.astype(np.int64) * 16 + RANK_IDX[None, :]
    order = np.argsort(-w, axis=1, kind="stable")[:, :5]   # (N,5) rank idx

    ns = _NSLOTS[cat]
    slot = np.arange(5, dtype=np.int64)[None, :]
    ranks5 = np.where(slot < ns[:, None], order, 0)
    tb_packed = (ranks5 << _SLOT_SHIFT[None, :]).sum(axis=1)

    # flush tiebreak: top-5 ranks within the flush suit
    fkey = np.where(fsel, 12 - r, 99)
    f5 = np.sort(fkey, axis=1)[:, :5]
    tb_flush = ((12 - f5) << _SLOT_SHIFT[None, :]).sum(axis=1)

    # two-pair kicker fix (3+ pairs in 7/8 cards): kicker = best rank
    # outside the two top pairs
    if K > 6:
        p1 = order[:, 0]
        p2 = order[:, 1]
        excl = counts >= 1
        rows_n = np.arange(N, dtype=np.int64)
        excl[rows_n, p1] = False
        excl[rows_n, p2] = False
        w2 = np.where(excl, RANK_IDX[None, :], np.int64(-1))
        kick = w2.max(axis=1)
        # rebuild two-pair tiebreak as (p1, p2, kick), p1 most significant
        tb2 = kick + p2 * 16 + p1 * 256                # monotonic packing
        use2 = cat == 2
        tb_packed = np.where(use2, tb2, tb_packed)

    tb = np.where(cat == 5, tb_flush, tb_packed)
    tb = np.where((cat == 4) | (cat == 8), st_hi, tb)
    tb = np.where(cat == 8, sf_hi, tb)

    return (cat.astype(np.int64) << 26) | tb


# ============================================================ 3. canonical
# 3-card hand classes: (ranks desc, canonical suit pattern) -> key int.
# key = ((r0*13 + r1)*13 + r2)*6 + pattern, pattern in 0..5.

def class_key_vec(cards3):
    """cards3: (N,3) int64 -> (N,) int64 canonical class keys."""
    r = cards3 // 4
    s = cards3 % 4
    c = r * 8 + (7 - s)                    # sort: rank desc, then suit asc
    c_desc = np.sort(c, axis=1)[:, ::-1]
    r0 = c_desc // 8
    s1 = 7 - c_desc % 8                    # suits in rank-desc order
    l1 = (s1[:, 1] != s1[:, 0]).astype(np.int64)
    l2 = np.where(s1[:, 2] == s1[:, 0], 0,
                  np.where(s1[:, 2] == s1[:, 1], 1, 2)).astype(np.int64)
    pat = l1 * 3 + l2
    return ((r0[:, 0] * 13 + r0[:, 1]) * 13 + r0[:, 2]) * 6 + pat


def class_key_single(cards3):
    return int(class_key_vec(np.asarray(cards3, dtype=np.int64)[None, :])[0])


# ------------------------------------------------------- embedded table
# Filled by tools/gen_preflop_table.py. Format (little endian, packed):
#   int32 keys (sorted) | uint8 equity | uint8 percentile | uint16 combos
# equity/percentile scaled to 0..255. Empty string -> heuristic fallback.

PREFLOP_TABLE_B64 = "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"

_PF = {"loaded": False, "keys": None, "eq": None, "pct": None}


def _load_table():
    if _PF["loaded"]:
        return
    _PF["loaded"] = True
    if not PREFLOP_TABLE_B64:
        return
    try:
        raw = zlib.decompress(base64.b64decode(PREFLOP_TABLE_B64))
        n = len(raw) // 8
        keys = np.frombuffer(raw, dtype="<i4", count=n)
        eq = np.frombuffer(raw, dtype=np.uint8, count=n, offset=4 * n) / 255.0
        pct = np.frombuffer(raw, dtype=np.uint8, count=n, offset=5 * n) / 255.0
        assert np.all(np.diff(keys) > 0)
        _PF["keys"], _PF["eq"], _PF["pct"] = keys, eq, pct
    except Exception as e:                                 # pragma: no cover
        _dbg("table load failed: %r" % (e,))
        _PF["keys"] = None


def _heuristic_pct(my3):
    """Fallback strength estimate if the table is missing (degraded mode)."""
    r = sorted((int(c) // 4 for c in my3), reverse=True)
    s = [int(c) % 4 for c in my3]
    if r[0] == r[1] == r[2]:
        return 0.997
    if r[0] == r[1] or r[1] == r[2]:
        p = 0.78 + 0.20 * (r[0] / 12.0)
    else:
        gap = (r[0] - r[2])
        p = 0.35 + 0.30 * (r[0] / 12.0) - 0.02 * gap
        if len(set(s)) == 1:
            p += 0.06
    return min(0.99, max(0.02, p))


def my_percentile(my3):
    """Percentile (0..1, fraction of hands weaker) of a 3-card holding."""
    _load_table()
    if _PF["keys"] is None:
        return _heuristic_pct(my3)
    k = class_key_single(my3)
    pos = int(np.searchsorted(_PF["keys"], k))
    if pos < len(_PF["keys"]) and _PF["keys"][pos] == k:
        return float(_PF["pct"][pos])
    return 0.5


def class_pct_vec(cards3):
    """Vectorized percentile lookup for (N,3) sampled holdings."""
    _load_table()
    keys = class_key_vec(cards3)
    if _PF["keys"] is None:
        return np.full(keys.shape[0], 0.5)
    pos = np.searchsorted(_PF["keys"], keys)
    pos = np.clip(pos, 0, len(_PF["keys"]) - 1)
    ok = _PF["keys"][pos] == keys
    return np.where(ok, _PF["pct"][pos], 0.5)


# ============================================================ 4. equity

_RNG = np.random.default_rng()


def _sample(rng, B, k, exclude):
    """Sample k distinct cards per row, excluding the given card indices."""
    r = rng.random((B, 52))
    if exclude.size:
        r[:, exclude] = 2.0
    return np.argsort(r, axis=1)[:, :k].astype(np.int64)


def equity_flop(my3, board3, B, opp_frac=None, rng=_RNG):
    """Discard-aware flop equity.

    Simulates: opponent keeps their 2 best cards by flop strength, I keep
    each of my 3 possible pairs. Returns (eqs[3], eq_best) where eqs[k] is
    the equity of keeping pair k (order: (0,1),(0,2),(1,2)).

    opp_frac: opponent range fraction (0..1]; None or 1.0 -> uniform.
    """
    my3 = np.asarray(my3, dtype=np.int64)
    board3 = np.asarray(board3, dtype=np.int64)
    excl = np.concatenate([my3, board3])
    idx = _sample(rng, B, 5, excl)
    opp3, run = idx[:, :3], idx[:, 3:]
    board5 = np.concatenate([np.tile(board3, (B, 1)), run], axis=1)

    if opp_frac is not None and opp_frac < 0.99:
        pct = class_pct_vec(opp3)
        keep = pct >= (1.0 - opp_frac)
        if keep.sum() >= max(120, B // 25):
            opp3, board5 = opp3[keep], board5[keep]
            idx, run = idx[keep], run[keep]
            B = opp3.shape[0]
        else:
            opp_frac = None                                # fall back to uniform

    # opponent keeps best 2 (by flop strength)
    o = opp3
    op_pairs = [np.stack([o[:, 0], o[:, 1]], 1),
                np.stack([o[:, 0], o[:, 2]], 1),
                np.stack([o[:, 1], o[:, 2]], 1)]
    flop_scores = []
    for p in op_pairs:
        flop_scores.append(eval_hands(np.concatenate([p, np.tile(board3, (B, 1))], 1)))
    fs = np.stack(flop_scores, 1)                          # (B,3)
    best = fs.argmax(axis=1)
    opp_keep = np.stack(op_pairs, 1)[np.arange(B), best]   # (B,2)
    opp_final = eval_hands(np.concatenate([opp_keep, board5], 1))

    m = my3
    my_pairs = [np.stack([m[0], m[1]], 0)[None, :],
                np.stack([m[0], m[2]], 0)[None, :],
                np.stack([m[1], m[2]], 0)[None, :]]
    eqs = []
    for p in my_pairs:
        pc = np.tile(p, (B, 1))
        sc = eval_hands(np.concatenate([pc, board5], 1))
        win = float((sc > opp_final).mean())
        tie = float((sc == opp_final).mean())
        eqs.append(win + 0.5 * tie)          # split pots count half
    return eqs, max(eqs)


def equity_turn(my2, board4, opp_disc, B, rng=_RNG):
    """Post-discard turn equity. opp_disc: opponent's exposed discard (idx or -1)."""
    my2 = np.asarray(my2, dtype=np.int64)
    board4 = np.asarray(board4, dtype=np.int64)
    excl = [my2, board4]
    if opp_disc is not None and opp_disc >= 0:
        excl.append(np.array([opp_disc], dtype=np.int64))
    excl = np.concatenate(excl)
    idx = _sample(rng, B, 3, excl)                         # opp2 + river
    opp2, riv = idx[:, :2], idx[:, 2:3]
    board5 = np.concatenate([np.tile(board4, (B, 1)), riv], axis=1)
    mine = eval_hands(np.concatenate([np.tile(my2, (B, 1)), board5], 1))
    theirs = eval_hands(np.concatenate([opp2, board5], 1))
    win = float((mine > theirs).mean())
    tie = float((mine == theirs).mean())
    return win + 0.5 * tie


def equity_river(my2, board5, opp_disc, B, rng=_RNG):
    """River equity: EXACT enumeration of all opponent 2-card combos.

    Board is complete, so only the opponent's 2 cards are unknown (their
    exposed discard removed from the pool). Split pots count 0.5. The B
    argument is kept for signature compatibility and ignored.
    """
    my2 = np.asarray(my2, dtype=np.int64)
    board5 = np.asarray(board5, dtype=np.int64)
    known = [my2, board5]
    if opp_disc is not None and opp_disc >= 0:
        known.append(np.array([opp_disc], dtype=np.int64))
    used = set(np.concatenate(known).tolist())
    avail = np.array([c for c in range(52) if c not in used], dtype=np.int64)
    i, j = np.triu_indices(len(avail), 1)
    opp2 = np.stack([avail[i], avail[j]], 1)               # all C(n,2) combos
    mine = int(eval_hands(np.concatenate([my2[None, :], board5[None, :]], 1))[0])
    b5t = np.tile(board5, (len(opp2), 1))
    theirs = eval_hands(np.concatenate([opp2, b5t], 1))
    win = float((mine > theirs).mean())
    tie = float((mine == theirs).mean())
    return win + 0.5 * tie


# ============================================================ 5. logging

_stderr_bytes = [0]


def _dbg(msg):
    if not DEBUG:
        return
    try:
        line = (msg + "\n").encode()
        if _stderr_bytes[0] + len(line) < STDERR_CAP:
            sys.stderr.write(msg + "\n")
            _stderr_bytes[0] += len(line)
    except Exception:
        pass


# ============================================================ 6. opponent model

class OppModel:
    """Statistics about the current opponent, accumulated within one leg."""

    def __init__(self):
        self.hands = 0
        # preflop
        self.my_btn_opens = 0            # denominators for fold-vs-open
        self.opp_fold_vs_open = 0
        self.opp_3bet = 0
        self.opp_btn_hands = 0
        self.opp_opens = 0
        self.opp_limps = 0
        # postflop fold-vs-bet, per street
        self.my_bets = {"flop": 0, "turn": 0, "river": 0}
        self.opp_folds = {"flop": 0, "turn": 0, "river": 0}
        # showdown percentile EMA of opponent's 3-card holding (0..1, higher=tighter)
        self.sd_pct = 0.5
        self.sd_n = 0
        # discard habits
        self.opp_discard_rank_sum = 0.0
        self.opp_discard_n = 0

    # ---- derived parameters
    def fold_rate(self):
        faced = sum(self.my_bets.values())
        folded = sum(self.opp_folds.values())
        if faced < 8:
            return 0.45
        return folded / max(1, faced)

    def bluff_mult(self):
        r = self.fold_rate()
        return max(0.05, min(2.2, 0.55 + 1.8 * (r - 0.45)))

    def range_frac(self, context_frac):
        """Blend this-hand context with showdown evidence."""
        f = context_frac
        if self.sd_n >= 8:
            width = max(0.08, min(0.9, 2.0 * (1.0 - self.sd_pct)))
            f = 0.5 * f + 0.5 * width
        return max(0.08, min(0.9, f))

    # ---- update from hand_result
    def update(self, msg, my_seat):
        self.hands += 1
        hist = msg.get("history") or []
        result = msg.get("result")
        opp_disc = msg.get("opp_discard")
        opp_hole = msg.get("opp_hole_cards")

        # per-street aggression / folds
        street_my_agg = {"preflop": False, "flop": False, "turn": False, "river": False}
        for e in hist:
            st = e.get("street", "preflop")
            if e.get("actor") == "me":
                if e.get("action") == "raise":
                    street_my_agg[st] = True
                    if st in self.my_bets:
                        self.my_bets[st] += 1
            elif e.get("actor") == "opp":
                if e.get("action") == "fold" and street_my_agg.get(st) and st in self.opp_folds:
                    self.opp_folds[st] += 1

        # preflop patterns
        opp_pf = [e for e in hist if e.get("street") == "preflop" and e.get("actor") == "opp"]
        my_pf_raise = any(e.get("actor") == "me" and e.get("action") == "raise"
                          for e in hist if e.get("street") == "preflop")
        if my_seat == "button" and my_pf_raise:
            self.my_btn_opens += 1
        opp_raised_pf = any(e.get("action") == "raise" for e in opp_pf)
        opp_folded_pf = any(e.get("action") == "fold" for e in opp_pf)
        if my_seat == "big_blind":                 # opponent is on the button
            self.opp_btn_hands += 1
            if opp_raised_pf:
                self.opp_opens += 1
            elif opp_pf and not opp_folded_pf:
                self.opp_limps += 1
            if my_pf_raise and opp_folded_pf:
                self.opp_fold_vs_open += 1
            if my_pf_raise and opp_raised_pf:
                self.opp_3bet += 1

        # showdown range evidence: opp original 3 cards = shown 2 + discard
        if result == "showdown":
            if opp_hole and opp_disc:
                try:
                    orig = sorted([card_to_idx(c) for c in opp_hole] + [card_to_idx(opp_disc)])
                    p = my_percentile(orig)
                    self.sd_pct = 0.85 * self.sd_pct + 0.15 * p
                    self.sd_n += 1
                except Exception:
                    pass

        if opp_disc:
            self.opp_discard_rank_sum += RANK_VALUE.get(opp_disc[0], 6)
            self.opp_discard_n += 1

    # ---- preflop context helpers used by strategy
    def open_thr(self):
        t = CFG["open_thr"]
        n = max(1, self.my_btn_opens)
        if self.my_btn_opens >= 10 and self.opp_fold_vs_open / max(1, self.my_btn_opens) > 0.60:
            t -= 0.10
        if self.opp_3bet >= 3 and self.opp_3bet / max(1, self.my_btn_opens) > 0.20:
            t += 0.05
        return max(0.18, min(0.50, t))

    def defcall_thr(self):
        t = CFG["defcall_thr"]
        n = max(1, self.opp_btn_hands)
        if self.opp_btn_hands >= 10 and self.opp_opens / n < 0.40:
            t -= 0.08
        return max(0.30, min(0.60, t))

    def def3bet_thr(self):
        t = CFG["def3bet_thr"]
        if self.opp_btn_hands >= 10 and self.opp_limps / max(1, self.opp_btn_hands) > 0.5:
            t -= 0.05
        return max(0.74, min(0.92, t))


# ============================================================ 7. time manager

class TimeMgr:
    def __init__(self):
        self.time_left = 30.0
        self.total_hands = 200
        self.hand_id = 1

    def on_request(self, msg):
        try:
            self.time_left = msg.get("time_left_ms", int(self.time_left * 1000)) / 1000.0
        except Exception:
            pass

    def samples(self, base):
        """Scale MC sample count to the remaining time bank."""
        reserve = CFG["reserve_s"]
        rem_hands = max(0, self.total_hands - self.hand_id)
        est_calls = rem_hands * 4.6 + 4.0
        avail = max(0.3, self.time_left - reserve)
        per_call_ms = avail / max(1.0, est_calls) * 1000.0
        base_ms = base * 0.006                      # ~6 us per sample (measured)
        scale = max(0.06, min(1.25, per_call_ms / max(1.0, base_ms)))
        return max(CFG["B_min"], int(base * scale))


# ============================================================ 8. strategy

_legal_fallback_order = ("check", "call", "fold")


def fallback_action(req):
    legal = req.get("legal_actions") or []
    for a in _legal_fallback_order:
        if a in legal:
            return {"type": "action", "action": a}
    return {"type": "action", "action": "fold"}


def clamp_raise(x, req):
    try:
        lo = int(req.get("min_raise", 2))
        hi = int(req.get("max_raise", 200))
    except Exception:
        return None
    x = int(max(lo, min(hi, int(x))))
    return x


class Strategy:
    def __init__(self):
        self.opp = OppModel()
        self.tm = TimeMgr()
        self.rng = random.SystemRandom()
        self.seat = None
        self.hole = []
        self.board = []
        self.opp_disc = -1
        self.my_disc = -1
        self.hand_id = 1
        self.ctx_f = None

    # ---------------- protocol events
    def on_init(self, msg):
        self.tm.total_hands = int(msg.get("total_hands", 200))
        self.tm.hand_id = 1
        self._warmup()

    def _warmup(self):
        """Absorb numpy lazy-init costs inside the free 5 s startup window."""
        try:
            my = cards_to_idx(("As", "Ah", "7d"))
            bd = cards_to_idx(("2h", "9h", "5d"))
            equity_flop(my, bd, 64, rng=_RNG)
            equity_turn(my[:2], np.append(bd, card_to_idx("Kc")), -1, 64, rng=_RNG)
            equity_river(my[:2], np.append(bd, [card_to_idx("Kc"),
                                                card_to_idx("3s")]), -1, 64, rng=_RNG)
            my_percentile(my)
        except Exception:
            pass

    def on_new_hand(self, msg):
        self.hand_id = int(msg.get("hand_id", 1))
        self.tm.hand_id = self.hand_id
        self.seat = msg.get("seat")
        self.hole = [card_to_idx(c) for c in msg.get("hole_cards", [])]
        self.board = []
        self.opp_disc = -1
        self.my_disc = -1

    def on_discard_result(self, msg):
        try:
            self.my_disc = card_to_idx(msg["my_discard"])
            self.opp_disc = card_to_idx(msg["opp_discard"])
            self.hole = [card_to_idx(c) for c in msg.get("hole_cards", self.hole)]
        except Exception:
            pass

    def on_hand_result(self, msg):
        try:
            self.opp.update(msg, self.seat)
        except Exception as e:
            _dbg("stats update failed: %r" % (e,))

    # ---------------- helpers
    def _pf_context_frac(self):
        """Opponent range fraction from this hand's preflop action."""
        return CFG["f_check"]

    def _opp_context_frac(self, req):
        """Infer opponent range fraction from history (this hand)."""
        hist = req.get("history") or []
        opp_raises = [e for e in hist if e.get("actor") == "opp"
                      and e.get("street") == "preflop" and e.get("action") == "raise"]
        opp_calls = [e for e in hist if e.get("actor") == "opp"
                     and e.get("street") == "preflop" and e.get("action") == "call"]
        my_raises = [e for e in hist if e.get("actor") == "me"
                     and e.get("street") == "preflop" and e.get("action") == "raise"]
        if len(opp_raises) >= 2:
            return CFG["f_4bet"]
        if len(opp_raises) == 1:
            return CFG["f_3bet"] if my_raises else CFG["f_open"]
        if opp_calls:
            return CFG["f_call"]
        if my_raises:
            return CFG["f_open"]
        return CFG["f_limp"]

    def _eq_adjust(self, f):
        """Threshold shift: tighter opponent -> higher value thresholds."""
        return max(-0.08, min(0.08, 0.10 * (0.55 - f)))

    # ---------------- preflop
    def act_preflop(self, req):
        p = my_percentile(self.hole)
        opp = self.opp
        hist = req.get("history") or []
        opp_raises = sum(1 for e in hist if e.get("actor") == "opp"
                         and e.get("street") == "preflop" and e.get("action") == "raise")
        my_raises = sum(1 for e in hist if e.get("actor") == "me"
                        and e.get("street") == "preflop" and e.get("action") == "raise")
        to_call = req["to_call"]
        legal = req["legal_actions"]
        stack = req["my_stack"]
        big = to_call > 0.30 * stack          # facing a large raise / shove

        if self.seat == "button":
            if opp_raises == 0:
                # open or fold (never limp)
                if p >= opp.open_thr() and "raise" in legal:
                    amt = clamp_raise(CFG["open_size"], req)
                    return {"type": "action", "action": "raise", "amount": amt}
                return {"type": "action", "action": "fold"} if "fold" in legal \
                    else {"type": "action", "action": "call"}
            if opp_raises == 1 and my_raises == 1:
                # my open was 3-bet
                if big:
                    thr = 0.52 if to_call >= 0.8 * stack else 0.46
                    if p >= thr and "call" in legal:
                        return {"type": "action", "action": "call"}
                    return {"type": "action", "action": "fold"}
                if p >= CFG["fourbet_thr"] and "raise" in legal:
                    amt = clamp_raise(int(req["opp_bet"] * CFG["fourbet_size"]), req)
                    return {"type": "action", "action": "raise", "amount": amt}
                if p >= CFG["call3bet_thr"] and "call" in legal:
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "fold"}
            if opp_raises >= 1 and my_raises == 0:
                # I limped (fallback) and now face a raise: percentile defense
                if big:
                    thr = 0.52 if to_call >= 0.8 * stack else 0.46
                    if p >= thr and "call" in legal:
                        return {"type": "action", "action": "call"}
                    return {"type": "action", "action": "fold"}
                if p >= 0.80 and "raise" in legal:
                    amt = clamp_raise(int(req["opp_bet"] * 2.8), req)
                    return {"type": "action", "action": "raise", "amount": amt}
                if p >= 0.50 and "call" in legal:
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "fold"}
            # cap reached or unexpected shape: fall back to a legal action
            return fallback_action(req)

        # big blind
        if opp_raises == 0:
            # BTN limped (to_call can be 0 or negative when checking is legal)
            if to_call <= 0 and "check" in legal:
                if p >= CFG["vs_limp_raise_thr"] and "raise" in legal:
                    amt = clamp_raise(CFG["vs_limp_size"], req)
                    return {"type": "action", "action": "raise", "amount": amt}
                return {"type": "action", "action": "check"}
            return fallback_action(req)
        if opp_raises >= 1 and my_raises == 0:
            # facing an open
            if big:
                thr = 0.52 if to_call >= 0.8 * stack else 0.46
                if p >= thr and "call" in legal:
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "fold"}
            if p >= opp.def3bet_thr() and "raise" in legal:
                amt = clamp_raise(int(req["opp_bet"] * CFG["def3bet_size"]), req)
                return {"type": "action", "action": "raise", "amount": amt}
            if p >= opp.defcall_thr() and "call" in legal:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}
        if my_raises >= 1 and opp_raises >= 1:
            # my 3-bet got 4-bet (possibly a shove)
            if big:
                thr = 0.52 if to_call >= 0.8 * stack else 0.46
                if p >= thr and "call" in legal:
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "fold"}
            if p >= CFG["vs4bet_call_thr"] and "call" in legal:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}
        return fallback_action(req)

    # ---------------- postflop (flop / turn / river)
    def act_post(self, req):
        street = req["street"]
        to_call = req["to_call"]
        pot = req["pot"]
        legal = req["legal_actions"]
        rng_np = _RNG

        f = self._opp_context_frac(req) if street == "flop" else self._pf_context_frac()
        if street == "flop":
            f = self.opp.range_frac(f)
            self.ctx_f = f                       # reuse on turn/river
        else:
            f = getattr(self, "ctx_f", self.opp.range_frac(CFG["f_check"]))
        adj = self._eq_adjust(f)

        if street == "flop":
            B = self.tm.samples(CFG["B_flop"])
            _, eq = equity_flop(self.hole, self.board[:3], B,
                                opp_frac=f if to_call > 0 else None, rng=rng_np)
        elif street == "turn":
            B = self.tm.samples(CFG["B_turn"])
            eq = equity_turn(self.hole, self.board[:4], self.opp_disc, B, rng=rng_np)
        else:
            B = self.tm.samples(CFG["B_river"])
            eq = equity_river(self.hole, self.board[:5], self.opp_disc, B, rng=rng_np)

        if to_call == 0:
            return self._bet_decision(req, street, eq, adj)
        return self._face_bet_decision(req, street, eq, adj)

    def _bet_decision(self, req, street, eq, adj):
        legal = req["legal_actions"]
        pot = req["pot"]
        if "raise" not in legal:               # cannot bet (cap reached / not legal)
            return {"type": "action", "action": "check"}

        if street == "river":
            if eq >= CFG["riv_value"] + adj:
                amt = clamp_raise(pot * CFG["riv_big"], req)
                if eq >= 0.85 or (req["my_stack"] <= pot * 0.9 and eq >= 0.72):
                    amt = clamp_raise(req["max_raise"], req)
                return {"type": "action", "action": "raise", "amount": amt}
            if eq >= CFG["riv_value2"] + adj:
                amt = clamp_raise(pot * CFG["riv_med"], req)
                return {"type": "action", "action": "raise", "amount": amt}
            if eq <= CFG["riv_bluff"]:
                # indifference-based bluff frequency for a half-pot bet
                b = pot * CFG["riv_med"]
                p_bluff = (b / (pot + 2.0 * b)) * self.opp.bluff_mult()
                if self.rng.random() < min(0.45, p_bluff):
                    amt = clamp_raise(b, req)
                    return {"type": "action", "action": "raise", "amount": amt}
            return {"type": "action", "action": "check"}

        # flop / turn value tiers
        if eq >= CFG["val1"] + adj:
            amt = clamp_raise(pot * CFG["bet_big"], req)
            if eq >= 0.85 and self.rng.random() < 0.5:
                amt = clamp_raise(req["max_raise"], req)      # occasional shove for value
            return {"type": "action", "action": "raise", "amount": amt}
        if eq >= CFG["val2"] + adj:
            amt = clamp_raise(pot * CFG["bet_med"], req)
            return {"type": "action", "action": "raise", "amount": amt}
        if eq >= CFG["val3"] + adj and self.rng.random() < CFG["val3_prob"]:
            amt = clamp_raise(pot * CFG["bet_small"], req)
            return {"type": "action", "action": "raise", "amount": amt}
        if eq <= CFG["bluff_hi"] - adj * 0.5:
            p_bluff = CFG["bluff_base"] * self.opp.bluff_mult()
            if street == "flop":
                p_bluff = min(0.40, p_bluff * 1.2)            # more room to bluff flop
            if self.rng.random() < p_bluff:
                amt = clamp_raise(pot * CFG["bet_small"], req)
                return {"type": "action", "action": "raise", "amount": amt}
        return {"type": "action", "action": "check"}

    def _face_bet_decision(self, req, street, eq, adj):
        legal = req["legal_actions"]
        pot = req["pot"]
        to_call = req["to_call"]
        my_stack = req["my_stack"]

        need = to_call / max(1.0, pot + to_call)
        # overbet / polarized-shove guard
        ob_floor = CFG["ob_call_thr"] if need >= CFG["ob_need"] else 0.0

        allin_call = to_call >= my_stack
        if allin_call:
            margin = CFG["margin_allin"]
        else:
            margin = {"flop": CFG["margin_flop"], "turn": CFG["margin_turn"],
                      "river": CFG["margin_river"]}[street]
            # opponent raised this street: their range is stronger than a bet
            opp_raised = False
            try:
                opp_raised = any(e.get("actor") == "opp" and e.get("street") == street
                                 and e.get("action") == "raise"
                                 for e in (req.get("history") or []))
            except Exception:
                opp_raised = False
            if opp_raised:
                margin *= CFG["face_raise_margin"]

        # value raise (monsters always raise, strong hands mix)
        if "raise" in legal and eq >= CFG["raise_eq"] + adj and \
                (eq >= CFG["val_raise"] or self.rng.random() < CFG["raise_prob"]):
            amt = clamp_raise(int(req["opp_bet"] + CFG["raise_size"] * (pot + to_call)), req)
            if eq >= CFG["val_shove"]:
                amt = clamp_raise(req["max_raise"], req)
            return {"type": "action", "action": "raise", "amount": amt}

        # call if equity beats the price (overbet floor overrides pot odds)
        if eq >= max(need * margin, ob_floor):
            return {"type": "action", "action": "call"}

        # big all-in calls need a bit less margin when pot-committed
        if allin_call and eq >= max(need * 0.92, ob_floor):
            return {"type": "action", "action": "call"}
        return {"type": "action", "action": "fold"}

    # ---------------- discard
    def discard_choice(self, req):
        board3 = [card_to_idx(c) for c in req.get("board", [])]
        hole = [card_to_idx(c) for c in req.get("hole_cards", self.hole)]
        B = self.tm.samples(CFG["B_discard"])
        if req.get("my_stack", 200) == 0 or req.get("opp_stack", 200) == 0:
            B = min(B, 500)
        try:
            f = self._opp_context_frac({"history": req.get("_hist", [])})
            eqs, _ = equity_flop(hole, board3, B, opp_frac=None, rng=_RNG)
        except Exception as e:
            _dbg("discard MC failed: %r" % (e,))
            eqs = None
        if eqs is None:
            return self._default_discard(hole)
        # pair order: (0,1),(0,2),(1,2) -> discard index 2, 1, 0
        best = int(np.argmax(eqs))
        eps = CFG["tie_eps"]
        cand = [i for i, e in enumerate(eqs) if e >= eqs[best] - eps]
        best = self.rng.choice(cand) if cand else best
        discard_pos = {0: 2, 1: 1, 2: 0}[best]
        return {"type": "discard", "card": idx_to_card(hole[discard_pos])}

    def _default_discard(self, hole):
        worst = min(hole, key=lambda c: (c // 4, c % 4))
        return {"type": "discard", "card": idx_to_card(worst)}

    # ---------------- dispatch
    def on_action_request(self, req):
        self.tm.on_request(req)
        self.board = [card_to_idx(c) for c in req.get("board", [])]
        try:
            if req.get("street") == "preflop":
                a = self.act_preflop(req)
            else:
                a = self.act_post(req)
        except Exception as e:
            _dbg("decision error: %r" % (e,))
            a = fallback_action(req)
        return self._validate(a, req)

    def _validate(self, a, req):
        legal = req.get("legal_actions") or []
        try:
            if a.get("type") != "action" or a.get("action") not in legal:
                return fallback_action(req)
            if a["action"] == "raise":
                amt = int(a.get("amount", 0))
                lo = int(req.get("min_raise", 2))
                hi = int(req.get("max_raise", amt))
                if not (lo <= amt <= hi):
                    amt = int(max(lo, min(hi, amt)))
                a = {"type": "action", "action": "raise", "amount": amt}
        except Exception:
            return fallback_action(req)
        return a


# ============================================================ 9. main loop

def main():
    strat = Strategy()
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if not isinstance(msg, dict):
            continue
        t = msg.get("type")
        if t == "action_request":
            reply = strat.on_action_request(msg)
        elif t == "discard_request":
            strat.tm.on_request(msg)
            try:
                reply = strat.discard_choice(msg)
            except Exception:
                reply = strat._default_discard(
                    [card_to_idx(c) for c in msg.get("hole_cards", [])])
        elif t == "init":
            strat.on_init(msg)
            continue
        elif t == "new_hand":
            strat.on_new_hand(msg)
            continue
        elif t == "discard_result":
            strat.on_discard_result(msg)
            continue
        elif t == "hand_result":
            strat.on_hand_result(msg)
            continue
        elif t == "match_end":
            break
        else:
            continue                      # ignore unknown types (forward compat)
        try:
            out.write(json.dumps(reply, separators=(",", ":")) + "\n")
            out.flush()
        except BrokenPipeError:
            return


if __name__ == "__main__":
    main()
