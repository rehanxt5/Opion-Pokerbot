"""pbcore.py — shared library for all Pokerbots approaches (Discard Hold'em).

Components:
  - card encoding (string <-> int), consistent with engine.py (RANKS 23456789TJQKA, SUITS shdc)
  - fully vectorized numpy 7-card evaluator (ordering identical to engine.rank5/best_hand)
  - 3-card hand isomorphism (canonical classes; 1755 expected)
  - preflop equity table loader (training/build_tables.py output)
  - runtime Monte-Carlo equity (flop/turn) and exact river equity
  - protocol loop + StateTracker mirroring the official engine message flow
  - opponent range sampling helpers
  - compact CFR+ river subgame solver (hybrid bot)
  - time-bank budgeting helpers

Runtime discipline (Rulebook 5 / Protocol Spec):
  - stdout carries ONLY protocol JSON, every reply flushed; debug output -> stderr
  - exit immediately after match_end
  - never crash: bot logic wrapped; any exception falls back to a legal action
"""
import sys, os, json, math, time
from itertools import combinations, permutations
import numpy as np

RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_IDX = {r: i for i, r in enumerate(RANKS)}
SUIT_IDX = {s: i for i, s in enumerate(SUITS)}


def c2i(s):
    """'As' -> int (rank_idx*4 + suit_idx)."""
    return RANK_IDX[s[0]] * 4 + SUIT_IDX[s[1]]


def i2c(c):
    return RANKS[c >> 2] + SUITS[c & 3]


def hand_to_ints(cards):
    return [c2i(x) for x in cards]


# ------------------------------------------------------------------ evaluator
_IDX7 = np.array(list(combinations(range(7), 5)), dtype=np.intp)  # (21,5)

# ---------------------------------------------------------- direct 7-card eval
_STR_HIGH = np.array([12, 11, 10, 9, 8, 7, 6, 5, 4, 3], dtype=np.int64)
_STR_MASK = np.array(
    [sum(1 << (h - k) for k in range(5)) for h in _STR_HIGH[:9]] +
    [(1 << 12) | 1 | 2 | 4 | 8], dtype=np.int64)              # wheel A-5
_RANKS13 = np.arange(13, dtype=np.int64)
_ROWIDX_CACHE = {}


def _rowidx(N):
    r = _ROWIDX_CACHE.get(N)
    if r is None:
        r = np.repeat(np.arange(N, dtype=np.int64), 7)
        if len(_ROWIDX_CACHE) > 512:
            _ROWIDX_CACHE.clear()
        _ROWIDX_CACHE[N] = r
    return r


def eval7_direct(cards):
    """Direct best-5-of-7 evaluation (no 21-subset expansion).

    Produces EXACTLY the same scores as eval7_batch_subset (engine.py rank5
    ordering), verified on >700k adversarial + random + biased hands.
    Score = ((((cat*13+v1)*13+v2)*13+v3)*13+v4)*13+v5.
    cat: 8 SF, 7 quads, 6 boat, 5 flush, 4 straight, 3 trips, 2 two-pair, 1 pair, 0 high.
    """
    cards = np.asarray(cards)
    single = cards.ndim == 1
    if single:
        cards = cards[None, :]
    N = cards.shape[0]
    c64 = cards.astype(np.int64)
    r = c64 >> 2                                    # (N,7) rank idx
    s = c64 & 3                                     # (N,7) suit idx
    row = _rowidx(N)
    rc = np.bincount(row * 13 + r.reshape(-1), minlength=N * 13).reshape(N, 13)
    sc = np.bincount(row * 4 + s.reshape(-1), minlength=N * 4).reshape(N, 4)

    # ---- category masks
    has_quad = (rc == 4).any(1)
    quad_rank = np.where(has_quad, (rc == 4).argmax(1), -1)
    tmask = rc == 3
    n_trips = tmask.sum(1)
    pmask = rc == 2
    n_pair = pmask.sum(1)
    boat = (n_trips >= 1) & ((n_pair >= 1) | (n_trips >= 2)) & ~has_quad
    trips_only = (n_trips == 1) & (n_pair == 0) & ~has_quad
    two_pair = (n_pair >= 2) & (n_trips == 0) & ~has_quad
    pair_only = (n_pair == 1) & (n_trips == 0) & ~has_quad

    # ---- flush (at most one suit can have >=5 of 7)
    flush = sc.max(1) >= 5
    fs = sc.argmax(1)
    in_fs = s == fs[:, None]                        # (N,7)
    fr = np.where(in_fs, r, -1)
    fr_top5 = np.sort(fr, axis=1)[:, ::-1][:, :5]   # top-5 flush ranks, desc

    # ---- straights (rank bitmask); SF needs the straight inside the flush suit
    pw = np.int64(1) << r                           # (N,7)
    bm = np.bitwise_or.reduce(pw, axis=1)
    hits = (bm[:, None] & _STR_MASK[None, :]) == _STR_MASK[None, :]        # (N,10)
    straight = hits.any(1)
    st_high = np.where(hits, _STR_HIGH[None, :], -1).max(1)
    bmfs = np.bitwise_or.reduce(np.where(in_fs, pw, 0), axis=1)
    hits_fs = (bmfs[:, None] & _STR_MASK[None, :]) == _STR_MASK[None, :]
    sf = flush & hits_fs.any(1)
    sf_high = np.where(hits_fs, _STR_HIGH[None, :], -1).max(1)

    # ---- tiebreak rank tables
    pres = np.where(rc > 0, _RANKS13[None, :], -1)          # distinct ranks
    pres_desc = np.sort(pres, axis=1)[:, ::-1]              # (N,13) desc
    pr_desc = np.sort(np.where(pmask, _RANKS13[None, :], -1), axis=1)[:, ::-1]
    p1, p2 = pr_desc[:, 0], pr_desc[:, 1]
    tr_desc = np.sort(np.where(tmask, _RANKS13[None, :], -1), axis=1)[:, ::-1]
    t1, t2 = tr_desc[:, 0], tr_desc[:, 1]

    def excl_desc(ranks):                       # pres_desc minus given rank
        m = pres_desc == ranks[:, None]
        return np.sort(np.where(m, -1, pres_desc), axis=1)[:, ::-1]

    ex_p1 = excl_desc(p1)
    m2 = (pres_desc == p1[:, None]) | (pres_desc == p2[:, None])
    ex_p1p2 = np.sort(np.where(m2, -1, pres_desc), axis=1)[:, ::-1]
    ex_t1 = excl_desc(t1)
    ex_q = excl_desc(quad_rank)
    zero = np.zeros(N, dtype=np.int64)

    # ---- assemble in category precedence order (later overwrites earlier)
    cat = np.zeros(N, dtype=np.int64)
    v1 = pres_desc[:, 0].copy(); v2 = pres_desc[:, 1].copy()
    v3 = pres_desc[:, 2].copy(); v4 = pres_desc[:, 3].copy()
    v5 = pres_desc[:, 4].copy()

    m = pair_only
    cat = np.where(m, 1, cat)
    v1 = np.where(m, p1, v1); v2 = np.where(m, ex_p1[:, 0], v2)
    v3 = np.where(m, ex_p1[:, 1], v3); v4 = np.where(m, ex_p1[:, 2], v4)
    v5 = np.where(m, zero, v5)

    m = two_pair
    cat = np.where(m, 2, cat)
    v1 = np.where(m, p1, v1); v2 = np.where(m, p2, v2)
    v3 = np.where(m, ex_p1p2[:, 0], v3)
    v4 = np.where(m, zero, v4); v5 = np.where(m, zero, v5)

    m = trips_only
    cat = np.where(m, 3, cat)
    v1 = np.where(m, t1, v1); v2 = np.where(m, ex_t1[:, 0], v2)
    v3 = np.where(m, ex_t1[:, 1], v3)
    v4 = np.where(m, zero, v4); v5 = np.where(m, zero, v5)

    m = straight & ~flush                        # plain straight (flush beats it)
    cat = np.where(m, 4, cat)
    # engine rank5 semantics: v1 = straight high; v2..v5 = the straight's four
    # lower ranks desc (wheel: v=[3,3,2,1,0] — the ace slot is replaced by sh)
    is_wheel = m & (st_high == 3)
    v1 = np.where(m, st_high, v1)
    v2 = np.where(m, np.where(is_wheel, 3, st_high - 1), v2)
    v3 = np.where(m, np.where(is_wheel, 2, st_high - 2), v3)
    v4 = np.where(m, np.where(is_wheel, 1, st_high - 3), v4)
    v5 = np.where(m, np.where(is_wheel, 0, st_high - 4), v5)

    m = flush & ~sf
    cat = np.where(m, 5, cat)
    v1 = np.where(m, fr_top5[:, 0], v1); v2 = np.where(m, fr_top5[:, 1], v2)
    v3 = np.where(m, fr_top5[:, 2], v3); v4 = np.where(m, fr_top5[:, 3], v4)
    v5 = np.where(m, fr_top5[:, 4], v5)

    m = boat
    cat = np.where(m, 6, cat)
    v1 = np.where(m, t1, v1); v2 = np.where(m, np.maximum(p1, t2), v2)
    v3 = np.where(m, zero, v3); v4 = np.where(m, zero, v4); v5 = np.where(m, zero, v5)

    m = has_quad
    cat = np.where(m, 7, cat)
    v1 = np.where(m, quad_rank, v1); v2 = np.where(m, ex_q[:, 0], v2)
    v3 = np.where(m, zero, v3); v4 = np.where(m, zero, v4); v5 = np.where(m, zero, v5)

    m = sf
    cat = np.where(m, 8, cat)
    v1 = np.where(m, sf_high, v1)
    v2 = np.where(m, zero, v2); v3 = np.where(m, zero, v3)
    v4 = np.where(m, zero, v4); v5 = np.where(m, zero, v5)

    score = ((((cat * 13 + v1) * 13 + v2) * 13 + v3) * 13 + v4) * 13 + v5
    out = score.astype(np.int64)
    return out[0] if single else out


def eval7_batch(cards):
    """Best-5-of-7 evaluation, direct implementation (exactly equivalent to
    eval7_batch_subset, verified on >700k adversarial + random hands)."""
    return eval7_direct(cards)


def eval7_batch_subset(cards):
    """Vectorized best-5-of-7 evaluation.

    cards: (N,7) int array. Returns (N,) int32 score; higher is better; ordering
    is identical to engine.py's rank5 tuple ordering.
    Score = ((((cat*13+v1)*13+v2)*13+v3)*13+v4)*13+v5, v* = rank idx (0..12).
    cat: 8 SF, 7 quads, 6 boat, 5 flush, 4 straight, 3 trips, 2 two-pair, 1 pair, 0 high.
    """
    cards = np.asarray(cards)
    single = cards.ndim == 1
    if single:
        cards = cards[None, :]
    C = cards[:, _IDX7]                      # (N,21,5)
    r = (C >> 2).astype(np.int8)             # 0..12
    s = (C & 3).astype(np.int8)
    rs = np.sort(r, axis=-1)[..., ::-1]      # descending (N,21,5)

    # multiplicity of each card's rank within the combo (position-independent)
    mult = np.empty(rs.shape, dtype=np.int8)
    for i in range(5):
        mult[..., i] = (rs == rs[..., i:i + 1]).sum(axis=-1)
    m1 = mult == 1
    m2 = mult == 2
    m3 = mult == 3
    m4 = mult == 4
    has2, has3, has4 = m2.any(axis=-1), m3.any(axis=-1), m4.any(axis=-1)
    p2 = np.where(m2, rs, -1)
    p2max = p2.max(axis=-1)                          # top pair rank (or -1)
    p2min = np.where(m2, rs, 13).min(axis=-1)        # second pair rank (13 sentinel)
    two_pair = has2 & (p2max != p2min)
    pair_only = has2 & ~two_pair
    quad = has4
    boat = has3 & has2
    trips = has3 & ~has2
    uniq5 = ~has2 & ~has3 & ~has4

    consec = uniq5 & (rs[..., 0] == rs[..., 1] + 1) & (rs[..., 1] == rs[..., 2] + 1) \
        & (rs[..., 2] == rs[..., 3] + 1) & (rs[..., 3] == rs[..., 4] + 1)
    wheel = uniq5 & (rs[..., 0] == 12) & (rs[..., 1] == 3) & (rs[..., 2] == 2) \
        & (rs[..., 3] == 1) & (rs[..., 4] == 0)
    straight = consec | wheel
    flush = (s == s[..., :1]).all(axis=-1)   # (N,21)
    sh = np.where(wheel, 3, rs[..., 0])      # straight-high rank idx (wheel -> idx of 5)

    # kickers: ranks with multiplicity 1, sorted desc, sentinel -1 sinks to the end
    k1 = np.where(m1, rs, -1)
    k1 = -np.sort(-k1, axis=-1)              # descending (N,21,5)
    z = np.zeros_like(sh)

    quad_rank = np.where(m4, rs, -1).max(axis=-1)
    trip_rank = np.where(m3, rs, -1).max(axis=-1)
    v1 = np.where(quad, quad_rank,
         np.where(boat, trip_rank,
         np.where(trips, trip_rank,
         np.where(two_pair, p2max,
         np.where(pair_only, p2max, rs[..., 0])))))
    v2 = np.where(quad, k1[..., 0],
         np.where(boat, p2min,
         np.where(trips, k1[..., 0],
         np.where(two_pair, p2min,
         np.where(pair_only, k1[..., 0], rs[..., 1])))))
    v3 = np.where(quad, z,
         np.where(boat, z,
         np.where(trips, k1[..., 1],
         np.where(two_pair, k1[..., 0],
         np.where(pair_only, k1[..., 1], rs[..., 2])))))
    v4 = np.where(quad | boat | trips | two_pair, z,
         np.where(pair_only, k1[..., 2], rs[..., 3]))
    v5 = np.where(quad | boat | trips | two_pair | pair_only, z, rs[..., 4])
    cat = np.select([quad, boat, trips, two_pair, pair_only, uniq5], [7, 6, 3, 2, 1, 0], default=0)

    is_sf = flush & straight
    is_fl = flush & ~straight
    is_st = (~flush) & straight
    cat = np.where(is_sf | is_fl, np.where(straight, 8, 5), np.where(is_st, 4, cat))
    v1 = np.where(is_sf | is_fl, np.where(straight, sh, rs[..., 0]), np.where(is_st, sh, v1))
    v2 = np.where(is_sf | is_fl, np.where(straight, 0, rs[..., 1]), v2)
    v3 = np.where(is_sf | is_fl, np.where(straight, 0, rs[..., 2]), v3)
    v4 = np.where(is_sf | is_fl, np.where(straight, 0, rs[..., 3]), v4)
    v5 = np.where(is_sf | is_fl, np.where(straight, 0, rs[..., 4]), v5)

    score = ((((cat.astype(np.int32) * 13 + v1) * 13 + v2) * 13 + v3) * 13 + v4) * 13 + v5
    out = score.max(axis=1)
    return out[0] if single else out


def showdown_win(my_cards7, opp_cards7):
    """Batch showdown comparison -> +1 / 0.5 / 0 per row."""
    a = eval7_batch(my_cards7)
    b = eval7_batch(opp_cards7)
    return np.where(a > b, 1.0, np.where(a == b, 0.5, 0.0))


# ------------------------------------------------------- isomorphism (3 cards)
_SUIT_PERMS = list(permutations(range(4)))
_ISO_CACHE = {}


def iso_key3(cards):
    """Canonical key for a 3-card hand under suit isomorphism.

    cards: iterable of 3 ints -> tuple of (rank_idx, suit_perm_id) pairs,
    minimized over all 24 suit bijections. Isomorphic hands share the key.
    """
    cards = tuple(sorted(int(c) for c in cards))
    key = _ISO_CACHE.get(cards)
    if key is not None:
        return key
    best = None
    for p in _SUIT_PERMS:
        t = tuple(sorted((c >> 2, p[c & 3]) for c in cards))
        if best is None or t < best:
            best = t
    _ISO_CACHE[cards] = best
    return best


# ------------------------------------------------------------ preflop table
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class PreflopTable:
    """Preflop equity per isomorphic class (vs uniform random opponent 3-card
    hand, both players discarding ~optimally per runout). Built offline."""

    def __init__(self):
        self.ok = False
        self.equity = None
        self.class_of_hand = {}
        try:
            z = np.load(os.path.join(_DATA_DIR, "preflop_equity.npz"))
            self.equity = z["equity"]
            hands = z["hands"]                    # (N,3) int8, one row per concrete hand
            classes = z["classes"]                # (N,) class id per row
            order = z["order"]                    # (n_classes,) first row index per class
            for k in range(hands.shape[0]):
                h = hands[k]
                self.class_of_hand[(int(h[0]), int(h[1]), int(h[2]))] = int(classes[k])
            self.n_classes = len(order)
            self.ok = True
        except Exception as e:
            print(f"[pbcore] preflop table not loaded: {e!r}", file=sys.stderr)

    def class_id(self, hole3):
        h = tuple(sorted(int(x) for x in hole3))
        return self.class_of_hand.get(h, None)

    def equity_of(self, hole3):
        cid = self.class_id(hole3)
        return float(self.equity[cid]) if cid is not None else 0.5

    def percentile_of_class(self, cid):
        if cid is None or not self.ok:
            return 0.5
        return float((self.equity < self.equity[cid]).mean())

    def percentile_of(self, hole3):
        return self.percentile_of_class(self.class_id(hole3))


# --------------------------------------------------------- runtime MC equity
def _deck_excluding(known):
    ks = set(int(x) for x in known)
    return np.array([c for c in range(52) if c not in ks], dtype=np.int8)


def _sample_cards(deck, n, rollouts, rng):
    """Sample n-of-|deck| cards per rollout, fully vectorized."""
    m = deck.size
    keys = rng.random((rollouts, m))
    idx = np.argpartition(keys, n - 1, axis=1)[:, :n] if m > n \
        else np.argsort(keys, axis=1)[:, :n]
    return deck[idx]


def equity_flop3(hole3, board3, rollouts=600, rng=None):
    """Pre-discard flop equity (3 hole cards vs uniform opp 3 cards): sampled
    turn+river; both players' strength = max over their 3 discard choices.
    All 6 evaluations batched into a single eval7 call for speed."""
    rng = rng or np.random.default_rng()
    hole3 = list(hole3)
    board3 = list(board3)
    deck = _deck_excluding(hole3 + board3)
    nxt = _sample_cards(deck, 5, rollouts, rng)             # opp3 + turn + river
    opp3, run = nxt[:, 0:3], nxt[:, 3:5]
    board5 = np.concatenate([np.tile(np.array(board3, dtype=np.int8), (rollouts, 1)), run], axis=1)
    rows = []
    for keep in ((0, 1), (0, 2), (1, 2)):
        rows.append(np.concatenate(
            [np.tile(np.array([hole3[keep[0]], hole3[keep[1]]], dtype=np.int8), (rollouts, 1)),
             board5], axis=1))
    for keep in ((0, 1), (0, 2), (1, 2)):
        rows.append(np.concatenate([opp3[:, list(keep)], board5], axis=1))
    scores = eval7_batch(np.concatenate(rows, axis=0))
    N = rollouts
    best = np.maximum(np.maximum(scores[0:N], scores[N:2 * N]), scores[2 * N:3 * N])
    obest = np.maximum(np.maximum(scores[3 * N:4 * N], scores[4 * N:5 * N]), scores[5 * N:6 * N])
    w = np.where(best > obest, 1.0, np.where(best == obest, 0.5, 0.0))
    return float(w.mean())


def equity_turn2(hole2, board4, rollouts=400, rng=None):
    """Turn equity (2 kept cards vs uniform opp 2 cards), river sampled.
    Single batched eval call."""
    rng = rng or np.random.default_rng()
    board4 = list(board4)
    deck = _deck_excluding(list(hole2) + board4)
    nxt = _sample_cards(deck, 3, rollouts, rng)             # opp2 + river
    opp2, river = nxt[:, 0:2], nxt[:, 2:3]
    board5 = np.concatenate([np.tile(np.array(board4, dtype=np.int8), (rollouts, 1)), river], axis=1)
    mine = np.concatenate([np.tile(np.array(hole2, dtype=np.int8), (rollouts, 1)), board5], axis=1)
    theirs = np.concatenate([opp2, board5], axis=1)
    return float(showdown_win(mine, theirs).mean())


def equity_river2(hole2, board5):
    """Exact river equity vs uniform random opponent 2-card holding."""
    deck = _deck_excluding(list(hole2) + list(board5))
    n = deck.size
    if n < 2:
        return 0.5
    i, j = np.triu_indices(n, k=1)
    opp = deck[np.stack([i, j], axis=1)]                    # (M,2)
    M = opp.shape[0]
    mine = np.tile(np.array(list(hole2) + list(board5), dtype=np.int8), (M, 1))
    theirs = np.concatenate([opp, np.tile(np.array(board5, dtype=np.int8), (M, 1))], axis=1)
    return float(showdown_win(mine, theirs).mean())


# ------------------------------------------------------------- state tracker
class Tracker:
    """Mirrors the engine's message flow for one leg; shared by all bots."""

    def __init__(self):
        self.reset_leg()

    def reset_leg(self):
        self.total_hands = 0
        self.time_bank_ms = 30000
        self.hand_id = 0
        self.seat = None
        self.hole3 = None
        self.hole2 = None
        self.my_discard = None
        self.opp_discard = None
        self.board = []
        self.street = "preflop"
        self.pot = 0
        self.my_bet = 0
        self.opp_bet = 0
        self.my_stack = 200
        self.opp_stack = 200
        self.raises_left = 4
        self.history = []
        self.time_left_ms = 30000.0
        self.leg_net = 0.0
        self.opp_stats = {"hands": 0, "bets_faced": 0, "folds": 0, "calls": 0, "raises": 0,
                          "showdowns": 0, "opp_shown": [], "sizes": [], "discards": [],
                          "my_showdown_hole": []}

    # ---- message handling; returns reply dict or None
    def on_message(self, msg):
        t = msg.get("type")
        if t == "init":
            self.total_hands = msg["total_hands"]
            self.time_bank_ms = msg["time_bank_ms"]
            self.time_left_ms = float(msg["time_bank_ms"])
            return None
        if t == "new_hand":
            self.hand_id = msg["hand_id"]
            self.seat = msg["seat"]
            self.hole3 = [c2i(x) for x in msg["hole_cards"]]
            self.hole2 = None
            self.my_discard = None
            self.opp_discard = None
            self.board = []
            self.street = "preflop"
            self.pot = 3
            self.my_bet = 1 if self.seat == "button" else 2
            self.opp_bet = 2 if self.seat == "button" else 1
            self.my_stack = msg["stack"]
            self.opp_stack = msg["opp_stack"]
            self.raises_left = 4
            self.history = []
            self.opp_stats["hands"] += 1
            return None
        if t == "discard_result":
            self.my_discard = c2i(msg["my_discard"])
            self.opp_discard = c2i(msg["opp_discard"])
            self.hole2 = [c2i(x) for x in msg["hole_cards"]]
            self.opp_stats["discards"].append(self.opp_discard)
            return None
        if t == "hand_result":
            self.leg_net += msg.get("net", 0)
            if msg.get("opp_hole_cards"):
                self.opp_stats["showdowns"] += 1
                self.opp_stats["opp_shown"].append([c2i(x) for x in msg["opp_hole_cards"]])
                self.opp_stats["my_showdown_hole"].append(list(self.hole3 or []))
            if result_hook is not None:
                try:
                    result_hook(self, msg)
                except Exception as e:
                    print(f"[pbcore] result hook error: {e!r}", file=sys.stderr)
            return None
        if t == "action_request":
            self.board = [c2i(x) for x in msg["board"]]
            self.street = msg["street"]
            self.pot = msg["pot"]
            self.my_bet = msg["my_bet"]
            self.opp_bet = msg["opp_bet"]
            self.my_stack = msg["my_stack"]
            self.opp_stack = msg["opp_stack"]
            self.raises_left = msg.get("raises_left", 4)
            self.history = msg.get("history", [])
            self.time_left_ms = float(msg.get("time_left_ms", self.time_left_ms))
            return self.decide_wrap(msg)
        if t == "discard_request":
            self.board = [c2i(x) for x in msg["board"]]
            self.time_left_ms = float(msg.get("time_left_ms", self.time_left_ms))
            return self.discard_wrap(msg)
        if t == "match_end":
            raise SystemExit(0)
        if t == "notice":
            print(f"[pbcore] NOTICE from engine: {msg}", file=sys.stderr)
            return None
        return None

    # ---- time budget for the current decision (ms)
    def budget_ms(self, cap_ms=120.0):
        hands_left = max(1, self.total_hands - self.hand_id + 1)
        per_decision = self.time_left_ms / max(1.0, hands_left * 2.2)
        return float(min(cap_ms, max(8.0, per_decision * 0.65)))

    # ---- wraps decision hooks with a bulletproof legal fallback
    def decide_wrap(self, req):
        t0 = time.monotonic()
        try:
            reply = decide_hook(self, req)
        except SystemExit:
            raise
        except Exception as e:
            print(f"[pbcore] decide error: {e!r}", file=sys.stderr)
            reply = None
        if reply is None:
            reply = fallback_action(req)
        dt = (time.monotonic() - t0) * 1000.0
        self.time_left_ms = max(0.0, self.time_left_ms - dt)
        return reply

    def discard_wrap(self, req):
        t0 = time.monotonic()
        try:
            reply = discard_hook(self, req)
        except SystemExit:
            raise
        except Exception as e:
            print(f"[pbcore] discard error: {e!r}", file=sys.stderr)
            reply = None
        if reply is None:
            reply = {"type": "discard", "card": default_discard_str(self.hole3)}
        dt = (time.monotonic() - t0) * 1000.0
        self.time_left_ms = max(0.0, self.time_left_ms - dt)
        return reply


def default_discard_str(hole3_ints):
    """Engine's default: lowest rank; ties by suit order s,h,d,c (= our idx order)."""
    return i2c(min(hole3_ints))


def fallback_action(req):
    legal = req["legal_actions"]
    if "check" in legal:
        return {"type": "action", "action": "check"}
    if "call" in legal:
        return {"type": "action", "action": "call"}
    return {"type": "action", "action": "fold"}


decide_hook = None
discard_hook = None
result_hook = None


def run_bot(decide_fn, discard_fn, result_fn=None):
    """Protocol main loop. stdout = protocol JSON only, flushed."""
    global decide_hook, discard_hook, result_hook
    decide_hook, discard_hook, result_hook = decide_fn, discard_fn, result_fn
    tr = Tracker()
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
        reply = tr.on_message(msg)      # raises SystemExit on match_end
        if reply is not None:
            sys.stdout.write(json.dumps(reply, separators=(",", ":")) + "\n")
            sys.stdout.flush()


# ------------------------------------------------------- opponent range model
def sample_opp3(known, rollouts, rng, min_pct=None):
    """Sample opponent 3-card hands from cards not in `known` (uniform, with an
    optional iso-class percentile floor for range tightening)."""
    deck = _deck_excluding(known)
    if deck.size < 3:
        return np.zeros((rollouts, 3), dtype=np.int8)
    opp = _sample_cards(deck, 3, rollouts, rng)
    if min_pct is not None and PREFLOP.ok:
        pct = np.array([PREFLOP.percentile_of_class(PREFLOP.class_id(opp[k]))
                        for k in range(rollouts)])
        keep = pct >= min_pct
        if keep.sum() >= max(8, rollouts // 8):
            opp = opp[keep]
    return opp


def sample_runout(board_known, n_missing, rollouts, rng):
    deck = _deck_excluding(board_known)
    return _sample_cards(deck, n_missing, rollouts, rng)


# ------------------------------------------------------- CFR+ river subgame
class RiverSolver:
    """CFR+ on the river betting subgame with a discretized opponent range.

    Root = current river state (pot, street bets, raises_left, who acts).
    Payoffs: fold values exact; showdown EV = mean(win vs opp range) applied to
    the pot geometry. We return the average strategy at OUR root node.
    """

    BET_FRACS = (0.66, 1.33, 2.5)

    def __init__(self, hole2, board5, pot, my_bet, opp_bet, stack, raises_left,
                 to_act, opp_combos, iters=250, root_acted=True):
        self.hole2 = list(hole2)
        self.board5 = list(board5)
        self.pot0 = int(pot)
        self.root_acted = bool(root_acted)
        self.my_bet0, self.opp_bet0 = int(my_bet), int(opp_bet)
        self.stack = int(stack)
        self.raises_left = int(raises_left)
        self.to_act = to_act                     # 'me' | 'opp'
        self.opp = np.array(opp_combos, dtype=np.int8).reshape(-1, 2)
        self.M = self.opp.shape[0]
        self.iters = int(iters)
        mine = np.tile(np.array(self.hole2 + self.board5, dtype=np.int8), (self.M, 1))
        theirs = np.concatenate([self.opp, np.tile(np.array(self.board5, dtype=np.int8),
                                                   (self.M, 1))], axis=1)
        self.win = showdown_win(mine, theirs)    # (M,) in {0, .5, 1}
        self.regs = [dict(), dict()]             # 0=me, 1=opp  key -> regret vector
        self.ssum = dict()                       # my root-strategy accumulator

    def _legal(self, to_act, my_bet, opp_bet, raises_left):
        to_call = (opp_bet - my_bet) if to_act == "me" else (my_bet - opp_bet)
        acts = []
        if to_call > 0:
            acts.append(("fold", 0))
            acts.append(("call", 0))
        else:
            acts.append(("check", 0))
        my_committed = my_bet if to_act == "me" else opp_bet
        if raises_left > 0 and to_call < self.stack:
            for f in self.BET_FRACS:
                target = max(my_bet, opp_bet) + max(2, int(round(f * (self.pot0 + my_bet + opp_bet))))
                target = min(target, my_committed + self.stack)
                if target > max(my_bet, opp_bet) and ("raise", target) not in acts:
                    acts.append(("raise", int(target)))
        return acts

    def _finish(self, my_bet, opp_bet, folded):
        if folded == "me":
            return -my_bet
        if folded == "opp":
            return opp_bet
        return float(self.win.mean()) * (my_bet + opp_bet) - my_bet

    def _traverse(self, to_act, my_bet, opp_bet, raises_left, acted, depth):
        """acted: the opponent has already acted at the current bet level."""
        if depth <= 0:
            return self._finish(my_bet, opp_bet, None)
        acts = self._legal(to_act, my_bet, opp_bet, raises_left)
        me = to_act == "me"
        key = (to_act, my_bet, opp_bet, raises_left, acted)
        regs = self.regs[0 if me else 1].setdefault(key, {a: 0.0 for a in acts})
        strat = regret_match(regs, acts)
        vals = {}
        for a in acts:
            kind, amt = a
            if kind == "fold":
                vals[a] = self._finish(my_bet, opp_bet, "me" if me else "opp")
            elif kind == "check":
                if acted:                      # both checked -> showdown
                    vals[a] = self._finish(my_bet, opp_bet, None)
                else:                          # pass action, opponent may bet
                    vals[a] = self._traverse("opp" if me else "me", my_bet, opp_bet,
                                             raises_left, True, depth - 1)
            elif kind == "call":
                nb = max(my_bet, opp_bet)
                vals[a] = self._finish(nb, nb, None)
            else:                              # raise/bet
                if me:
                    vals[a] = self._traverse("opp", amt, opp_bet, raises_left - 1,
                                             False, depth - 1)
                else:
                    vals[a] = self._traverse("me", my_bet, amt, raises_left - 1,
                                             False, depth - 1)
        v = sum(strat[a] * vals[a] for a in acts)
        if me:
            for a in acts:
                regs[a] += vals[a] - v
                if key[:4] == ("me", self.my_bet0, self.opp_bet0, self.raises_left) \
                        and self.to_act == "me":
                    ss = self.ssum.setdefault(a, 0.0)
                    self.ssum[a] = ss + strat[a]
        else:
            for a in acts:
                regs[a] -= vals[a] - v
        return v

    def solve(self):
        # root: opponent has already acted iff we face a bet or they checked into us
        acted = self.root_acted
        for _ in range(self.iters):
            self._traverse("me", self.my_bet0, self.opp_bet0, self.raises_left,
                           acted, depth=6)
        if self.ssum:
            tot = sum(self.ssum.values())
            if tot > 0:
                return {a: p / tot for a, p in self.ssum.items()}
        acts = self._legal("me", self.my_bet0, self.opp_bet0, self.raises_left)
        key = ("me", self.my_bet0, self.opp_bet0, self.raises_left, acted)
        regs = self.regs[0].get(key)
        return regret_match(regs, acts) if regs else None


def regret_match(regs, acts):
    pos = {a: max(0.0, regs[a]) for a in acts}
    tot = sum(pos.values())
    if tot <= 0:
        n = len(acts)
        return {a: 1.0 / n for a in acts}
    return {a: pos[a] / tot for a in acts}


# --------------------------------------------------------------- global init
PREFLOP = PreflopTable()


def stderr_log(*args):
    print(*args, file=sys.stderr)
