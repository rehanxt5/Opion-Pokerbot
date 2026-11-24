#!/usr/bin/env python3
"""opion spec v1: the dataset encoder AND the future inference encoder.

Single source of truth for turning raw engine records into fixed-size model
features. The trainer (Phase 3) and the shipped numpy engine (Phase 4) both
import this exact file, so feature parity is guaranteed by construction.

Record schemas (see engine_log.py / elite_instr.py):
  act  {t, hand, leg, seat, seq, seed, bpid, bot, opp_bot, is_t, street,
        hole[3], board[<=5], opp_disc|null,
        req_echo{pot,my_bet,opp_bet,to_call,my_stack,opp_stack,legal_actions,
                 min_raise,max_raise,raises_left,hand_id,street},
        hist[[street,actor_rel,act_idx,amt]...], act, amt, tleft}
  disc {t, hand, leg, seat, seed, bpid, is_t, hole[3], board, pot,
        my_stack, opp_stack, card}
  soft {t, hand, seq, seat, street, mode(rule|argmax|sample),
        p{str(idx):prob over 6 buckets}, act_idx, rep{type,action,amount},
        cfr[6], temp, pid}

Elitev6's 6 action buckets: 0=fold, 1=check/call, 2=small raise, 3=mid raise,
4=big raise, 5=all-in. decode_bucket() is a byte-exact port of elitev6's
decode_action() (incl. the deterministic anti-fingerprint sizing), so the
student can reproduce the teacher's raise amounts at inference time.
"""
import hashlib
from itertools import combinations

import numpy as np

# ---------------------------------------------------------------- constants
SCHEMA_VERSION = "opion-spec-v1"

# card ids: 0 = none/pad, 1..52 = c2..cA, d2..dA, h2..hA, s2..sA
_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
CARD_ID = {}
for _s, _suit in enumerate(_SUITS):
    for _r, _rank in enumerate(_RANKS):
        CARD_ID[_rank + _suit] = _s * 13 + _r + 1
N_CARD = 53  # 0 = pad

# action-history token: 0 = pad, 1..32 = (street*2+actor)*4 + act + 1
N_HIST = 32
HIST_TOKEN_MAX = 32
N_TOK_ACT = 3 + 5 + 1 + N_HIST          # hole3 + board5 + opp_disc + hist32 = 41
N_TOK_DISC = 3 + 5                      # hole3 + board5 = 8

# context (float16) for act decisions
N_CTX_ACT = 17
# 0-3   street one-hot
# 4     pot/200            5  to_call/200     6  my_bet/200     7  opp_bet/200
# 8     my_stack/200       9  opp_stack/200   10 min_raise/200  11 max_raise/200
# 12    raises_left/3      13 hand_strength   14 amt/pot (teacher's action)
# 15    hist_len/32        16 tleft/30000

N_CTX_DISC = 16
# 0-3   street one-hot
# 4     pot/200            5  my_stack/200    6  opp_stack/200
# 7-15  per hole card: rank/12, is_pair_rank, board_suit_matches/5
# (disc target = index 0-2 of the discarded hole card)

N_ACT = 6  # elitev6 bucket space


def card_id(s):
    return CARD_ID[s] if s else 0


def parse_cards(lst):
    return [CARD_ID[c] for c in (lst or [])]


# ------------------------------------------------------------- hand strength
def _eval5(cards):
    """5-card evaluator -> (category 0..8, kicker list). cat: 0 high,
    1 pair, 2 two_pair, 3 trips, 4 straight, 5 flush, 6 boat, 7 quads,
    8 straight-flush."""
    ranks = sorted((c - 1) % 13 for c in cards)
    suits = [(c - 1) // 13 for c in cards]
    flush = len(set(suits)) == 1
    uniq = sorted(set(ranks), reverse=True)
    straight_hi = -1
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_hi = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:  # wheel A2345
            straight_hi = 3
    if flush and straight_hi >= 0:
        return (8, [straight_hi])
    cnt = {}
    for r in ranks:
        cnt[r] = cnt.get(r, 0) + 1
    groups = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    if groups[0][1] == 4:
        return (7, [groups[0][0], groups[1][0]])
    if groups[0][1] == 3 and groups[1][1] == 2:
        return (6, [groups[0][0], groups[1][0]])
    if flush:
        return (5, uniq)
    if straight_hi >= 0:
        return (4, [straight_hi])
    if groups[0][1] == 3:
        kick = [g[0] for g in groups[1:]]
        return (3, [groups[0][0]] + kick)
    if groups[0][1] == 2 and groups[1][1] == 2:
        kick = max(g[0] for g in groups[2:])
        return (2, [groups[0][0], groups[1][0], kick])
    if groups[0][1] == 2:
        kick = [g[0] for g in groups[1:]]
        return (1, [groups[0][0]] + kick)
    return (0, uniq)


def _best(cards):
    best = None
    for combo in combinations(cards, 5):
        ev = _eval5(combo)
        key = (ev[0], ev[1])
        if best is None or key > best:
            best = key
    return best


def hand_strength(hole_ids, board_ids):
    """Deterministic made-hand strength in [0,1]. With >=5 total cards:
    best 5-card rank (cat/8 + normalized kickers). Preflop/few cards:
    crude hole-only score (trips/pair/high)."""
    cards = [c for c in hole_ids + board_ids if c]
    if len(cards) >= 5:
        cat, kick = _best(cards)
        minor = sum(k * 13 ** (3 - i) for i, k in enumerate(kick[:4]))
        return (cat + minor / (13 ** 4)) / 9.0
    cnt = {}
    for c in cards:
        cnt[(c - 1) % 13] = cnt.get((c - 1) % 13, 0) + 1
    top = max(cnt.values())
    hi = max(cnt) / 12.0
    if top == 3:
        return 0.40 + 0.05 * hi
    if top == 2:
        return 0.25 + 0.08 * hi
    return 0.15 * hi


# --------------------------------------------------------------- legal mask
def legal_mask(req):
    """6-dim uint8 legality of the elitev6 buckets given req_echo."""
    la = req.get("legal_actions") or []
    to_call = req.get("to_call") or 0
    can_raise = ("raise" in la) and (req.get("raises_left") or 0) > 0 \
        and (req.get("min_raise") or 0) <= (req.get("max_raise") or 0)
    m = np.zeros(N_ACT, dtype=np.uint8)
    m[0] = to_call > 0 and "fold" in la
    m[1] = ("check" in la) or (to_call > 0 and "call" in la)
    m[2] = m[3] = m[4] = can_raise
    m[5] = can_raise and (req.get("my_stack") or 0) > 0
    return m


# ------------------------------------------------- bucket -> concrete action
def decode_bucket(act_idx, req):
    """Byte-exact port of elitev6 decode_action (deterministic sizing)."""
    legal = req["legal_actions"]
    to_call = req["to_call"]
    street = req["street"]
    pot = req["pot"]
    opp_bet = req["opp_bet"]

    if act_idx == 0 and to_call > 0 and "fold" in legal:
        return {"type": "action", "action": "fold"}

    if act_idx == 1 or "raise" not in legal or req.get("raises_left", 0) <= 0:
        if "check" in legal:
            return {"type": "action", "action": "check"}
        if "call" in legal:
            return {"type": "action", "action": "call"}
        return {"type": "action", "action": "fold"}

    min_r = req["min_raise"]
    max_r = req["max_raise"]
    if street == "preflop":
        if act_idx == 2:
            amt = max(min_r, min(4, max_r))
        elif act_idx == 3:
            if to_call > 0 and opp_bet <= 2:
                r_seed = (int(req.get("hand_id", 1)) * 37
                          + int(req.get("my_stack", 200))) % 100
                if r_seed < 35:
                    amt = max(min_r, min(4, max_r))
                elif r_seed < 75:
                    amt = max(min_r, min(5, max_r))
                else:
                    amt = max(min_r, min(6, max_r))
            else:
                amt = opp_bet + max(3, int(pot * 0.80)) if to_call > 0 \
                    else max(min_r, 6)
        elif act_idx == 4:
            r_seed = (int(req.get("hand_id", 1)) * 53 + int(pot)) % 100
            mult = 2.45 if r_seed < 33 else (2.85 if r_seed < 67 else 3.15)
            amt = int(opp_bet * mult) + 2 if opp_bet >= 4 else max(min_r, 8)
        else:
            amt = max_r
    else:
        eff_pot = pot + to_call
        if act_idx == 2:
            amt = opp_bet + max(2, int(eff_pot * 0.40))
        elif act_idx == 3:
            amt = opp_bet + max(4, int(eff_pot * 0.72))
        elif act_idx == 4:
            amt = opp_bet + max(8, int(eff_pot * 1.15))
        else:
            amt = max_r

    amt = max(min_r, min(max_r, int(amt)))
    return {"type": "action", "action": "raise", "amount": amt}


# --------------------------------------------------------------- partitions
def partition_of(seed):
    """92/4/4 split at match-seed level — MUST match qc_report.py exactly."""
    h = int(hashlib.md5(str(seed).encode()).hexdigest(), 16) % 100
    return 0 if h < 92 else (1 if h < 96 else 2)  # train / val / test


# ----------------------------------------------------------------- encoders
def encode_act(rec, soft):
    """act record + its soft-label record -> feature dict (fixed shapes)."""
    hole = parse_cards(rec["hole"])[:3]  # 3 pre-discard, 2 post-discard
    board = parse_cards(rec["board"]) + [0] * (5 - len(rec["board"]))
    opp_disc = card_id(rec.get("opp_disc"))
    hist = rec["hist"][-N_HIST:]
    toks = np.zeros(N_TOK_ACT, dtype=np.uint8)
    toks[0:len(hole)] = hole
    toks[3:8] = board
    toks[8] = opp_disc
    for i, (st, actor, ai, _amt) in enumerate(hist):
        toks[9 + i] = (st * 2 + actor) * 4 + ai + 1
    req = rec["req_echo"]
    ctx = np.zeros(N_CTX_ACT, dtype=np.float16)
    st = rec["street"]
    ctx[st] = 1.0
    ctx[4] = (req.get("pot") or 0) / 200.0
    ctx[5] = (req.get("to_call") or 0) / 200.0
    ctx[6] = (req.get("my_bet") or 0) / 200.0
    ctx[7] = (req.get("opp_bet") or 0) / 200.0
    ctx[8] = (req.get("my_stack") or 0) / 200.0
    ctx[9] = (req.get("opp_stack") or 0) / 200.0
    ctx[10] = (req.get("min_raise") or 0) / 200.0
    ctx[11] = (req.get("max_raise") or 0) / 200.0
    ctx[12] = (req.get("raises_left") or 0) / 3.0
    ctx[13] = hand_strength(hole, parse_cards(rec["board"]))
    ctx[14] = ((rec["amt"] or 0) / req["pot"]) if req.get("pot") else 0.0
    ctx[15] = len(rec["hist"]) / N_HIST
    ctx[16] = rec.get("tleft", 30000) / 30000.0

    p = np.zeros(N_ACT, dtype=np.float16)
    for k, v in (soft.get("p") or {}).items():
        p[int(k)] = v
    cfr = np.asarray(soft.get("cfr") or [0.0] * N_ACT, dtype=np.float16)
    return {
        "tokens": toks, "ctx": ctx, "legal": legal_mask(req),
        "p_soft": p, "cfr": cfr, "act_idx": np.uint8(soft["act_idx"]),
        "part": np.uint8(partition_of(rec["seed"])),
        "hand": np.uint32(rec["hand"] & 0xFFFFFFFF),
        "seq": np.uint16(rec["seq"]),
        "bpid": np.uint32(rec["bpid"] & 0xFFFFFFFF),
        "street": np.uint8(st),
        "amt": np.uint16(min(rec["amt"] or 0, 65535)),
    }


def encode_disc(rec):
    """disc record -> feature dict. Target = hole index of discarded card."""
    hole = parse_cards(rec["hole"])
    board = parse_cards(rec["board"]) + [0] * (5 - len(rec["board"]))
    toks = np.zeros(N_TOK_DISC, dtype=np.uint8)
    toks[0:3] = hole
    toks[3:8] = board
    board_cs = parse_cards(rec["board"])
    # disc records carry no street field: derive from board width
    st = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3}.get(len(board_cs), 3)
    ctx = np.zeros(N_CTX_DISC, dtype=np.float16)
    ctx[st] = 1.0
    ctx[4] = rec["pot"] / 200.0
    ctx[5] = rec["my_stack"] / 200.0
    ctx[6] = rec["opp_stack"] / 200.0
    for i, c in enumerate(hole):
        r = (c - 1) % 13
        s = (c - 1) // 13
        ctx[7 + i * 3] = r / 12.0
        ctx[8 + i * 3] = 1.0 if sum(1 for x in hole if (x - 1) % 13 == r) > 1 \
            else 0.0
        ctx[9 + i * 3] = (sum(1 for b in board_cs
                              if b and (b - 1) // 13 == s) / 5.0)
    ctx[15] = len(board_cs) / 5.0
    target = np.uint8(hole.index(CARD_ID[rec["card"]]))
    return {
        "tokens": toks, "ctx": ctx, "target": target,
        "part": np.uint8(partition_of(rec["seed"])),
        "hand": np.uint32(rec["hand"] & 0xFFFFFFFF),
        "street": np.uint8(st),
    }


# ------------------------------------------------------------------ decoder
def decode_features(tokens, ctx):
    """Inverse of encode_act on the lossless fields — used by the parity
    test (encode -> decode -> re-encode must be byte-identical)."""
    hole = list(tokens[0:3])
    board = list(tokens[3:8])
    opp_disc = int(tokens[8])
    hist = []
    for t in tokens[9:N_TOK_ACT]:
        if t == 0:
            continue
        v = int(t) - 1
        hist.append((v // 8, (v % 8) // 4, v % 4, 0))
    st = int(np.argmax(ctx[0:4])) if ctx[0:4].sum() > 0 else 0
    req = {
        "pot": int(round(float(ctx[4]) * 200)),
        "to_call": int(round(float(ctx[5]) * 200)),
        "my_bet": int(round(float(ctx[6]) * 200)),
        "opp_bet": int(round(float(ctx[7]) * 200)),
        "my_stack": int(round(float(ctx[8]) * 200)),
        "opp_stack": int(round(float(ctx[9]) * 200)),
        "min_raise": int(round(float(ctx[10]) * 200)),
        "max_raise": int(round(float(ctx[11]) * 200)),
        "raises_left": int(round(float(ctx[12]) * 3)),
        "legal_actions": [], "street": "preflop",
        "hand_id": 1,
    }
    return {"hole": hole, "board": board, "opp_disc": opp_disc or None,
            "hist": hist, "street": st, "req_echo": req}
