#!/usr/bin/env python3
"""Opion student player: a bot for engine_log.py's stdin/stdout protocol.

Plays via the numpy port (np_policy.NumpyPolicy) + spec.py features so the
live feature vectors are byte-compatible with the packed training features.

Engine protocol (one bot process = ONE leg, so hand ids are unique here):
  init / new_hand / action_request / discard_request / discard_result /
  hand_result / notice / match_end  (JSON lines on stdin; one JSON reply per
  request on stdout).

Rollout sidecar (env OPION_ROLLOUT_LOG, shared by both seats in self-mirror
matches — lines are keyed by pid == engine's bpid):
  {"pid","hand","seq","st","seat","tokens[41]","ctx[17]","legal[6]",
   "b"(bucket),"logp","amt","pot"}   per action decision
  {"pid","hand","net","r":true}      per hand_result (the PPO reward)

Live-feature convention (§9.3, post-v2): ctx[14] (own-action amt/pot) is 0.0
everywhere — the v2 re-distilled model was trained with it zeroed, so live
play is a plain single forward. amt+pot are still logged for diagnostics.
The ORIGINAL final_act.pt is NOT live-playable (leaky own-action feature).

Discards come from the FROZEN disc model (argmax, deterministic).
"""
import json
import os
import random
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import np_policy  # noqa: E402
import spec  # noqa: E402

_SI = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
_AI = {"fold": 0, "check": 1, "call": 2, "raise": 3}
_SEATS = {"button": 0, "big_blind": 1}
ACT_PT = os.environ.get("OPION_ACT_PT", "")
ACT_NPZ = os.environ.get("OPION_ACT_NPZ", "")
DISC_PT = os.environ.get("OPION_DISC_PT", "")
SIDECAR = os.environ.get("OPION_ROLLOUT_LOG", "")

POLICY = None
DISC = None
SIDE = None
RNG = random.SystemRandom()
MYPID = os.getpid()


def load_models():
    global POLICY, DISC, SIDE
    sd = np_policy.load_npz(ACT_NPZ) if ACT_NPZ else \
        np_policy.load_pt(ACT_PT)["model"]
    POLICY = np_policy.NumpyPolicy(sd)
    if DISC_PT:
        DISC = np_policy.NumpyPolicy(np_policy.load_pt(DISC_PT)["model"],
                                     n_tok=8, n_ctx=16, n_act=3,
                                     map_tokens=False)
    SIDE = open(SIDECAR, "a") if SIDECAR else None


def log(line):
    if SIDE:
        SIDE.write(json.dumps(line, separators=(",", ":")) + "\n")
        SIDE.flush()


class HandState:
    __slots__ = ("hand", "seat", "hole", "opp_disc", "seq")

    def __init__(self):
        self.hand = self.seat = 0
        self.hole = []
        self.opp_disc = None
        self.seq = 0


H = HandState()


def card_ids(cards):
    return [spec.CARD_ID[c] for c in cards if c]


def build_act_features(msg):
    """action_request -> (tokens[41], ctx[17]) matching spec.encode_act."""
    hole = card_ids(H.hole)
    board = card_ids(msg["board"])
    req = {k: v for k, v in msg.items() if k not in ("history", "time_left_ms",
                                                     "type", "hand_id")}
    req["hand_id"] = msg["hand_id"]
    hist = msg["history"]
    toks = np.zeros(spec.N_TOK_ACT, dtype=np.int64)
    toks[0:len(hole)] = hole
    toks[3:3 + len(board)] = board
    toks[8] = spec.card_id(H.opp_disc) if H.opp_disc else 0
    for i, h in enumerate(hist[-spec.N_HIST:]):
        toks[9 + i] = (_SI[h["street"]] * 2 + (0 if h["actor"] == "me" else 1)) \
            * 4 + _AI[h["action"]] + 1
    st = _SI[msg["street"]]
    ctx = np.zeros(spec.N_CTX_ACT, dtype=np.float32)
    ctx[st] = 1.0
    pot = msg.get("pot") or 0
    ctx[4] = pot / 200.0
    ctx[5] = (msg.get("to_call") or 0) / 200.0
    ctx[6] = (msg.get("my_bet") or 0) / 200.0
    ctx[7] = (msg.get("opp_bet") or 0) / 200.0
    ctx[8] = (msg.get("my_stack") or 0) / 200.0
    ctx[9] = (msg.get("opp_stack") or 0) / 200.0
    ctx[10] = (msg.get("min_raise") or 0) / 200.0
    ctx[11] = (msg.get("max_raise") or 0) / 200.0
    ctx[12] = (msg.get("raises_left") or 0) / 3.0
    ctx[13] = spec.hand_strength(hole, board)
    ctx[14] = 0.0            # live convention: own action not known yet
    ctx[15] = len(hist) / spec.N_HIST
    ctx[16] = (msg.get("time_left_ms") or 30000) / 30000.0
    return toks, ctx, req, pot, st


def handle_action(msg):
    toks, ctx, req, pot, st = build_act_features(msg)
    legal = spec.legal_mask(req)
    # Single forward on the base state (ctx[14]=0). Valid for the v2
    # re-distilled model, which was trained with ctx[14]=0 everywhere —
    # the original checkpoint had a leaky own-action feature and is NOT
    # live-playable (dummy test 2026-11-14: forced amt=0 -> 0 raises,
    # per-bucket assembly -> over-raising; see plan.md §9.3).
    logits = POLICY.forward(toks[None], ctx[None])[0].astype(np.float64)
    masked = np.where(legal != 0, logits, -np.inf)
    m = masked.max()
    p = np.exp(masked - m)
    p /= p.sum()
    b = int(RNG.choices(range(spec.N_ACT), weights=p)[0])
    logp = float(np.log(max(p[b], 1e-12)))
    rep = spec.decode_bucket(b, req)
    amt = int(rep.get("amount") or 0) if rep.get("action") == "raise" else 0
    H.seq += 1
    log({"pid": MYPID, "hand": H.hand, "seq": H.seq, "st": st,
         "seat": H.seat, "tokens": toks.tolist(),
         "ctx": [round(float(v), 6) for v in ctx],
         "legal": legal.tolist(), "b": b, "logp": round(logp, 6),
         "amt": amt, "pot": pot})
    return rep


def handle_discard(msg):
    hole = card_ids(msg["hole_cards"])
    board = card_ids(msg["board"])
    toks = np.zeros(spec.N_TOK_DISC, dtype=np.int64)
    toks[0:len(hole)] = hole
    toks[3:3 + len(board)] = board
    ctx = np.zeros(spec.N_CTX_DISC, dtype=np.float32)
    st = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3}.get(len(board), 3)
    ctx[st] = 1.0
    ctx[4] = (msg.get("pot") or 0) / 200.0
    ctx[5] = (msg.get("my_stack") or 0) / 200.0
    ctx[6] = (msg.get("opp_stack") or 0) / 200.0
    for i, c in enumerate(hole):
        r = (c - 1) % 13
        s = (c - 1) // 13
        ctx[7 + i * 3] = r / 12.0
        ctx[8 + i * 3] = 1.0 if sum(1 for x in hole if (x - 1) % 13 == r) > 1 \
            else 0.0
        ctx[9 + i * 3] = sum(1 for b in board if b and (b - 1) // 13 == s) / 5.0
    ctx[15] = len(board) / 5.0
    idx = int(np.argmax(DISC.forward(toks[None], ctx[None])[0]))
    return {"type": "discard", "card": msg["hole_cards"][idx]}


def safe_action(msg):
    legal = msg["legal_actions"]
    if msg.get("to_call", 0) > 0:
        return {"type": "action", "action": "call" if "call" in legal
                else ("fold" if "fold" in legal else legal[0])}
    return {"type": "action", "action": "check" if "check" in legal
            else ("fold" if "fold" in legal else legal[0])}


def main():
    load_models()
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        msg = json.loads(raw)
        t = msg.get("type")
        try:
            if t == "new_hand":
                H.hand = msg["hand_id"]
                H.seat = _SEATS[msg["seat"]]
                H.hole = list(msg["hole_cards"])
                H.opp_disc = None
                H.seq = 0
            elif t == "discard_result":
                H.hole = list(msg["hole_cards"])
                H.opp_disc = msg["opp_discard"]
            elif t == "action_request":
                sys.stdout.write(json.dumps(handle_action(msg)) + "\n")
                sys.stdout.flush()
            elif t == "discard_request":
                sys.stdout.write(json.dumps(handle_discard(msg)) + "\n")
                sys.stdout.flush()
            elif t == "hand_result":
                log({"pid": MYPID, "hand": msg["hand_id"],
                     "net": int(msg["net"]), "r": True})
            elif t == "match_end":
                break
        except Exception:
            sys.stderr.write(traceback.format_exc() + "\n")
            sys.stderr.flush()
            if t == "action_request":
                sys.stdout.write(json.dumps(safe_action(msg)) + "\n")
                sys.stdout.flush()
            elif t == "discard_request":
                hole = msg["hole_cards"]
                lowest = min(hole, key=lambda c: spec.CARD_ID[c])
                sys.stdout.write(json.dumps({"type": "discard",
                                             "card": lowest}) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    main()
