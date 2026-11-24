"""bot.py — Synergy: Discard Hold'em competition bot (Pokerbots Synergy'26).
Revision 2 ("gen6"): direct 7-card evaluator (4.65x faster, exact-equivalent,
verified on 700k+ hands) + doubled Monte-Carlo sample sizes at the same wall
time. Policy unchanged from the tournament-validated generation.

Architecture: real-time range-conditioned equity engine.
  - one Monte-Carlo sample per street, cached: new information is folded in by
    importance re-weighting (preflop-action bands, exposed-discard synergy
    model, street-aggression strength filter), never by re-simulation
  - exact river enumeration over all C(45,2) opponent holdings
  - opponent modeling: street-resolved fold/call/aggression stats + showdown
    calibration that MEASURES how informative their actions are (vs random
    aggressors the conditioning flattens back to uniform - never confused)
  - value bets sized by equity vs their continue range; fold-EV gated bluffs
  - merged flop pass drives both betting and the discard decision
  - late-leg variance overlay: protects a lead / chases a deficit (point-max)
  - adaptive sample sizes from the live time bank; hard legality guards and
    legal-action fallbacks on every reply (stdout = protocol JSON only)

Files: pbcore.py (protocol loop + 7-card evaluator + preflop table loader),
synergy.py (strategy), data/preflop_equity.npz (offline-built 3-card equity
table over the 1755 isomorphic classes). Stdlib + NumPy only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbcore as pb
import synergy as sy

# Test-only decision logger for the offline EV-loss evaluator (gen7/evloss).
# Never active in submission (no SYN_EVALLOG env in prod; writes nothing).
_EVALLOG = os.environ.get("SYN_EVALLOG", "")


def _log(tr, req, reply):
    if not _EVALLOG:
        return
    try:
        import json
        rec = {"hand": int(getattr(tr, "hand_id", 0)),
               "street": getattr(tr, "street", "?"),
               "hole": [int(c) for c in getattr(tr, "hole3", [])],
               "board": [int(c) for c in getattr(tr, "board", [])],
               "d": None if getattr(tr, "opp_discard", None) is None
                    else int(tr.opp_discard),
               "pot": int(req.get("pot", 0)), "to_call": int(req.get("to_call", 0)),
               "my_bet": int(req.get("my_bet", 0)),
               "opp_bet": int(req.get("opp_bet", 0)),
               "min_raise": int(req.get("min_raise", 2)),
               "max_raise": int(req.get("max_raise", 200)),
               "legal": list(req.get("legal_actions", [])),
               "hist": [[h.get("street"), h.get("actor"), h.get("action"),
                         h.get("amount")] for h in (req.get("history") or [])],
               "reply": reply}
        with open(_EVALLOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def decide2(tr, req):
    r = sy.decide(tr, req)
    _log(tr, req, r)
    return r


def discard2(tr, req):
    r = sy.discard(tr, req)
    if _EVALLOG:
        try:
            import json
            rec = {"hand": int(getattr(tr, "hand_id", 0)), "street": "discard",
                   "hole": [int(c) for c in getattr(tr, "hole3", [])],
                   "board": [int(c) for c in getattr(tr, "board", [])], "d": None,
                   "pot": int(req.get("pot", 0)), "to_call": 0, "my_bet": 0,
                   "opp_bet": 0, "legal": ["discard"], "hist": [],
                   "reply": r}
            with open(_EVALLOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass
    return r


if __name__ == "__main__":
    pb.run_bot(decide2, discard2, sy.on_result)
