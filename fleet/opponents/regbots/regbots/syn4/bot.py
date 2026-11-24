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

if __name__ == "__main__":
    pb.run_bot(sy.decide, sy.discard, sy.on_result)
