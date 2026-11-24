"""bot.py — Synergy v6: Discard Hold'em competition bot (Pokerbots Synergy'26).

Architecture: v4-strength robust core + EXPLOITABILITY-GATED real-time
subgame solving.

  - robust core (always on): one Monte-Carlo sample per street, cached and
    importance re-weighted (preflop-action bands, exposed-discard synergy
    model, street-aggression strength filter, showdown calibration that
    MEASURES how informative their actions are); exact river enumeration;
    our own discard excluded from their sampling deck; conditioned-equity
    value ladder with fold-EV-gated bluffs and texture sizing
  - REAL-TIME CFR+ SUBGAME SOLVING (river exact combos; turn and flop via
    rollout-value worlds = depth-limited solving), with empirical
    best-response blending — but ONLY while the opponent is measured as
    statically EXTREME (calling stations, over-folders).  vs solid
    adaptive opponents a best-response-to-estimated-range is
    counter-exploited over a long match (measured -750 chips/400 hands vs
    a v4-class peer; win-rate alone is NOT a safe open signal), so the
    solver gate closes and play falls back to the robust ladder
  - calibrated on measured per-archetype solver value: station +6700
    (open), over-folder +656 (open), maniac -664 (closed), solid
    equity/CFR bots -500..-900 (closed)
  - bank-fraction solve governor: solving downgrades (turn+flop -> turn+
    river -> river -> off) as the leg time bank drains, so an all-match
    exploit crush cannot forfeit on time; hard legality guards and
    legal-action fallbacks on every reply
  - late-leg variance overlay: protects a lead / chases a deficit (point-max)

Files: pbcore.py (protocol loop + 7-card evaluator + weighted CFR+ street
solver + preflop table loader), synergy.py (strategy),
data/preflop_equity.npz (offline-built 3-card equity table over the 1755
isomorphic classes). Stdlib + NumPy only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbcore as pb
import synergy as sy

if __name__ == "__main__":
    pb.run_bot(sy.decide, sy.discard, sy.on_result)
