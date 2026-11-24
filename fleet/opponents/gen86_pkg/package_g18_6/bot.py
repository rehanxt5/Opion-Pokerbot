"""bot.py — Synergy v8.3: Discard Hold'em competition bot (Pokerbots Synergy'26).

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

Gen 8.2 additions: nnu.py (neural range prior — pure-numpy MLP trained
offline on 67k labeled decision points; maps their observed action line +
board texture + measured profile to a category distribution and re-weights
the opponent-range sample by likelihood ratio; dormant vs honest-gated and
white-box-exploited opponents; realized-EV governor), data/nnu_w.npz.

Gen 8.3 additions (SYN_G83, default on; sub-flags SYN_G83_D/_V/_T/_P for
ablation): value-density discipline — the repair kit for exact-equity
"DL-solver" opponents (measured vs elite.py: 8.2 lost the showdown war
-4824/4000 hands on 3 leaks: river bluff-catching 2/29, thin value bets
losing 44/65, kicker-war stack-offs):
  - G83_D  sharpened discard-explain range read: vs measured predictable
    discarders the kept/non-kept likelihood contrast deepens toward a full
    Bayesian ratio (mirroring their own discard_likelihood models of us)
  - G83_V  bet-range equity: their turn/river bet is faced as a narrowed
    top-q range (q = measured aggression frequency scaled by bet size);
    call needs eq vs THAT range; monsters only raise; big bets never get
    raised short of eq_b > 0.80 (kicker-war repair); margin scales with
    measured value density of their shown bets
  - G83_T  thin-value thinning: when our showdown'd turn/river bets win
    < 40%, the value ladder bets only beats-their-call-range hands and
    river bluff frequency is trimmed
  - G83_P  preflop guard: vs measured heavy 3betters the call3 fold
    schedule tightens one tier; vs 3betters whose shown hands are strong
    the value-jam line needs a tighter percentile
All gates are per-leg MEASURED (no opponent hardcoding); SYN_G83=0
restores exact 8.2 behavior.

Files: pbcore.py (protocol loop + 7-card evaluator + weighted CFR+ street
solver + preflop table loader), synergy.py (strategy), nnu.py (g82 neural
range prior), data/preflop_equity.npz (offline-built 3-card equity table
over the 1755 isomorphic classes), data/nnu_w.npz (g82 MLP weights).
Stdlib + NumPy only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbcore as pb
import synergy as sy

if __name__ == "__main__":
    pb.run_bot(sy.decide, sy.discard, sy.on_result)
