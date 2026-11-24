# Opion: an RL poker agent

Opion is a transformer policy for **Pokerbots 2026**, a heads-up no-limit poker competition
playing **Discard Hold'em** (Texas Hold'em with three hole cards and a simultaneous, exposed
one-card discard after the flop). The agent was distilled from a hand-crafted teacher bot,
refined with PPO self-play, and shipped as a single-file pure-NumPy bot that talks to the
match engine over stdin/stdout.

```
hand-crafted teacher ──▶ instrument for soft labels ──▶ 40-sandbox data fleet
        │                                                      │
        ▼                                                      ▼
  25.4M labeled decisions  ──▶  transformer distillation  ──▶  PPO self-play
        (KL 0.090, 96.5% agreement)          (adaptive KL tether, 15+ iterations)
                                                               │
                                    pure-NumPy port ◀──────────┘
                                     1.2 ms/decision ──▶ submission bot.py
```

## Results

| Stage | Result |
|---|---|
| Distillation | 25.4M teacher decisions (22.1M actions + 3.3M discards), act model KL 0.090, 96.5% argmax agreement; discard model 89.2% (30.8 points over the best heuristic baseline) |
| Test partition | Test metrics ≥ val on every metric (no leakage, no overfit); action mix matches the teacher within 0.3 points in every bucket |
| RL self-play | 15+ PPO iterations on real-engine rollouts; probe KL 0.375 → 0.194; screens near breakeven vs the teacher and positive vs the team's frozen tournament bot |
| Inference | Pure-NumPy forward pass, 1.2 ms/decision single-threaded (budget: 10 ms on 1 vCPU) |
| Submission validation | 5/5 official-format 400-hand matches won, 0 protocol notices, 0 bytes on stderr, 0.00 s of the 30 s time bank used, 177 MB peak RSS (limit 1 GB) |

## What makes it interesting

**Soft-label distillation of mixed strategies.** A poker policy IS its randomization: where the
teacher raises 60% / calls 40%, a one-hot-labeled student learns "always" one of them and becomes
exploitable. The teacher bot was instrumented to expose its intended distribution at every
decision (bluff frequencies, CFR+ subgame strategies, raise-size weights), and the student trains
on KL to those soft labels. Frequency fidelity, not just argmax accuracy, was a formal gate.

**A leaky feature that was caught before it shipped.** The v1 student agreed with the teacher on
96.6% of decisions but could not raise at all in live play: one context feature (own
committed-to-pot ratio) perfectly separated raise records, so the model learned "amt = 0 means
never raise" as a shortcut. A sandbox dummy test exposed it, the feature was zeroed everywhere
(training, validation, live inference, RL tether probe), and the model was re-distilled. The
honest state-only version scores lower on imitation metrics and plays far better.

**Evaluation designed before training.** The RL rulebook was frozen before any run: disjoint seed
namespaces for training, checkpoint selection, and a blind final; paired matches on identical
seeds with mirror legs (same deals, seats swapped) so card luck cancels; promotion only when a
confidence interval excludes zero; win rate reported everywhere and trusted nowhere.

**Competition-shaped engineering.** The PyTorch model is ported to pure NumPy with a diff-test
gate (10k+ states, max logit delta under 1e-3), weights stored fp16, and the whole bot packed
into one 4 MB file that fits the rulebook's 1 vCPU / 1 GB / 5 MB / 5 s-per-reply envelope.

## Repository layout

```
plan.md                 full build log and design doc (the project's source of truth)
fleet/                  data generation: instrumented teacher, engine-side logger,
                        E2B sandbox orchestration (run_shard.py), packing, QC, census
fleet/teacher/          the teacher bot (elite.py) and its soft-label instrumentation
fleet/opponents/        opponent pool: regression bots, parametric styles, gen 8.6 package
train/                  distillation: model.py, data.py, train.py, final checkpoints,
                        Colab notebooks (train, disc, re-distill, locked test quiz)
rl/                     PPO self-play: numpy policy port, student rollout bot, iteration
                        drivers, automated Colab PPO loop (ppo_auto/), champion checkpoint
rl/ckpts/               policy_np_iter14 (champion) + policy_rl_iter14.pt
submission/             bot.py, the self-contained competition submission (weights embedded)
```

## Running

The submission bot needs Python 3.11, NumPy, and nothing else:

```
python submission/bot.py
```

Feed it the organizers' engine over stdin/stdout (any starter-kit match runner works). To
rebuild the pipeline: `fleet/` generates and packs data on E2B sandboxes, `train/` notebooks
run distillation on a T4 GPU, and `rl/` runs the PPO iteration loop (fleet rollouts plus the
automated Colab update in `rl/ppo_auto/`).

## Not included

- The 25.4M-decision dataset (~460 MB packed). `fleet/run_shard.py` regenerates it; `SHA256`
  manifests in `fleet/` describe the exact waves used.
- Per-iteration RL checkpoints (only the final champion is kept).
- The organizers' rulebook and problem statement, and the team's earlier hand-crafted bot line.

## Competition context

Pokerbots rules: bots in Python 3.11 (stdlib + numpy only), 1 vCPU and 1 GB RAM, no network,
read-only filesystem, 5 MB submission cap, cumulative time bank per leg. All code here is the
team's own work; the game uses virtual chips only. Written by Rehan Akhtar.
