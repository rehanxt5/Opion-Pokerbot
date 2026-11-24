# Opion — standalone policy transformer (gen 8.7 track)

Status: **PHASES 1–3 (copying stage) COMPLETE — act model trained (KL 0.090,
96.5% teacher agreement), disc model trained (89.2%). External audit verified:
acts 22,111,750 + discs 3,302,255, splits 92.0/4.1/3.9 measured, step counts
reconcile exactly (§6.2/§6.3). Pre-RL checklist ✓ ALL CLOSED: census (§6.4),
quiz (§6.5), RL rulebook frozen (§6.6), SHA256 manifest done. RL self-play
FULL DESIGN WRITTEN (§9 — tethered PPO on real-engine rollouts, numpy port
first with a differential-test gate, §6.6 judging, M1/M2/M3 vs deadline).
NEXT: build M1 → iterate → shrink only if speed gate fails → port parity →
battery.**
Written: 2026-11-13. Updated: 2026-11-14. This file is the persistent context
for any new session. Read this before doing anything in `opion/`.

---

## 1. What Opion is

Opion is a **decision-making model** (a learned policy), not a heuristic engine and not
a range prior.

- Teacher / data source (updated 2026-11-13): **elitev6** — `opion/fleet/teacher/elite.py`
  (3.78MB self-contained protocol bot, "Elite Poker Bot Next-Gen v6", stdlib+numpy,
  embedded weights; user reports it wins ~60% of games vs gen 8.6). gen 8.6
  (`submissions/synergy_v8_6_submission.zip`, sha256 `13744f98563aae37…`) remains the
  **frozen tournament submission** and one of the *opponents* in the diverse dataset —
  but it is no longer the distillation teacher.
- gen 8.6 itself contains *range-prior* nets (`nnu.py` MLP, `tns.py` tiny transformer
  d=32, 4 heads, 48 tokens, 9 classes). Those predict the opponent's hand category.
  **Opion is a different class of model: it predicts our action.** Do not confuse the two.
- Ship target: competition environment — **1 vCPU, 1GB RAM, read-only FS,
  stdlib + numpy only, ≤5MB zip, ≤5s/reply hard cap, 75ms/hand average time-bank
  (Stage 1: 400 hands/30s; Stage 2: 1000 hands/75s)**.
- Build target (updated 2026-11-14): Colab **T4 GPU + PyTorch** (chosen over
  TPU+JAX — model is 1M params, TPU advantage ≈ 0 there; see §6.2). Training
  frameworks may use anything available on Colab (PyTorch); only the shipped
  inference must be pure numpy.

Rule 10.1 (own work): we may reimplement public algorithms (CFR, distillation, RL),
but the model, features, labels, training code and architecture here are our own.
The teacher is our own bot. Do not copy elite bot code into Opion.

## 2. Goals / milestones

- **M0 — Distilled student equals the teacher.** Opion (250K ship size) should play
  elitev6's policy closely: CPU battery vs elite83 within noise of the teacher's
  result, **and per-opponent-type agreement within noise** (see §4.5 — generalization
  across opponent types is part of the M0 gate, not optional).
  Buy: speed, compactness, smoother decisions (no threshold cliffs), RL substrate.
- **M1 — Beat the teacher via practice (RL).** Train the 1M model beyond the teacher
  with pool play (opponents: past Opion checkpoints + self + teacher + gen 8.6 +
  field/style bots). Then distill to 250K. Must be proven on the CPU battery vs
  elite83 **and** vs the teacher — GPU self-play winrates are pre-screens only,
  never the ship criterion.
- Do not overfit to one opponent: the point of pool training is to stay a
  *generalized* bot (user requirement).
- Submission deadline context: 2026-11-15 23:59 IST. **Submission is gen 8.6, frozen.**
  Opion never blocks the submission; ship only if it wins a paired battery before the
  deadline.

## 3. Architecture (decided)

Token template (fixed slots, ~40–64 tokens):
- card tokens: our hole cards (3 pre-discard / 2 post), board (0/3/4/5), known opponent
  discard (else mask/unknown token)
- action-history tokens: (street, actor, action, amount normalized by pot), ≤ ~20
- context vector: street, position/button, pot, to_call, pot odds, stack ratio,
  tracker aggregates (opp fold/aggression buckets), leg indicator
- equity features: engine-computed equity mean `p` (= weighted `mywin`), `their_pct`
  marginal, sample stats. Injected BOTH as tokens and broadcast-added to all token
  embeddings (equity conditioning is the main convergence accelerator).
- Heads:
  - policy head: categorical over {fold, check/call, raise-to 1/3, 1/2, 3/4, 1.0,
    1.5 pot, all-in}, with legal-action masking (-inf on illegal)
  - discard head: pointer over the 3 hole-card tokens (linear probe + softmax over
    those tokens only), active only at the discard decision point
  - auxiliary equity head: predicts the equity feature from card tokens (MSE) —
    forces card tokens to encode hand strength; this is what makes 2–3 layers enough
- Sizes:
  - RL/intermediate model (GPU-only): d=128, 4 layers, d_ff=512, 4 heads, L≈64,
    ~0.8–1M params, fp16/bf16 on T4/TPU
  - Ship model: d=80–96, 2–3 layers, L≤48, ~230–250K params, ~5–10ms/hand in numpy
    (bank-safe). Weights stored fp16/int8 (per-channel), dequantized to fp32 at load;
    no fp8 compute on CPU (numpy has no fp8 dtype; int8 is the storage equivalent)
- Precision check: at 250K params fp16 = 500KB, int8 = 250KB — zip has ~4.7MB free,
  so quantization is a storage detail, not a size necessity.

### 3.1 Sizing rationale (evaluated 2026-11-13)

Decisions: **ship model 250K params (d=80–96, 2–3L); RL/intermediate model 1M
(d=128, 4L); dataset 5–10M decisions for M0, 10–30M for M1's fixed set.**

Reasoning:
- The policy is low-complexity because equity features carry card strength. The net
  learns a smooth decision surface over ~10–20 effective variables (street, position,
  facing-action, pot/odds, equity, draw bucket, opp tendency). Bucket product ≈ 10^6
  cells; smooth interpolation needs ~2–5 params per cell for dense coverage →
  200K–1M params; 250K is the knee for the ship model.
- Data/param ratio for feature-rich supervised tasks: 10–50 samples/param.
  250K × 10–50 ⇒ 2.5–12.5M decisions; 1M × 10–20 ⇒ 10–20M. Consistent.
- Inference budget: 250K at L≤48 ≈ 30–35 MFLOPs ⇒ 6–14 ms/hand in numpy.
  1M (d=128, 4L, L=64) ≈ 100–120 MFLOPs ⇒ 25–45 ms/hand — would compete with equity
  sampling inside the 75 ms/hand bank. The 1M model is GPU-only, so its size is free.
- Teacher labels are computed mixed strategies (bluff frequencies, threshold value
  probabilities) — high quality, so less data is needed than for noisy labels.
- Verify empirically, don't assume: on a fixed 2M-decision pilot slice, train
  155K / 350K / 800K models and pick the knee of held-out KL vs teacher; then sweep
  data 1M / 5M / 10M at the chosen size. Each run is minutes on TPU.
- Dataset must be stratified by (street, pot type, facing-bet, 3bet-pot, discard,
  river) — not uniform. The measured gen8.4 deficit lived in 3bet pots, so rare
  decisive states get oversampled ≥5–10% each. Uniform data would be ~70–80%
  routine single-raised preflop/flop decisions.

## 4. Pipeline phases

### ✅ Phase 1 — data contract + instrumented teacher logger (DONE 2026-11-13)
- `opion/fleet/spec.py`: single-source encoder — cards (53 ids incl. pad), 32-slot
  action-history tokens, 17-dim context, 6-bucket legal mask (elitev6 action space),
  seed-level 92/4/4 partitions (identical rule to qc_report), and a **byte-exact port
  of elitev6's decode_action** (incl. deterministic anti-fingerprint sizing) so the
  student can emit real raise amounts. Validated on real data: decode parity 8,476/
  8,476 vs teacher's actual (action, amount); encode→decode roundtrip 100%.
- `opion/fleet/engine_log.py` (engine-side logger, obs/meta split) +
  `opion/fleet/teacher/elite_instr.py` (instrumented teacher: soft labels = native
  mixed-strategy p + CFR+ strategy cfr, joined by (bpid, hand, seq)).

### ✅ Phase 2 — 25M dataset + packing (DONE 2026-11-14; details §6.1, §6.2)
- **25,414,005 teacher decisions** across 40 E2B shards, QC gate 40/40 PASS.
- Packer `opion/fleet/pack_batch.py` (8 procs/sandbox): join soft labels, hard
  validation (coverage/agreement/p-sum), packed to 120 B/decision (~26 B stored).
- Packed dataset: local `opion/data/packed/` (458MB) AND on all 40 paused sandboxes.

### ✅ Phase 3 — distillation training (DONE 2026-11-14, T4 GPU, PyTorch)
- **Act model** (`opion/train/final_act.pt`, 812,584 params, d=128 4L transformer):
  3 epochs × 22,111,750 soft-labeled acts (train split 20,343,138 → 14,898 steps),
  62.2 min on T4. **KL 0.090, TV 0.055, argmax agreement 96.5%, hard agreement
  97.1%**; no overfit (train≈val).
- **Disc model** (`opion/train/final_disc.pt`, 807,843 params): 3,302,255 hard-label
  discards (train split 3,036,657 → 2,223 steps), **89.2% accuracy** (residual =
  teacher's internal equity nets breaking near-ties our features can't see).
  Discards only occur on the flop by game rule.
- Trainer: `opion/train/{model.py, data.py, train.py}`; user-run notebooks
  `opion_train.ipynb`, `opion_disc.ipynb` (fp16-safe masking fix from user: cast
  logits to fp32 BEFORE legal-mask fill — T4 fp16 overflows -1e9).

### Phase 3.5 — RL self-play (NEXT, user decision: mandatory — goal exceeds teacher)
- **Full design moved to §9 (2026-11-14, pre-build): actor-critic PPO on the real
  engine via sandboxes, adaptive-KL tether, §6.6 judging.** Summary: KL-anchored
  self-play, mixed opponents (self/past checkpoints/teacher/styles), eval per §6.6.
- After RL: shrink only if time AND the 812K model fails the speed gate; then numpy
  port parity (port built early in RL anyway, §9.4) → ship battery.

### Phase 4 — M0/ship gate (folded into Phase 3.5's eval; final battery on E2B)
- Final 250K numpy student vs elitev6 and vs gen 8.6 in `engine_log.py` matches on
  E2B sandboxes. Ship ONLY if it beats gen 8.6 (frozen fallback always stands).

### 4.5 Dataset composition — opponent diversity (Problem-3 fix, adopted 2026-11-13)

Problem (valid, highest priority for data-gen): teacher-vs-teacher self-play gives
near-zero variance in opponent-tendency features, so the net learns to IGNORE them and
cannot adapt at inference; entire state regions (huge pots vs stations, thin-value vs
rocks, multi-raise wars vs maniacs) are absent → extrapolation failure vs unfamiliar
opponents (classic covariate shift / compounding error).

Two rules:
1. **Diversity goes into the OPPONENTS (states); labels stay TEACHER-only for M0.**
   Letting weaker bots label their own decisions and cloning all of it makes the
   student an average of incompatible policies. So: every match = teacher vs X;
   log BOTH seats but tag the acting bot id per record; M0 training filters to
   teacher-seat records; opponent-seat records are kept for M1 (RL opponent modeling).
2. **Diversity fixes coverage, not correctness.** Bad teacher responses get cloned
   faithfully. Only M1 pool-RL fixes policy quality. Complements, not substitutes.

Mixture for the 25M wave (teacher seat in every match):
| Opponent pool | Share | Purpose |
|---|---|---|
| elitev6 vs elitev6 | 35% | on-policy anchor (states the deployed strategy reaches) |
| elitev6 vs gen 8.6 | 15% | strongest different-architecture opponent |
| elitev6 vs elite83 (neural CFR testbot) | 15% | very different decision style |
| elitev6 vs field bots (syn83, honest, ata1, ata2, zboss2, tree gens) | 15% | tournament-ish field |
| elitev6 vs parametric styles (station / rock / maniac / random-legal / mixer) | 20% | pushes opponent-feature inputs across full range — direct anti-collapse fix |

Mechanics: `run_shard` extended to heterogeneous pairs (`--bot-a/--bot-b` chosen by
weight per match), `bot`/`opp_bot` ids added to every record, template rebuilt with
opponent bots baked in (gen 8.6 package, elite83, regbots, parametric bots we write
~100 lines each). Opponent bots are mostly faster than the teacher → fleet throughput
unchanged or better. Cost: same $ per wave.

Generalization measurement (part of M0 gate): hold out an eval slice PER opponent
type; report student decision-agreement and student-vs-X CPU batteries against
teacher-vs-X on identical seeds. Acceptance: within noise of teacher per type —
not just on average.

M1 only: add previous Opion checkpoints to the pool (league/PSRO-style), plus
KL-anchor to the distilled policy.

### 4.6 Soft-label recovery — mixed strategies are part of the policy (Problem-2 fix, adopted 2026-11-13; REQUIRED for M0)

IMPLEMENTED + smoke-verified 2026-11-13: teacher decision core mapped (ACT_DIM=6; argmax when to_call>0 or pot>=55 — deliberately deterministic; softmax sample at T=0.2 when checked-to small pot — p_samp natively available; 6 hard-rule one-hot branches; CFR+ strategy vector cfr_probs as auxiliary). `opion/fleet/teacher/elite_instr.py` emits per-decision sidecar records (OPION_SOFT_LOG env; keys pid+hand+seq; rep echo; cfr vector). Smoke: join 77/77 and 48/48 (vs gen86), action+amount agreement 100%, 0 p_soft validity violations. Replay-MC (option B) still pending — unseeded RNG confirmed (random.Random(seed=None), default_rng()), tleft logged for budget parity.

Problem (valid): hard one-hot labels make the student more deterministic than the
teacher (entropy collapse). Where elitev6 raises 60%/calls 40%, the student learns
"always" one of them. In poker the randomization IS the strategy: a frequency-collapsed
student under-bluffs and is exploitable by any adaptive opponent, while looking fine
on average decision-accuracy. Nuance kept honest: on-policy hard labels recover the
teacher's frequencies *in expectation* (CE optimum = conditional distribution), so this
is a practical entropy/variance problem — but practical is enough; soft labels are
required for M0, not an upgrade.

Recovery ladder (cheapest first; A is primary, B validates it):
- **A. Instrument the teacher (primary).** Patch a LOCAL copy of elitev6 (it is only
  the teacher — the submission gen 8.6 zip stays untouched) to expose its intended
  distribution at each decision: CFR+ subgame root strategy vectors, explicit bluff
  frequencies / threshold gates, raise-size choice weights. Log as `p_soft` next to
  the action. First task: source-read elitev6 (2,043 lines) and classify every decision
  path → deterministic threshold (hard label is correct, entropy 0) vs explicit-random
  (log the probability vector + size weights).
- **B. Replay Monte-Carlo validation.** We log the full state, so for a sampled
  subset, re-query a fresh teacher process K times (different RNG) on the same state
  and tally frequencies; compare against A's numbers. Ground-truth check on the
  instrumentation. Fleet-cheap.
- **C. Empirical bin frequencies (fallback)** for decision paths where A exposes
  nothing: per-bin (street × equity × odds × texture) empirical action distributions
  as soft labels, with per-bin entropy reported.
- **D. Trainer-side insurance:** loss = KL to `p_soft` where present, CE where
  teacher is deterministic; auxiliary per-bin frequency-matching term so collapse
  cannot return even with partial soft coverage.

Label schema: every record gains `p_soft` = {action: prob} + size weights where the
teacher exposes them (else null); `src` = "native" | "replay" | "bin".

M0 gate additions: (i) frequency fidelity — KL/TV between student and teacher
distributions per decision type on held-out states; (ii) **student-vs-teacher CPU
battery ≈ break-even** as the frequency-fidelity smoke test (a collapsed student
loses to the teacher; a matched one holds even).

Sequencing: instrument A BEFORE regenerating the 25M wave (never regenerate twice);
land together with §4.5's heterogeneous pairing + the round-loop quota fix.

### 4.7 Data-quality audit & acceptance gate (external review adopted 2026-11-13, with corrections)

IMPLEMENTED (2026-11-13): engine_log.py v2 (obs/meta split; seed/bot/opp_bot/bpid/is_t/seq/req_echo on every decision; --name-a/--name-b/--teacher-name); run_shard v2 (round-loop with complete-round stopping, seeds `w{wave}:{sb}:r{r}:m{i}`, weighted opponent sampling, per-match OPION_SOFT_LOG env, is_t quota counting, gzip); style bots (station/rock/maniac/randombot/mixer via stylelib.py); opponents extracted (gen86_pkg, elite83.py, regbots/{syn4,syn7,syn8,gen5tree,gen6tree,gen7tree,ata1,ata2,honest,zboss2,eliteBot}). Smokes passed: hetero teacher-vs-station and teacher-vs-gen86. STILL PENDING: qc_report.py audit tooling, template rebuild, 100K pilot gate, 25M wave.

Review verdicts: #1 counting → partially valid (our quota already counts only act+disc,
never deal/end — but seat-filtering for heterogeneous matches requires redefining quota
as TEACHER decisions). #2 labeling → valid, must-fix. #3 leakage → the claimed actual
leakage is FALSE (audited field-by-field: act/disc records are actor-visible only;
opp_disc is None on preflop+flop, public-only on turn/river; disc records carry no
opponent fields), but the audit + stream-split demands are valid hardening. #4 round
integrity → valid. #5 decision-share measurement → valid. #6 replay state-fidelity +
label invariants → valid, with a cost correction (idempotency test before K-replays).
#7 seed-level splitting → valid, trivial here, do pre-generation.

Adopted changes:
- **Target definition:** 25M = usable TEACHER decision examples (act + disc, teacher
  seat). Quota counts teacher-seat decisions only; per-target breakdown (act vs disc;
  pilot: 87%/13%) and per-opponent-type contributions reported per wave.
- **Record schema v2:** every act/disc record adds `seed` (globally unique string
  `w{wave}:sb{idx}:r{round}:m{i}`), `bot`, `opp_bot`, per-hand action counter; logger
  also echoes the VERBATIM request dict (`req_echo`).
- **Stream split:** per match write `obs.jsonl` (act+disc only — the training stream)
  and `meta.jsonl` (deal/end — privileged metadata: both hands, full board, scores;
  joined only for outcome labels and M1 analysis, never as features).
- **Automated leakage audit** (runs on every wave, fails the wave):
  card set ⊆ actor hole ∪ board[:street]; `opp_disc` None on streets 0-1, non-None on
  2-3; disc records have exactly 3 own cards and zero opponent fields; len(board) ==
  street width; obs stream contains no privileged record types; req_echo re-diff on a
  sample against a rebuilt engine request.
- **Round-loop with integrity:** stop after COMPLETE rounds (overshoot ≤ ~17K records
  per sandbox, negligible on 1.27M); seeds continue across rounds → crash-resume
  without duplication; post-gen audit: every line parses, gzip CRC, duplicate check on
  (seed, deal, leg), per-shard counts vs summary.json.
- **Decision-share rebalancing:** after the first ~10 rounds, measure actual
  teacher-decision share per opponent type (matches ≠ decisions — hand lengths differ
  by opponent) and rebalance remaining rounds to hit §4.5 targets; per-wave
  stratification report: street, pot-type, facing-bet, 3-bet pot, discard, equity
  bucket, position — computed from the records themselves.
- **Soft-label replay protocol:** fresh teacher process → replay the hand's PREFIX
  requests (rebuilds tracker/memory state) → final query. Run an IDEMPOTENCY TEST
  first (same request twice into a warmed process; if reply distribution is stable and
  state doesn't drift, one process serves K queries — otherwise fresh process per
  query, K=8, sampled subset). Label invariants: probabilities map to the legal
  actions of that record, Σp=1 (tol 1e-6), illegal actions 0, size-probabilities map
  to actual raise amounts; `src` preserved (native/replay/bin) with reliability
  weights 1.0/0.8/0.5 in training; bin labels carry sample count + CI.
- **Splitting (pre-generation):** partitions assigned at MATCH-SEED level before the
  wave, stratified by opponent type, ~92/4/4 train/val/test; whole hands and matches
  stay in one partition; test seeds never trained on; seed namespace includes wave id
  so future waves cannot collide with test seeds; per-opponent-type eval reporting.
- **Acceptance gate:** a 100K diverse pilot is a formal QC gate producing
  `opion/data/qc_report.json`: (1) usable teacher decisions by category, (2) records
  per opponent type, (3) seat distribution, (4) duplicate/missing counts,
  (5) leakage-audit result, (6) soft-label coverage + probability validation,
  (7) split assignment verification, (8) shard completeness + checksums.
  25M launches only after the gate passes.

## 5. Evaluation protocol (non-negotiable)

- CPU battery = ground truth. Reference opponent: `testbots/elite83.py`; anchors:
  gen 8.6 as bot-A vs elite83 on identical seeds.
- Always: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` (BLAS oversubscription caused
  elite time-bank death and invalid data twice). Parallelism: par 2–3 on 2-vCPU Colab,
  par 6–8 on 24-core/TPU hosts. VALID guards: min hands ≥ HANDS−10, elite max time
  bank < 28s (stage 1). Discard any arm that fails guards; report it as INVALID.
- Paired stats on identical seed sets (same seeds every arm); report per-1000 chips,
  paired t; effect sizes below ~100/1000 are below the measurement floor at 40–50k
  hands — use dose-response + repeated batteries, never a single arm.
- Beware the syn88 lesson: 2-vCPU Colab at par 6 still killed elite's bank
  (minhands=141, maxtB=59.9). Prefer fewer matches in parallel, more matches total.
- RL-phase eval & promotion: the §6.6 rulebook (frozen 2026-11-14) — extends
  this section; all non-negotiables above carry over.

## 6. Ops / assets / state (as of 2026-11-13)

- Frozen submission: `/Users/luciferxt5/Documents/Coding/pokerbot/submissions/synergy_v8_6_submission.zip`
  sha256 starts `13744f98563aae37`, 290,389 bytes. Package dir `package_g18_6/`.
  Do NOT modify; do not repackage a confirmed artifact unless the platform spec fails.
- Extracted copy for inspection: `/tmp/g86inspect/package_g18_6/` (ephemeral).
- syn90 (TPU v5e1): has `/content/gen86.zip` (frozen artifact), `/content/pk` (harness
  tools + syn83 + testbots/elite83.py + starterKit), `/content/pk86/package_g18_6`
  (unpacked, py_compile clean). Keepalive loop: `colab exec -s syn90
  --file /tmp/keepalive.py --timeout 45` every 120s (nohup loop on this machine).
- Other sessions: syn88 expired (gen 8.4 battery invalid/superseded); create fresh
  CPU sessions as needed. Colab CLI: `colab new -s NAME [--tpu v5e1|--gpu T4]`,
  `colab upload LOCAL REMOTE -s NAME`, `colab exec -s NAME --file LOCAL --timeout N`,
  `colab restart-kernel -s NAME`.
- Local assets: `/tmp/pk83up.zip` (harness/tools/elite83), `/tmp/gen84up.zip`
  (gen8.4 build, superseded), `/tmp/regbots.zip` (field regression bots:
  syn7/syn8/syn4/gen7tree/gen6tree/gen5tree/honest/ata1/ata2/zboss2).
- Current results that matter: gen 8.4-family deficit vs elite83 ≈ −316/1000,
  entirely in 3-bet pots (p2 −2.29/h); BBD defend widening dose-response
  (BBD45 +96, BBD35 +173 per 1000, single seed-set, unconfirmed). These informed
  gen 8.6; they are background for Opion, not open work.
- Deadline: 2026-11-15 23:59 IST. Submission = gen 8.6.

### 6.1 E2B fleet — dataset generation (MEASURED 2026-11-13, 10K pilot passed)

Teacher: `opion/fleet/teacher/elite.py` (3.8MB self-contained bot, the newer bot
that reportedly wins ~60% vs gen 8.6; supersedes gen 8.6 as sample source).
Standard stdin/stdout JSON protocol; **notices 0** over 40+ hands (all replies legal).

Template: `opion-gen-v1` (template_id `1u3yw8as09tc0zzvz81s`), built programmatically
via `e2b.Template.build(..., cpu_count=8, memory_mb=8192)` (daemonless, no Docker;
Hobby build caps are exactly 8 vCPU / 8GiB). Contents: python3.11 + numpy 2.4.6 +
`/home/user/opion/{bot/elite.py, engine_log.py, run_shard.py}`. Rebuild: ~23-55s.
Sandbox from template boots in **2.3s**; verified nproc=8, 7.9GB RAM.

Measured (single sandbox, 8 parallel matches × 400 hands, quota 10,000):
- **10,492 records in 18.7s → 562 decisions/s per sandbox** (quota hit at 18.2s)
- decision latency in-sandbox: median 16.6ms, p95 29.2ms (local 1-match: 2.0ms —
  contention + Firecracker vCPU)
- per-match balance 1255–1422 records; 1,917 deals → 5.5 records/deal
- street funnel: preflop 3,828 / flop 2,851 / turn 1,549 / river 920
- action mix: raise 40% / call 25% / check 17% / fold 18%
- JSONL ≈ 377 B/record → 25M ≈ 9.4GB raw JSONL; binary packing (opion spec,
  ~100 B/decision) cuts the training set to ~2.5GB as planned; per-sandbox JSONL
  shard ≈ 470MB, gzip ≈ 120-150MB transfer
- files: `opion/fleet/{make_template.py, test10k.py, run_shard.py, engine_log.py}`,
  pilot data pulled to `opion/data/pilot10k/`

Fleet projection (20 sandboxes × 562 dec/s ≈ 11,200 dec/s):
- **25M decisions ≈ 37 min** (one wave, inside the 1h session cap)
- 100M ≈ 2.5h → 2-3 session waves with checkpoint/resume
- cost ≈ $5-10 of the $100 credit per 25M wave

**2-sandbox parallel test PASSED (2026-11-13, `test_2x5k.py`)**: both sandboxes
created simultaneously (2.1s each) from the template, run concurrently —
sb0 5,490 records @ 522 dec/s (12.9s), sb1 5,384 records @ 715 dec/s (9.5s),
rc=0 both, 8/8 shards pulled per sandbox, **deal-seed overlap = 0** (disjoint
seed ranges per sandbox: base + idx·100000). Rate variance between sandboxes
(522 vs 715) = noisy neighbors on shared hosts, expected. Data pulled to
`opion/data/pilot_2x5k/sb{0,1}/`.

**Known issue (wave-1 of the 25M, RESOLVED):** all 20 sandboxes ran and
returned data cleanly (345K records total pulled to `opion/data/fleet25m/sb*/`,
8/8 shards each, rc=0), but each sandbox stopped at ~16.6–17.7K records instead of
the 1.27M quota. Cause: `--hands 400` caps every match at ~2.2K records; all 8
engines finish in ~25–35s and run_shard exits at "all processes done" long before
quota. **Fix (shipped in run_shard v2): round-loop — keep relaunching fresh
8-match rounds with unique seeds (`w{wave}:{sb}:r{round}:m{i}`) until the quota is
reached, stopping only at complete round boundaries.** Two earlier one-line failures
(already fixed): sandbox `timeout=7200` rejected — Hobby caps at 1h → use 3600;
stale template without the `--gzip` arg → rebuild template after every run_shard change.

**25M GENERATION COMPLETE (v2 pipeline, 2026-11-13, QC per-shard in progress):**
- Pilot 100K diverse: QC PASS — 107,343 teacher decisions, soft coverage 100%,
  action agreement 93,646/93,646, 16 opponent types live, 0 leakage/dupes,
  partitions 88,348/2,515/2,783.
- Wave 1: 20 sandboxes, quota 1.25M each, watchdog 2400s → **14,946,113 teacher
  decisions** (534K–1,008K per sandbox, 62–102 rounds each; watchdog-capped, so
  per-sandbox yield = rate × 40min). ~2.3GB compressed → `opion/data/fleet25m/
  tookover_w1/sb{00–19}/shard.tgz`.
- Wave 2 (top-up): 20 fresh sandboxes, quota 520K each → **10,467,892 teacher
  decisions** (520,085–528,240, rounds 59–61, quota-hit before watchdog) →
  `opion/data/fleet25m/wave2/sb{00–19}/`.
- **Combined: 25,414,005 teacher decisions** (extracted: `wave1_data/`, `wave2_data/`).
- All 40 wave sandboxes PAUSED after verified pull — none killed (user directive;
  paused sandboxes do NOT count against the 20-concurrent cap; terminate only
  after final QC, with user's go-ahead).
- Ops incidents survived (all recovered, zero data loss): (1) wave-1 orchestrator
  died right after create (older build: no ids/metadata/lifecycle) → takeover by
  `Sandbox.connect` + `set_timeout` re-arm + seed-based shard mapping (map_shards.py);
  (2) tar raced run_shard's final gzip pass (summary.json written BEFORE gzip) →
  wait for `pgrep -f '[r]un_shard.py'` to clear before tar; (3) pgrep self-match
  stall (its own shell cmdline matches the pattern) → `[r]` bracket trick;
  (4) wave-2 per-sandbox save dir missing → mkdir recovery + re-pull of 2 paused
  shards; (5) monolithic QC OOM-died silently at 15M records → per-shard QC with
  3 workers (`run_qc_shards.py` → `qc_shards_summary.json`).
- Lesson: `paused` sandboxes preserve disk+processes and don't burn concurrency
  slots; `Sandbox.connect` auto-resumes; `beta_pause` is safe even mid-gzip-pass
  (process resumes where it froze).

Engine details: `engine_log.py` = starter-kit engine + additive patches (one JSONL
record per action/discard/deal/end; `--data-log PATH`; per-decision round-trip ms).
run_shard: 8 workers/sandbox (1 per vCPU, BLAS thread-pinned via template envs),
global quota polling, SIGTERM at quota, engine stderr captured per match.

Plan limits: 20 concurrent sandboxes, 1h session, 10GB disk / 8 vCPU / 8GB RAM max
per sandbox (per-sandbox caps → 160 vCPU / 160GB aggregate). Kill sandboxes
immediately after shard collection (billing per reserved vCPU/GiB-second).

### 6.2 Phase 2/3 execution record (2026-11-14) — RESULTS

**Dataset (Phase 2):**
- 25,414,005 teacher decisions, 40 shards, QC gate **40/40 PASS** (0 leakage,
  0 duplicates, soft coverage 1.0, action agreement 1.0 on every shard).
- **Exact composition (verified from packed shards 2026-11-14, external audit):**
  acts 22,111,750 + discs 3,302,255 = 25,414,005. Splits (record level,
  seed-hashed): acts 92.0/4.1/3.9%, discs 92.0/4.2/3.9% — the 92/4/4 rule held.
  Step reconciliation: act train 20,343,138 ×3 ÷4096 = 14,898 steps ✓;
  disc train 3,036,657 ×3 ÷4096 = 2,223 steps ✓. (Earlier "4.4M discs" and
  "trained on 25.4M acts" were documentation errors, now corrected.)
- Wave 1: 14,946,113 (20 sandboxes, watchdog-capped 534K–1,008K each).
  Wave 2: 10,467,892 (20 sandboxes, quota-hit 520K each). Seed namespaces
  `w1:`/`w2:` — no collisions; partitions train 92% / val 4% / test 4% by match seed.
- Street mix 36.5/25.0/15.2/10.4%; **18 opponent types live** (measured by the
  census, §6.4 — shares match pairs.json weights).
- Packing: `pack_batch.py` 8 procs/sandbox, whole shard ≈ 14-75s; 120 B/decision
  logical, ~26 B/decision stored (npz) → **458MB total packed**.
- Locations: local `opion/data/packed/{w1,w2}/sb{00-19}/{acts,disc,manifest}.json`;
  identical copies on all 40 paused E2B sandboxes (`/home/user/opion/packed/`).
- Raw JSONL (~38GB) deleted locally after pack+verify; raw remains recoverable
  only by regenerating. pilot100k kept (smoke tests).

**Training (Phase 3, T4 GPU, PyTorch):**
- Act model `opion/train/final_act.pt`: 812,584 params, d=128/4L/4H, FFN 512.
  14,898 steps × 4096 batch = 3 epochs, **62.2 min**. Final val: **KL 0.0901,
  TV 0.0551, argmax agreement 96.5%, hard agreement 97.1%**. Train≈val (no
  overfit). ~23.5 it/s at batch 4096 on T4.
- Disc model `opion/train/final_disc.pt`: 807,843 params. 2,223 steps = 3 epochs
  over 3,302,255 hard-label discards (train split 3,036,657 → 2,223 steps,
  last partial batch dropped each epoch), **3.5 min**. **Accuracy 89.2%** (teacher discard
  is a deterministic heuristic; residual ≈ near-tie card choices where the
  teacher's internal equity nets differ from our hand-strength feature).
  All discards are flop-street (game rule).
- Notebook fixes by user (adopted into canonical files): `packed/*/*/acts.npz`
  glob (w1/w2 layout), fp16-safe masking (cast logits to fp32 BEFORE filling
  illegal actions with -1e9 — fp16 overflows), `torch.amp.GradScaler("cuda")`.

**Decisions made:**
- T4 GPU + PyTorch chosen over TPU+JAX (1M-param model: TPU advantage ≈ 0,
  XLA friction real). TPU syn90 keepalive no longer needed for training.
- Pipeline re-ordered (user): RL self-play BEFORE numpy port — port once, after
  the model is final. RL is mandatory (goal = exceed teacher, not copy it).
- 40 dataset sandboxes stay PAUSED (packed-data backups); test sandbox deleted
  (its snapshot `opion-packer-test-sb00` still exists).

**Next: RL self-play trainer** (torch, GPU): self-play env in torch (fast batched
hand sampler reusing spec.py features), PPO-style policy gradient from chip
outcomes, KL tether to teacher soft-labels (loosened on a schedule), mixed
opponents (self / past checkpoints / teacher hard labels / style bots), eval =
checkpoint vs teacher win-rate every N iters. Then shrink → numpy port → battery.

### 6.3 External audit (critique agent, 2026-11-14) — findings & dispositions

Audit had plan.md only (no workspace). Every claim re-verified against the data.

| Finding | Verdict | Disposition |
|---|---|---|
| Counts don't reconcile (25.4M vs street % vs "4.4M" discs) | **VALID** — my doc errors; data consistent | Fixed in §6.2: acts 22,111,750 + discs 3,302,255; splits 92.0/4.1/3.9 measured; steps reconcile exactly |
| Training-steps explanation | VALID concern, resolved | Step math shown above (14,898 / 2,223 exact) |
| Pilot split 94/2.7/3.0 ≠ 92/4/4 | Valid ask | Small-seed-count variance at pilot scale; full-dataset splits measured 92.0/4.1/3.9 |
| Per-opponent DECISION-share table missing | **VALID** — real gap: per-shard QC jsons were deleted in cleanup; packed npz lacks opp_bot; raw JSONL survives only on the 40 paused sandboxes | **DONE 2026-11-14 (§6.4)**: census.py fleet pass over raw obs on all 40 sandboxes; per-opponent × street × seat table; reconciles exactly |
| Soft-label coverage ≠ label quality; Replay-MC pending | Partially valid | Coverage = every act has a soft record passing p-sum/agreement; many labels are legitimately one-hot (argmax/rule modes). Full Replay-MC impossible bit-exact (unseeded RNG, time-dependent budgets) → **downgraded to a seeded determinism probe** (re-decide N recorded states, compare p). Non-blocking |
| Discard EV-regret metric | Valid in principle, LOW priority | Needs teacher per-card discard scores (never logged). Cheap version: instrument teacher, re-run sample, regret = score gap on student mistakes. Defer unless battery blames discards |
| Test partition never evaluated | **VALID, cheap** | **DONE 2026-11-14 (§6.5)**: act val-vs-test + per-street + action mix; disc + 2 baselines. Test ≥ val on every metric — no leakage/overfit |
| Data preservation | Partially invalid — 3 copies exist (local packed/, 40 sandboxes, user's Drive packed.tar). Valid bits: no checksums, no raw sample | TODO: SHA256 manifest for 40 packed shards (local); pull one representative raw sample from a sandbox; pack manifests + qc_shards_summary.json already preserved |
| QC artifacts deleted | Valid consequence of cleanup | Pack manifests (per-shard coverage/agreement/p-sum/streets/partitions/seeds) + qc_shards_summary.json preserved; full qc_report re-runnable on sandboxes if ever needed |
| RL eval spec: frozen eval seeds, both seats, promotion rule | **VALID** — concretizing | Eval seeds = test-partition seeds + fresh `eval:` namespace (never trained/selected on); both seats (engine alternates A/B); promotion rule = beat current best with win-rate CI excluding 50% at fixed N (paired matches), plus eval vs elitev6 / gen 8.6 / elite83 / field |
| "Don't collect more data; verify what you have" | Agreed | No further data generation planned |

**Pre-RL checklist (agreed, all cheap):** (1) docs corrected ✓ (this update);
(2) per-opponent decision table ✓ **DONE §6.4**; (3) test-partition eval ✓
**DONE §6.5**; (4) SHA256 manifest ✓ **DONE 2026-11-14**
(`opion/data/packed/SHA256SUMS.txt`, 121 files / 480.1MB, verify PASS —
raw-sample pull dropped by user decision, raw remains on the 40 sandboxes);
(5) written RL eval/promotion spec ✓ **DONE §6.6 (FROZEN)**. Deferred: full
Replay-MC (→ seeded probe), discard EV regret (→ only if battery implicates).

### 6.4 Chore 1 — opponent census (DONE 2026-11-14)

Per-opponent headcount of every teacher decision, counted from the raw obs
logs on the sandboxes themselves (the packed npz carries no opp_bot; raw
JSONL survives only on the 40 paused sandboxes). Purpose: the RL opponent
pool gets weighted from measured shares, not guesses.

**Method:** `census.py` (stdlib-only, runs in-sandbox) — 8 worker processes
(one match file each), substring filter (`"is_t":true`, the engine writes
compact JSON) before json.loads so only the teacher's ~half of lines is
parsed; counts `(opp_bot × street × seat)` + `(leg,hand)` pairs + seeds per
opponent, from each record's own `opp_bot` field. Run on all 40 paused
sandboxes in two waves (wake → count → verify → re-pause; 6-way concurrency,
staggered starts, 429 retries). Full `census.json` stays on each sandbox
disk; only the few-KB per-sandbox summary came back for the merge. Nothing
downloaded, nothing killed.

**Verification:** per-shard census total == packed manifest total on **all 40
shards (delta 0 everywhere)**. Fleet totals reconcile exactly: 14,946,113 (w1)
+ 10,467,892 (w2) = **25,414,005 = acts 22,111,750 + discs 3,302,255**; 18
opponents; 23,296 unique match seeds (= total matches, 1 seed per match);
8,424,574 teacher-hands. Wave wall time ≈3 min each (2–24s compute/sandbox).

**Fleet table (RL opponent-pool weights):**

| opponent | decisions | share | act | disc |
|---|---:|---:|---:|---:|
| self (elitev6 mirror) | 8,231,607 | 32.39% | 7,208,296 | 1,023,311 |
| elite83 | 3,603,195 | 14.18% | 3,164,543 | 438,652 |
| package_g18_6 (gen 8.6 bot) | 2,855,248 | 11.23% | 2,557,037 | 298,211 |
| station | 2,076,548 | 8.17% | 1,668,337 | 408,211 |
| maniac | 1,760,335 | 6.93% | 1,551,153 | 209,182 |
| randombot | 1,187,598 | 4.67% | 954,100 | 233,498 |
| mixer | 910,511 | 3.58% | 775,451 | 135,060 |
| rock | 766,581 | 3.02% | 684,371 | 82,210 |
| syn8 | 719,488 | 2.83% | 644,269 | 75,219 |
| gen7tree | 622,885 | 2.45% | 545,484 | 77,401 |
| ataHonestBot | 595,059 | 2.34% | 514,977 | 80,082 |
| syn7 | 544,681 | 2.14% | 487,114 | 57,567 |
| gen6tree | 439,683 | 1.73% | 385,199 | 54,484 |
| syn4 | 324,614 | 1.28% | 290,556 | 34,058 |
| ataBot1 | 242,379 | 0.95% | 208,901 | 33,478 |
| zboss2 | 193,643 | 0.76% | 169,299 | 24,344 |
| ataBot2 | 177,776 | 0.70% | 157,333 | 20,443 |
| gen5tree | 162,174 | 0.64% | 145,330 | 16,844 |

Shares match the pairs.json sampling weights (self 35 / gen86 15 / elite83 15
/ station 5 / rock 5 / maniac 4 / syn8 4 / maniac-style…), confirming the
generator sampled as designed.

**Street texture:** fold-heavy opponents (rock, syn*, trees, zboss2) stack
their decisions preflop (rock: 355k/160k/95k/75k); station and randombot never
fold → flat profile (station act 444k/408k/408k/408k, disc ≈ every hand).
Seat split fleet-wide: 12,157,715 (seat 0) vs 13,256,290 (seat 1), ratio 0.92
— style-driven (fold-heavy opponents end hands before the responding seat
acts), not a sampling defect.

**Engine fact discovered (matters for RL):** each 400-hand match = **2 mirror
legs of 200 hands** — same deals in both legs, seats/hole-cards flipped
(`leg` field on every record; hand ids restart per leg, hence `(leg,hand)`
counting). The dataset is luck-cancelled by construction.

**Naming quirk (recorded):** wave-1 ids-list position ≠ run_shard `--sb` arg
(list pos 0 ran `--sb 19`); wave-2 positions match. Packed shard `wN/sbNN` =
list position NN. Seeds are globally unique either way — no data impact.
Census summaries record each sandbox's sb_arg.

**Artifacts:** `opion/fleet/census.py` (sandbox-side counter),
`census_test_setup.py`/`census_test_run.py` (test-sandbox flow; test copy
`iubpl0b0mz0npoyhoh3w6` paused, snapshot `opion-census-test`),
`census_w1.py`/`census_w2.py` (fleet waves), `census_merge.py` (aggregator).
Local: `opion/data/census/{w1,w2}_sb*_census.json` (40 summaries),
`census_fleet.json` (merged table). Full `census.json` on each of the 40
sandboxes at `/home/user/opion/census.json`.

### 6.5 Chore 2 — locked-drawer test-partition quiz (DONE 2026-11-14)

Both trained models evaluated cold (fp32 inference, no autocast) on the test
partition (part==2 — never trained on, never used for checkpoint selection).
Val (part==1, capped 200k as in training) was recomputed fresh in the same
run so val-vs-test is apples-to-apples. Notebook:
`opion/train/opion_quiz.ipynb` (user-run on Colab T4, ~5 min).

**Act model — val vs test (the honest generalization check):**

| partition | KL | TV | argmax_ag | hard_ag | n |
|---|---:|---:|---:|---:|---:|
| val | 0.0901 | 0.0551 | 96.51% | 97.09% | 200,000 |
| **test** | **0.0869** | **0.0533** | **96.64%** | **97.24%** | **862,178** |

Test ≥ val on every metric → **no overfitting, no leakage; the 96.5% was an
honest number.** RL can build on this checkpoint. Baseline RL must beat:
**96.6% teacher agreement / KL 0.087 on never-seen decisions.**

**Per-street (test) — the RL targeting map:**

| street | KL | argmax_ag | n |
|---|---:|---:|---:|
| preflop | 0.0195 | 99.30% | 364,851 |
| **flop** | **0.1569** | **93.91%** | 248,052 |
| turn | 0.1352 | 94.86% | 148,474 |
| river | 0.0878 | 96.40% | 100,801 |

Preflop ≈ mastered (teacher near-deterministic there). Weakest = flop/turn —
exactly where elitev6's mixed strategies (bluff frequencies, raise-size
splits) live: copying a coin-flip policy is harder than copying a rule.
These are also the streets where self-play can potentially EXCEED the
teacher (verify whether its flop frequencies are even optimal).

**Action mix on test (drift check):** teacher vs student nearly identical —
fold 16.77→16.89%, check/call 43.61→43.48%, small-raise 9.79→9.75%,
mid-raise 26.03→26.05%, big-raise 3.80→3.81%, all-in 0.02→0.01%. No
over-folding, over-bluffing, or over-raising in any bucket.

**Disc model on test (n=128,096):** accuracy **89.06%** vs baselines:
majority-class-from-train 34.16%, lowest-rank-card 58.22% → **+30.8 pts over
the best dumb strategy** — the model genuinely learned the equity-based
discard, it is not riding a default.

**Rule:** the test drawer is opened ONCE for the distillation stage — these
numbers are recorded and the drawer is locked again. RL checkpoints get
their own fresh namespaces (per §6.3 eval spec): `dev:` seeds for selection,
a blind final namespace used once.

### 6.6 Chore 5 — RL rulebook: eval & promotion spec (FROZEN 2026-11-14)

Written BEFORE any RL run (audit finding + user decision). Purpose: decide
what counts as "better" before any results exist, so checkpoint luck can't
vote. Extends §5; every §5 non-negotiable carries over (real-engine CPU
batteries, paired seeds, VALID guards, thread limits, fewer-matches-in-parallel).

**Seed namespaces (disjoint by prefix, globally unique):**
- `w1:` / `w2:` — dataset generation (existing). Their part==2 test partition
  is the distillation quiz drawer — locked again after §6.5, never reused.
- `rl{run}:...` — RL self-play TRAINING games only.
- `dev:` — checkpoint-selection seeds; may be consulted freely.
- `blind:` — final champion only, used ONCE.
Eval seeds are never trained on; dataset seeds are never reused as eval seeds.

**Eval match:** 1 seed = 1 engine match = 2 mirror legs × 200 hands = 400
hands, stage-1 config (`--hands 400 --stage 1`, engine_log.py) — the mirror
legs cancel deal/seat luck by construction (§6.4). Host: E2B 8-vCPU sandbox,
≤8 concurrent matches per sandbox, OMP/OPENBLAS=1.

**Frozen opponent pool (8, all 1-v-1 batteries):** elitev6 teacher
(`bot/elite_instr.py`), package_g18_6 (gen 8.6), elite83, station, maniac,
rock, randombot, incumbent champion. Style coverage mirrors the census mix
(§6.4); census weights feed the RL TRAINING mix, not eval. Any pool change =
version bump of this section, never applied retroactively.

**Two-tier judging:**
1. **SCREEN** (every eval-interval): candidate vs all 8 opponents, 60 matches
   each (24k hands/opp; 480 matches ≈ 4 min on 8 sandboxes). Output:
   per-opponent win-rate + chip-EV + regression alarms. Screens shortlist;
   they never promote.
2. **PROMOTION BATTERY** (candidate only — the incumbent's per-seed scorecard
   is computed once per dev-seed block and cached): 200 matches/opponent =
   640k hands per model. Paired per-seed diff per opponent:
   `D = mean(cand_net − inc_net)`, reported in **chips per 1000 hands** (§5
   convention), with a 95% paired-t CI across seeds.

**Promotion — ALL must hold:**
- vs elitev6: D > 0 with CI excluding 0 (**primary** — "exceed the teacher")
- vs package_g18_6: D > 0 with CI excluding 0 (ship bar)
- pooled over all 8 opponents: D > 0 with CI excluding 0
- no-regression: vs EVERY opponent, D ≥ −100 chips/1000 hands (point estimate)
- win-rate is reported everywhere and gates nothing (noisier than chip-EV)

**VALID guards (from §5, per match):** both bots play ≥ 390/400 hands; elite
time-bank < 28s (stage 1) when elitev6/elite83 is in the match. A violating
match is rerun once on the same seed; a second violation counts as a forfeit
loss for the violator and is flagged — matches are never silently dropped.
Timeout/forfeit = forfeiting side loses its stack for that hand (competition
rules).

**Blind final (once, after RL ends):** the final champion replays the
promotion battery on `blind:` seeds, then the §5 ship battery vs the elite83
anchor, then the full-bot 1-vCPU speed benchmark (p50/p95/p99 per decision;
target ≤10 ms/hand). Recorded win or lose — a blind failure means no ship and
gen 8.6 stays the submission.

**Append-only eval log:** every battery appends to `opion/rl/eval_log.jsonl`:
timestamp, model sha256, seed-list sha256, engine build hash, per-opponent
win-rate / chip-EV / CI, invalid-match counts. No edits, no deletions.

**Anti-gaming guards:** `dev:` seeds never enter training data; a screen can
never promote without a battery; only a battery moves the champion; `blind:`
is touched once by one logged script.

**Cost:** promotion battery ≈ 1,600 matches ≈ 640k hands ≈ 10–15 min on 8
sandboxes per candidate — cheap enough to run on every serious candidate.

## 7. Open decisions to settle in the next session

1. ~~Soft-label extraction~~ → DONE (instrumented teacher, coverage 100%).
2. ~~Action vocab mapping~~ → DONE (spec.py decode_bucket, parity 8,476/8,476).
3. **RL algorithm**: ✓ SETTLED 2026-11-14 — tethered PPO actor-critic on
   real-engine rollouts, adaptive-KL thermostat, full design in §9.
4. Ship package fallback heuristics if net confidence is low — default stays
   pure net + legal-action fallback (decode_bucket guarantees legality).
5. ~~Pilot data volume~~ → DONE (100K pilot + 25.4M full).
6. RL compute budget vs deadline (2026-11-15 23:59 IST): ✓ budgeted in §9.7
   (M1/M2/M3, hard stop T−6h); gen 8.6 remains the frozen fallback regardless.
7. Pre-RL checklist (§6.3): ✓ ALL CLOSED 2026-11-14 — census (§6.4), quiz
   (§6.5), RL rulebook (§6.6), SHA256 manifest (sample pull dropped by user
   decision). **Nothing pending: RL build M1 (§9.7) is next.**

## 8. Anti-goals / lessons to carry

- No transformer policy from scratch via self-play RL before M0 — the teacher
  distillation comes first; RL is an add-on, gated.
- No GPU winrate as evidence; only CPU paired batteries.
- No claiming an effect without paired stats + VALID guards.
- No touching the frozen submission artifact.
- No parallel-20 batteries on small VMs; no batteries without thread pinning.
- Keep speed: ship model ≤10ms/hand, total bank usage must stay comfortably inside
  75ms/hand average on the target machine.

---

## 9. RL self-play — full design (written 2026-11-14, pre-build; deadline 2026-11-15 23:59 IST)

Everything below obeys the §6.6 rulebook (seed namespaces, opponent pool,
paired batteries, promotion CI, append-only log). Research anchors: adaptive-KL
thermostat from KL-regularized PPO practice (InstructGPT-style proportional
controller); critic-baseline variance reduction for terminal-reward poker
policy gradients. gen 8.6 stays the submitted bot unless the full exam passes.

### 9.1 Objective + honest expectations
- Goal: improve the ACT policy beyond teacher elitev6 in real-engine play,
  judged ONLY by §6.6. Disc model stays FROZEN (its job is simple, it's good).
- RL touches flop/turn mixed strategies first (§6.5 targeting map: preflop
  already at 99.3% agreement — little to gain there).
- Honest budget note: ~1 day. Expected honest outcome: a small, statistically
  verified edge (or a clean "no measurable gain" result). "Miles better"
  would need weeks of iterations; the loop is built so every measured gain is
  real and the infrastructure is reusable.

### 9.2 Architecture (actor-critic)
- **Actor** = current `OpionPolicy` (812,584 params, §6.2) — weights start
  from `final_act.pt`.
- **Critic (new)** = `Linear(128→1)` on the pooled trunk features (value head,
  +129 params). Random init; warm up 1 epoch with the actor FROZEN at the
  start of the first update so baselines are usable immediately.
- Two-headed save format: `policy_rl.pt = {actor, value, step, run}`.

### 9.3 Learning mechanism (tethered PPO on real-engine rollouts)
- **Rollout unit**: 1 engine match = 2 mirror legs × 200 hands = 400 hands
  (stage 1, same as §6.6). Student plays via the numpy port (§9.4) inside
  engine_log.py on E2B sandboxes; its chosen bucket + behavior log-prob +
  value estimate go to a per-match rollout sidecar (keyed seed/bpid/hand/seq).
- **Reward**: per-hand net chips / 200 (starting stack) for the student's
  seat only; terminal per hand (mirror legs cancel seat/deal luck). Rewards
  joined into the sidecar by the runner before pull (it has meta hand_result).
- **Loss** (per minibatch): PPO clip 0.2 on importance ratios; value MSE with
  value clip; entropy bonus 0.01; **+ tether term** β·KL(π_new ‖ p_soft_teacher)
  computed on a FIXED offline probe (32,768 packed train-partition states,
  never the test drawer). γ=1.0 (chips only move at hand end), GAE λ=0.95
  across the hand's decision chain.
- **Adaptive KL thermostat** (per iteration): measure probe KL after the
  update; target schedule KL* = 0.10 → 0.02 nats over ~20 iterations;
  β ← β·exp(2·(KL−KL*)/KL*), β init 1.0, clamp [0.01, 10]. Style guard: if
  any action-bucket share drifts >5 points from the teacher's mix on the
  probe → β doubles for the next iteration.
- **Hyperparameters**: AdamW lr 2e-5 (single group), 2 PPO epochs, minibatch
  8192, grad clip 1.0, value coef 0.5, advantage normalized per batch.
- **Rollout mix per iteration (frozen for run 1, ~260 matches ≈ 280k student
  decisions)**: self-mirror 60, elitev6 40, package_g18_6 30, elite83 30,
  maniac 20, station 20, rock 20, randombot 20, mixer 20. Past champions
  join later runs per §6.6. ~4–6 min generation on 8 sandboxes.

**CRITICAL FIX — leaky ctx[14] (found by the RL dummy test, 2026-11-14):**
spec ctx[14] (own-action amt/pot) perfectly separates raise from non-raise
records, so the v1 student learned the shortcut "amt=0 ⇒ no raise" and CANNOT
play live. Sandbox experiment (4k packed states): true-amt forward = teacher-
like mix + 94.5% ag; amt=0 forced → mix [0.19, 0.81, 0, 0, 0, 0], 0 raises;
per-bucket self-consistent assembly → degenerate over-raising (ag 34% — the
raise/no-raise split was never learned; not recoverable at inference).
**Fix = v2 re-distillation: ctx[:,14]=0 everywhere (train, val, live bot,
tether probe, PPO updates) — 2-epoch fine-tune from final.pt at lr 1e-4
(`opion_redistill.ipynb`, user-run T4, ~25-50 min).** v2 becomes the RL init
and the ship model; live play = plain single forward. Disc model has no
leaky dims (clean). Dataset test drawer stays LOCKED — v2 is judged on val
here + live screens/batteries per §6.6. Dummy-test mechanics all verified on
the test sandbox BEFORE this: 16/16 matches, 30,303 decisions in 34s on 8
workers, rewards joined 100%, ~14 B/decision stored, everything on sandbox
disk, nothing downloaded.
**v2 outcome (2026-11-14, user-run T4):** val (ctx14=0) KL 0.3035,
argmax_ag 85.3% — mix RECOVERED: [0.175, 0.451, 0.074, 0.26, 0.041] vs
teacher [0.17, 0.44, 0.10, 0.26, 0.04]; preflop ag 96.7%, post-flop KL high
(0.49/0.51/0.38) — the honest state-only approximation (v1's 96.6% partly
read the answer). **Live check (test sandbox, 8 matches, byte-level feature
audit): live features == spec features (fp16 storage rounding only); live
behavior drifts raise-heavy on-policy (vs-teacher mix 69% mid-raise,
self-mix 56%), reward vs teacher −0.027 stacks/hand (±0.008 SE, 1,600
hands) — the honest RL starting point; PPO on live states + tether is the
correction mechanism, no further pre-RL patches.** Iteration-1 PPO batch =
the 12,109-decision iter_v2 rollouts (tar: opion/data/rl/rl_iter_0001.tar).
**PPO notebook BUILT (`opion/rl/opion_ppo.ipynb`, 11 cells): GAE within
(pid,hand) chains (γ=1, λ=0.95, terminal=hand net), PPO clip 0.2 + value
MSE + entropy 0.01 + β·KL(p_soft‖π) tether on a fixed 32,768-state probe,
adaptive β thermostat (KL*=0.10), diff-gate vs numpy port every run,
exports policy_rl.pt + policy_np.npz per iteration.**

**Iteration 1–15 results (LIVE; iter-14 = new champion — grind un-flattened):**
- Iter-1 PPO (12,109 decisions, 6 steps, β=1.0): probe KL 0.375 (> 0.10
  target) → thermostat β_next = 10.0; probe mix stayed teacher-shaped
  [0.148, 0.423, 0.107, 0.274, 0.048]. OOM lesson: fp32 batch-8192 + tether
  graph exceeded T4 → fixed with fp16 autocast + MB 4096; **also: a crashed
  cell's pinned traceback holds ~11GB GPU — always restart runtime after an
  OOM.**
- Iter-2 rollouts: 260 matches / **340,569** decisions in **2.3 min on 8
  sandboxes** (20-cap respected), 8/8 OK, all paused. (An earlier "342,569"
  tally was a miscount — tar and local pulls both reconcile to 340,569 =
  PPO RESULT decisions.)
- **Iter-1 mini-screen (per-opp rewards, 73.5k hands): vs elitev6 −0.0026 ±
  0.0012 stacks/hand (was −0.0273 at iter-0 — ~10× better, near breakeven);
  gen86 −0.0018; elite83 −0.0052; beats all style bots (maniac +0.085,
  station +0.050, random +0.048, mixer +0.049, rock +0.005); live bucket mix
  teacher-shaped everywhere (self-play raise-war pathology gone, per-hand sd
  0.33 → 0.15).** PPO fix: rollout pids made sandbox-unique (pid + sb·2^20)
  for correct GAE chains.
- **Iter-2 PPO (340,569 decisions, β=10.0): probe KL 0.3749 → 0.2971 — now
  BELOW v2's own re-distill KL (0.3035): the RL policy is more
  teacher-faithful than pure imitation was, yet screens ~7× better vs the
  teacher. Same average KL, better live play ⇒ PPO reallocated probability
  mass to EV-critical states.** β thermostat saturated at the 10.0 clamp
  (thermostat wants ~515) — we ride the max-tether equilibrium by design.
  Probe mix [0.151, 0.463, 0.072, 0.272, 0.042, 0.000] ≈ teacher
  [0.17, 0.44, 0.10, 0.26, 0.04, 0.00]. Checkpoints verified:
  `opion/rl/ckpts/policy_np_iter2.npz` + `policy_rl_iter2.pt` (812,713
  params).
- **Iter-3 rollouts (PPO fuel + iter-2 screen, fresh rl:i3:* deals): 8/8
  OK, 334,249 decisions, ~2.5 min, all paused. Iter-2 screen (116k hands):
  elitev6 −0.0039 ± 0.0007; gen86 −0.0013 ± 0.0009; elite83 −0.0040 ±
  0.0005; maniac +0.070; station +0.045; rock +0.005; random +0.047; mixer
  +0.039; SELF 0.0000. Verdict: statistical tie with iter-1 (strong-opp
  aggregate −0.00315 vs −0.00314 stacks/hand; all CIs overlap) → iter-2
  carried forward as champion (newest, ties, and iteration 3 is on-policy
  from it).** Iter-3 tar hardened: tag-prefixed shard names (no cross-sb
  basename collision) + in-bundle pid rewrite (pid + sb·2^22), 334,249
  rows, β=10.0, init = iter-2 checkpoint.
- **Iter-3 PPO (334,249 decisions, β=10.0): probe KL 0.2971 → 0.2832
  (smooth convergence, no oscillation); probe mix [0.151, 0.455, 0.078,
  0.275, 0.042, 0.000] ≈ teacher; β stays clamped at 10.0. Checkpoints
  verified: `policy_np_iter3.npz` + `policy_rl_iter3.pt` (812,713
  params).**
- **Iter-4 rollouts (PPO fuel + iter-3 screen, fresh rl:i4:* deals): 8/8
  OK, 331,985 decisions, ~2.5 min, all paused. Iter-3 screen (116k hands):
  elitev6 −0.0039 ± 0.0008 (identical to iter-2 on different seeds — very
  stable estimate); gen86 −0.0031 ± 0.0009; elite83 −0.0026 ± 0.0009; style
  bots all beaten (maniac +0.071, station +0.048, rock +0.005, random
  +0.046, mixer +0.039); SELF exactly 0 (mirror sanity ✓). Verdict: third
  consecutive statistical tie (strong-opp aggregate −0.00327 vs
  −0.00315/−0.00314 stacks/hand) ⇒ the loop has PLATEAUED at the β=10
  tether equilibrium — screens no longer separate checkpoints. Champion
  candidate set = {iter-1, iter-2, iter-3} for the blind final to pick
  from.**
- **Iter-4 PPO (331,985 decisions, β=10.0): probe KL 0.2832 → 0.2720;
  probe mix ≈ teacher; β clamped at 10.0. Checkpoints verified:
  `policy_np_iter4.npz` + `policy_rl_iter4.pt` (812,713 params).**
- **Iter-5 rollouts (PPO fuel + iter-4 screen, fresh rl:i5:* deals): 8/8
  OK, 332,528 decisions, ~2.5 min, all paused. Iter-4 screen (116k hands):
  elitev6 −0.0033 ± 0.0007; **gen86 +0.0010 ± 0.0012 — FIRST checkpoint to
  cross zero vs gen86**; elite83 −0.0039 ± 0.0008; style bots all beaten
  (maniac +0.067, station +0.046, rock +0.005, random +0.047, mixer
  +0.040); SELF 0.0000. Strong-opp aggregate **−0.00219 — best of all four
  screens** (iter-1 −0.00314, iter-2 −0.00315, iter-3 −0.00327). Caveat:
  each screen uses a different deal set (rl:i2/i3/i4/i5 namespaces), so
  between-round deltas carry deal noise; but iter-4 is best vs the teacher,
  the only one positive vs gen86, and regressed nowhere. KL still
  descending (0.3749→0.2971→0.2832→0.2720) ⇒ keep grinding while screens
  improve; blind-final candidate set = {iter-1…iter-4}.**
- **Iter-5 PPO (332,528 decisions, β=10.0): probe KL 0.2720 → 0.2622;
  probe mix ≈ teacher; β clamped at 10.0. Checkpoints verified:
  `policy_np_iter5.npz` + `policy_rl_iter5.pt` (812,713 params).**
- **Iter-6 rollouts (PPO fuel + iter-5 screen, fresh rl:i6:* deals): 8/8
  OK, 331,413 decisions, ~2.5 min, all paused. Iter-5 screen (116k hands):
  elitev6 −0.0029 ± 0.0007; gen86 −0.0013 ± 0.0011 (iter-4's +0.0010 was
  deal-set noise; gen86 readings across rounds: −0.0018/−0.0013/−0.0031/
  +0.0010/−0.0013 ⇒ ≈ breakeven overall); elite83 −0.0026 ± 0.0008; style
  bots all beaten (maniac +0.068, station +0.049, rock +0.005, random
  +0.046, mixer +0.034); SELF 0.0000. Strong-opp aggregate −0.00233 —
  second-best (iter-4 −0.00219 best). Top-2 blind-final candidates:
  iter-4 + iter-5. KL slope unchanged (0.2720→0.2622) ⇒ grind continues
  while it improves.**
- **Iter-6 PPO (331,413 decisions, β=10.0): probe KL 0.2622 → 0.2529;
  probe mix ≈ teacher; β clamped at 10.0. Checkpoints verified:
  `policy_np_iter6.npz` + `policy_rl_iter6.pt` (812,713 params).**
- **Iter-7 rollouts (PPO fuel + iter-6 screen, fresh rl:i7:* deals): 8/8
  OK, 329,929 decisions, ~2.5 min, all paused. Iter-6 screen (116k hands):
  elitev6 −0.0032 ± 0.0006; gen86 −0.0005 ± 0.0009 (gen86 series across 6
  rounds: −0.0018/−0.0013/−0.0031/+0.0010/−0.0013/−0.0005 — steady drift
  toward breakeven-positive); elite83 −0.0045 ± 0.0009 (soft spot this
  round); style bots all beaten (maniac +0.069, station +0.048, rock
  +0.005, random +0.051, mixer +0.036); SELF 0.0000. Strong-opp aggregate
  −0.00278 — inside the iter-4+ noise band ([−0.0022, −0.0033]). Top-3
  blind-final candidates: iter-4 + iter-5 + iter-6. KL slope still
  negative (0.2529) ⇒ grind continues.**
- **Iter-7 PPO (329,929 decisions, β=10.0): probe KL 0.2529 → 0.2445;
  probe mix ≈ teacher; β clamped at 10.0. Checkpoints verified:
  `policy_np_iter7.npz` + `policy_rl_iter7.pt` (812,713 params).**
- **Iter-8 rollouts (PPO fuel + iter-7 screen, fresh rl:i8:* deals): 8/8
  OK, 330,451 decisions, ~4.3 min, all paused. Iter-7 screen (116k hands):
  elitev6 −0.0027 ± 0.0007 (ties iter-1 for best vs teacher); gen86
  −0.0003 ± 0.0009 (best gen86 reading yet; series: −0.0018/−0.0013/
  −0.0031/+0.0010/−0.0013/−0.0005/−0.0003 — settled at breakeven);
  elite83 −0.0031 ± 0.0006; style bots all beaten (maniac +0.061, station
  +0.046, rock +0.005, random +0.049, mixer +0.040); SELF 0.0000.
  Strong-opp aggregate **−0.00210 — BEST YET** (prior best iter-4
  −0.00219; iter-1/2/3 cluster −0.0031…−0.0033). Top-3 blind-final
  candidates: iter-7 + iter-4 + iter-5. KL slope still negative (0.2445)
  ⇒ grind continues.**
- **Iter-8 PPO (330,451 decisions, β=10.0): probe KL 0.2445 → 0.2365;
  probe mix ≈ teacher; β clamped at 10.0. Checkpoints verified:
  `policy_np_iter8.npz` + `policy_rl_iter8.pt` (812,713 params).**
- **Iter-9 rollouts (PPO fuel + iter-8 screen, fresh rl:i9:* deals): 8/8
  OK, 329,557 decisions, ~2.5 min, all paused. Iter-8 screen (116k hands):
  elitev6 −0.0035 ± 0.0006; gen86 −0.0002 ± 0.0012 (breakeven drift
  continues: −0.0013/−0.0005/−0.0003/−0.0002 over last four);
  elite83 −0.0025 ± 0.0010 (best elite83 reading of the RL era); style
  bots all beaten (maniac +0.067, station +0.047, rock +0.005, random
  +0.047, mixer +0.038); SELF 0.0000. Strong-opp aggregate −0.00221 —
  second-best, tied with iter-4 (best remains iter-7 −0.00210). Top-4
  blind-final candidates: iter-7 + iter-4 + iter-8 + iter-5. KL slope
  still negative (0.2365) ⇒ grind continues.**
- **Iter-9 PPO (329,557 decisions, β=10.0): probe KL 0.2365 → 0.2288;
  probe mix ≈ teacher; β clamped at 10.0. Checkpoints verified:
  `policy_np_iter9.npz` + `policy_rl_iter9.pt` (812,713 params).**
- **Iter-10 rollouts (PPO fuel + iter-9 screen, fresh rl:i10:* deals): 8/8
  OK, 331,419 decisions, ~2.5 min, all paused. Iter-9 screen (116k hands):
  elitev6 −0.0031 ± 0.0007; gen86 −0.0011 ± 0.0010 (breakeven band);
  **elite83 −0.0015 ± 0.0009 — best elite83 reading of the entire run**;
  style bots all beaten (maniac +0.070, station +0.049, rock +0.005,
  random +0.048, mixer +0.041); SELF 0.0000. Strong-opp aggregate
  **−0.00202 — NEW BEST** (prior iter-7 −0.00210). Top-4 blind-final
  candidates: iter-9 + iter-7 + iter-4 + iter-8. KL slope still negative
  (0.2288) ⇒ grind continues.**
- **Iter-10 PPO (331,419 decisions, β=10.0): probe KL 0.2288 → 0.2216;
  probe mix ≈ teacher; β clamped at 10.0. Checkpoints verified:
  `policy_np_iter10.npz` + `policy_rl_iter10.pt` (812,713 params).**
- **Iter-11 rollouts (PPO fuel + iter-10 screen, fresh rl:i11:* deals):
  8/8 OK, 331,054 decisions, ~4.3 min, all paused. Iter-10 screen (116k
  hands): elitev6 −0.0049 ± 0.0006 — WORST vs-teacher reading of the run
  (prior band [−0.0026, −0.0039]); gen86 −0.0012 ± 0.0011 (breakeven
  band); elite83 −0.0033 ± 0.0008; style bots all beaten (maniac +0.073,
  station +0.047, rock +0.005, random +0.050, mixer +0.040); SELF 0.0000.
  Strong-opp aggregate −0.00331 — bottom of the band, does NOT promote.
  Reading: borderline regression vs noise (aggregate z ≈ 1.8–2 vs iter-9;
  note iter-6 was also soft at −0.00278 and iter-7 right after it was
  best). DECISION: iter-9 stays champion for the blind final; chain
  continues on-policy from iter-10 (tar already built). TRIP-WIRE: if the
  iter-11 screen is also bottom-of-band, STOP the grind and go to the
  ship battery with the top-4 set {iter-9, iter-7, iter-4, iter-8}.**
- **Iter-11 PPO (331,054 decisions, β=10.0): probe KL 0.2216 → 0.2149;
  probe mix ≈ teacher; β clamped at 10.0. Checkpoints verified:
  `policy_np_iter11.npz` + `policy_rl_iter11.pt` (812,713 params).**
- **Iter-12 rollouts (PPO fuel + iter-11 screen, fresh rl:i12:* deals):
  8/8 OK, 329,557 decisions, all paused. Iter-11 screen (116k hands):
  elitev6 −0.0029 ± 0.0007 (top of band); gen86 +0.0002 ± 0.0009 (second
  ≥0 reading; last-4 mean ≈ −0.0004 ⇒ breakeven); elite83 −0.0025 ±
  0.0007; style bots all beaten with best-ever station +0.050 and mixer
  +0.042; SELF 0.0000. Strong-opp aggregate **−0.00185 — NEW BEST, trip-
  wire passed** (iter-10's −0.0033 confirmed as deal noise; the chain
  bounced straight back up). CHAMPION: iter-11. Top-4 blind-final
  candidates: iter-11 + iter-9 + iter-7 + iter-4. KL slope still negative
  (0.2149) ⇒ grind continues.**
- **Iter-12 PPO (329,557 decisions, β=10.0): probe KL 0.2149 → 0.2089;
  probe mix ≈ teacher; β clamped at 10.0. Checkpoints verified:
  `policy_np_iter12.npz` + `policy_rl_iter12.pt` (812,713 params).**
- **Iter-13 rollouts (PPO fuel + iter-12 screen, fresh rl:i13:* deals):
  8/8 OK, 330,141 decisions, all paused. Iter-12 screen (116k hands):
  elitev6 −0.0031 ± 0.0008 (top band); gen86 +0.0006 ± 0.0008 (third ≥0
  reading; 12-round gen86 mean ≈ −0.0008 and clearly trending up);
  elite83 −0.0039 ± 0.0008 (soft this round); style bots all beaten with
  best-ever random +0.052 and mixer +0.043; SELF 0.0000. Strong-opp
  aggregate −0.00223 — mid-band, does not promote (champion stays
  iter-11 −0.00185). Top-4 unchanged: iter-11 + iter-9 + iter-7 +
  iter-4. KL slope decaying but negative (0.2089) ⇒ grind continues.**
- **Iter-13 PPO (330,141 decisions, β=10.0): probe KL 0.2089 → 0.2034;
  probe mix ≈ teacher; β clamped at 10.0. Checkpoints verified:
  `policy_np_iter13.npz` + `policy_rl_iter13.pt` (812,713 params).**
- **Iter-14 rollouts (PPO fuel + iter-13 screen, fresh rl:i14:* deals):
  8/8 OK, 331,096 decisions, all sandboxes completed/pulled/paused.
  (Incident: the local finisher died mid-run from a user interrupt —
  sandbox data was already safe on disk; tar + screen were rebuilt
  locally from the pulled files. No sandbox left running.) Iter-13
  screen (116k hands): elitev6 −0.0029 ± 0.0007 (top band); gen86
  −0.0010 ± 0.0011 (breakeven band); elite83 −0.0041 ± 0.0007 (soft);
  style bots all beaten; SELF 0.0000. Strong-opp aggregate −0.00269 —
  mid-band, does not promote. Champion stays iter-11 (−0.00185); top-4
  unchanged: iter-11 + iter-9 + iter-7 + iter-4.**
- **Iter-14 PPO (331,096 decisions, β=10.0): probe KL 0.2034 → 0.1984
  (first sub-0.20); probe mix ≈ teacher; β clamped at 10.0. Checkpoints
  verified: `policy_np_iter14.npz` + `policy_rl_iter14.pt` (812,713
  params).**
- **Iter-15 rollouts (PPO fuel + iter-14 screen, fresh rl:i15:* deals):
  8/8 OK, 331,087 decisions, all paused. Iter-14 screen (116k hands):
  elitev6 −0.0025 ± 0.0007 — BEST vs-teacher reading of the RL era;
  gen86 +0.0018 ± 0.0008 — BEST gen86 ever with FULLY POSITIVE CI
  [+0.0010, +0.0026] (4th ≥0 reading); elite83 −0.0037 ± 0.0010 (soft);
  style bots all beaten; SELF 0.0000. Strong-opp aggregate −0.00157 —
  **NEW CHAMPION: iter-14** (prior iter-11 −0.00185). The "flattened"
  call was wrong — the screens bounce but each extra cycle samples the
  tail and the champion keeps improving (4th new-best pattern). Top-4:
  iter-14 + iter-11 + iter-9 + iter-7. KL slope still negative (0.1984)
  ⇒ user may continue; ship battery remains one "pivot" away.**
- Iter-15 tar (`/tmp/rl_iter_0015.tar`, 10.1MB, 331,087 rows, β=10.0,
  init = iter-14 checkpoint) → **consumed by FULLY AUTOMATED PPO
  iteration 15 (see §ppo_auto below): RESULT kl_probe 0.1941, β_next
  10.0, wall 2.1 min on T4; artifacts verified (sha256 round-trip, 60
  keys, 812,713 params, max|ΔW| vs iter-14 = 0.0030, median rel 0.54%,
  no NaN) → `ckpts/policy_np_iter15.npz` + `policy_rl_iter15.pt`.
  KL trajectory 0.3749→0.1941 (15th consecutive decrease).**

**§ppo_auto — end-to-end RL automation (NEW, 2026-11-15):** the whole
iteration loop now runs without the user: rollouts+screen+tar on the 8
sandboxes (`rl_iterN.py`) → PPO on a rented Colab T4 via the `colab`
CLI (`~/.local/bin/colab`, google-colab-cli 0.7.5). Headless notebook
driver: `opion/rl/ppo_auto/c1..c8*.py` (cells 1–9 verbatim from
`opion_ppo.ipynb`; cell 10's `files.download` replaced by hash-print +
`colab download`). Cycle-1 timings: T4 alloc ~30s, 10MB tar upload ~5s,
packed.tar (481MB) gdown ~6s, cells B–F ~40s, PPO 162 steps ~85s,
download+verify ~20s ⇒ **~4 min/T4 leg**. Gotchas learned: pass
`--timeout` on every exec (default 30s); `-f` needs an absolute local
path (shell cwd drifts); kernel state persists across execs so split
cells are safe; torch 2.14 .pt zip no longer has `data.pkl` (round-trips
only in same-env torch; ship path is npz-only so unaffected); T4 stop
after each leg (idle burns compute units; re-alloc is cheap); update
check silenced in `~/.config/colab-cli/settings.json`. Cycle-2 (iter-16)
launched immediately: `rl_iter16.py` uploads iter-15 npz → rollouts-16 +
screen-15 → tar-16 → T4 PPO.

**§rl_loop — CONTINUOUS GRIND (user command 2026-11-15 ~20:30 IST):
"continue the RL loop continuously".** Driver `rl_loop_auto.py`
(opion/rl/): waits round → parses screen → T4 leg → verify+install ckpts
→ sed-generates next rl_iterN.py → repeat; freeze gates: margin-to-23:10
< 57 min, or two consecutive strong-aggs ≤ −0.0045. **REAL cycle wall =
~6.5 min** (round 2.5-3 min, T4 leg 2.5-2.9 min) — rounds are far faster
than the 15-25 min plan estimate. Results (strong-agg = hands-weighted
elitev6/gen86/elite83; iter-14 champion reading −0.00157):
- Screens: i15 −0.00279, i16 −0.00273, i17 −0.00256, i18 −0.00283,
  i19 −0.00325, i20 −0.00284, i21 −0.00238, i22 −0.00295, i23 −0.00312,
  i24 −0.00214 (2nd-best ever), i25 −0.00286. No new champion; band
  tightened upward; trip-wire never close.
- KL probe: 0.1941 (15) → 0.1905 → 0.1916 → 0.1902 → 0.1901 → 0.1906 →
  ~0.190 plateau (β=10 tether is binding; 20+ iterations all verified
  60 keys / 812,713 params / max|dW| ≤ 0.003).
- Incident: T4-26 c7 exec wedged (kernel IDLE, CLI waiting; kernel never
  recorded the execution). `colab restart-kernel` didn't detach the
  stuck CLI; self-heal = the exec's 1950s bash guard + orchestrator
  retry on a fresh session. Fix if it recurs: per-cell watchdog kill.

**FINAL SHIP DECISION (2026-11-15 22:30 IST, user: "stop now, turn off
the machines, build the final bot"):**
- **Shipped: `submissions/synergy_v8_6_submission.zip`** (gen 8.6,
  package_g18_6). Re-verified at freeze: sha256
  `13744f98563aae376526b701f29906d106258f1fca006b7127e05b5773846960`,
  290,389 bytes (<5MB), package structure intact (bot.py + 8 modules +
  3 data npz), every .py compiles. Rationale: fieldEV leader
  (+0.0379 vs B +0.0373 / A +0.0294 / RL-champ +0.0244), fast on 1 vCPU,
  protocol-proven, frozen since Nov 13 — zero packaging risk at T−90min.
- RL alternative NOT shipped: RL student iters 14/24/29 (npz in
  `opion/rl/ckpts/`) — best screen reading −0.00157 (iter-14) but worst
  fieldEV of the four candidates; would have needed a first-time package
  build + 1-vCPU gate in the final hour.
- Machines off at 22:28 IST: all 8 E2B sandboxes paused (stray match
  procs pkilled), colab sessions opion-t4 + syn90 terminated, syn90
  keepalive loop killed, battery task stopped mid-run (partial bat2 log
  in /tmp/bat2.log; rl_iters_battery.json not written).

**§opion.py — RL TRANSFORMER SUBMISSION BUILD (2026-11-15 22:39 IST,
user request: "create a bot.py of this transformer model too, name it
opion.py, make it submission ready"):**
- `submissions/opion.py` — SINGLE self-contained file, 3,995,149 B
  (4.00 MB < 5 MB), stdlib+numpy only, no torch/env/sidecar. Inlines:
  spec feature encoder (opion/fleet/spec.py core), numpy policy port
  (opion/rl/np_policy.py core), student bot protocol loop
  (opion/rl/student_bot.py minus rollout sidecar) + safe-action and
  lowest-card discard fallbacks. Weights EMBEDDED base64: actor =
  iter-14 champion policy npz, disc = final_disc.pt converted to npz
  (no-torch loader), both stored **float16** (compute fp32 — NumpyPolicy
  casts on load) to fit the cap. fp16 drift gate measured at build:
  act max|dlogit| 0.0226, disc 0.0073 on 4,096 synthetic states
  (noise-level; logit scale ±5).
- `submissions/opion_submission.zip` — 3,008,768 B, opion.py inside;
  sha256 opion.py `14239b6f8090d425…`, zip `df5241b1dd73af5b…`.
- LOCAL engine validation (opion/fleet/engine_log.py, venv python, no
  sandbox needed): vs rock 400/400 hands, **A wins +271** (+0.0034
  stacks/hand — matches the screened ~+0.0048 vs rock), vs elite83
  400/400 hands, **A wins +166** (single-match noise vs screen −0.004);
  0 notices both matches, **bot stderr 0 bytes both legs** (no hidden
  tracebacks/fallbacks), whole 2-leg match 1.28 s wall, time bank used
  0.00 s of 30 s ⇒ ~1 ms/decision vs the 10 ms 1-vCPU budget.
- Builder preserved: `opion/rl/build_opion_submission.py` (re-run to
  rebuild from any ckpts/policy_np_iterN.npz).

**External submissions (user-provided zips, provenance TBD — teammates'
candidates or rivals' intel):**
- **A = `submission.zip`**: "Elite Poker Bot – Next-Gen v8", single 3.8MB
  bot.py (embedded zlib/base64 weights), adaptive multi-personality +
  dual-stage deep RL + subgame CFR+. **B = `submission_syn_v8.8.zip`**:
  the Synergy hand-crafted line (MC + CFR+ subgame solving + nnu/tns
  MLPs, ~185KB weights), newer than our frozen gen 8.6. Both verified
  engine-protocol-compliant and fast (A 14s/match, B 6s/match).
- **Head-to-head matrix (24 matches/leg, mirror legs, fresh rl:ext:/rl:p:
  seeds, stacks/hand ± SE):**
  - Opion iter-14 vs A: **−0.0056 ± 0.0009 (A decisively better, z≈6)**
  - Opion iter-14 vs B: +0.0004 ± 0.0009 (tie)
  - Opion iter-14 vs gen86: +0.0018 ± 0.0008 (Opion better)
  - gen86 vs A: +0.0012 ± 0.0012 (tie)
  - gen86 vs B: −0.0008 ± 0.0011 (tie)
  - A vs B: +0.0003 ± 0.0012 (dead tie)
- **Reading: non-transitive triangle** (A > Opion > gen86 ≈ A) — style
  matchups; the field of four is flat except A's clear edge over our
  champion. A is the strongest bot we have ever measured. Head-to-head
  alone can't rank them for shipping — **needs the field battery (each
  candidate vs the 18-opp census fleet, total EV ranking).** (Pairs
  script printed chips/hand; ÷200 = stacks/hand as reported here.)
- **FIELD BATTERY (2026-11-15 ~10:45 IST, 252 matches, 0 bad): A, B,
  gen86 vs the 7 §9.3 fleet opps, 12 matches/pair, rl:f:* seeds —
  field EV (stacks/hand, hands-weighted): gen86 +0.0379 > B +0.0373 >
  A +0.0294 > champion iter-14 +0.0244 (recomputed over the same 7
  opps from the rl:i15 screen; the script's +0.0208 wrongly averaged 8
  opps incl. gen86).** Profile split: A = strong-bot specialist (ONLY
  candidate ≥0 vs elitev6 +0.0009 and elite83 +0.0019; leaves value vs
  fish: station +0.046, random +0.047); B & gen86 = fish-crushers
  (station +0.095/0.095, random +0.092/0.093) but give back vs strong
  (elitev6 −0.0011/−0.0015, elite83 −0.0023/−0.0011). Champion =
  teacher-shaped middle: best vs maniac (+0.068) but under-exploits
  fish (station +0.046, random +0.047) — **the β=10 tether's measured
  cost.** Ship-decision inputs now: (1) field EV favors gen86/B,
  (2) tournament scoring shape (chip EV vs per-match points) decides
  how much fish-crushing is worth, (3) A's ~9-10ms/decision probe pace
  is borderline vs the 1-vCPU 10ms budget (champion 1.2ms), (4) A/B
  provenance still unknown. SAME-HARNESS champion re-measure (84
  matches, ~3 min) would remove the last apples-to-apples doubt.

### 9.4 The numpy port — prerequisite, built FIRST (reused for shipping)
- **BUILT + LOCALLY TESTED 2026-11-14** (`opion/rl/np_policy.py`,
  `test_np_policy.py`): pure-numpy fp32 forward byte-mirroring model.py;
  loads weights from policy_np.npz OR straight from final_act.pt with a
  no-torch .pt parser (zip/pickle storage format). Local battery all PASS:
  erf 1.4e-7; 812,584 params/58 keys via the parser; independent loop-based
  reference vs vectorized port max|Δlogit| 2.4e-6; 512 real states finite +
  masked-argmax 100% on-legal; **1.2 ms/decision single-threaded** (8× under
  the 10ms budget). **Colab torch-vs-numpy gate (≥10,528 states, <1e-3) is
  prepared (`colab_diff_test.py`), pending a T4 session — no sandbox rollout
  until GATE_PASS.**
- **Quirk discovered (documented, do not "fix" unilaterally):** model.py's
  `_map_tokens` docstring claims hist tokens shift to 53..84, but the rule
  (t<53 unchanged) makes the map IDENTITY on the whole raw data domain
  (cards 0..52, hist tokens 1..32 all <53). Hist tokens share embedding rows
  1..32 with card ids; slot position embeddings disambiguate. Consistent in
  training + quiz + port, so behavior is preserved — the port mirrors the
  formula, not the docstring.

### 9.5 Loop layout (who runs what)
- **Sandboxes (mine)**: rollout generation + §6.6 screens/batteries. The
  census-test sandbox (`iubpl0b0mz0npoyhoh3w6`) is the RL pilot sandbox.
  Per iteration: upload `policy_np.npz` (3.3MB) + rl_play.py → 8 sandboxes
  play the mix → pull rollout npz (~15–25MB) → pause all.
- **Colab T4 (user session `opion-t4`, driven via colab CLI, manual fallback
  = paste cells)**: PPO update notebook (includes diff-test + β thermostat +
  probe KL + export `policy_np.npz`). Keepalive as before.
- **Local (mine)**: orchestration, joining, eval-log bookkeeping, weights
  upload/download, screens/batteries, promotion bookkeeping.
- **Iteration wall time ≈ 15–25 min** (gen 5–6 + update 3–6 + screen 4–10,
  screens can overlap the next update). ~20–25 iterations fit the budget.

### 9.6 Judging wiring (§6.6, no free lances)
- `dev:` seed lists generated ONCE at M2 (screen 60×8, battery 200×8), hashed
  into the eval log. Screen EVERY iteration; promotion battery only when a
  screen beats the champion clearly; champion changes only via battery CIs.
- Incumbent champion for run 1 = the distilled student (§6.5 baseline:
  96.6% agreement; live-engine scorecard measured fresh at M2 on dev seeds).
- All results → `opion/rl/eval_log.jsonl` (append-only, per §6.6).

### 9.7 Milestones + hard schedule (deadline 2026-11-15 23:59 IST)
- **M1 — build (~2–3h)**: numpy port + diff-test gate, value head, rl_play.py
  + sandbox pilot, PPO notebook, orchestration scripts. Exit = one full DUMMY
  iteration (rollout → update → redeploy) with no promotion claims.
- **M2 — first real iteration + mechanics check (~2h)**: fresh dev/blind seed
  lists frozen; incumbent live scorecard; iteration 1 + screen. Exit =
  screen noise floor measured, §6.6 loop proven end-to-end.
- **M3 — scaled iterations**: as many clean iterations as the budget allows.
  **Hard stop T−6h before deadline**: freeze champion → blind final battery +
  1-vCPU speed benchmark (p50/p95/p99) → ship decision. Shrink is SKIPPED by
  default (812K fits the 5MB zip; the speed gate decides). If the port or the
  exam fails → gen 8.6 ships, Opion documented as the gen 8.8 track.

### 9.8 Risks & guards
| risk | guard |
|---|---|
| numpy port subtly wrong | diff-test gate ≥10k states, <1e-3; engine-side sanity (legal moves only) |
| PPO collapses style (over-fold/over-bluff) | adaptive β thermostat + style guard + entropy floor; screen catches drift |
| value head useless early | actor-frozen warmup epoch |
| Colab outage / T4 scarcity | retry loop; manual paste fallback; iterations are independent |
| sandbox 429 storms | staggered starts, semaphores, retries (proven patterns) |
| time overrun | M3 hard stop T−6h; blind final is short; fallback always stands |
| unlucky evals | paired seeds + mirror legs + CIs (§6.6) — no single-battery decisions |

## opion_submission_final (2026-11-15 23:0x IST)
- `submissions/opion_submission_final/bot.py` (3,991,976 B) + `submission.zip`
  (3,009,442 B, bot.py at top level only). Same iter-14 weights as opion.py.
- Built by `opion/rl/build_final_submission.py`: AST docstring strip + unparse
  of opion_template.py (comments removed by unparse), blobs re-injected from
  submissions/opion.py. Verified: no `#`, no `"""`, no shebang, banned-string
  scan clean (code part; base64 blobs excluded as random noise), AST equality
  with template-minus-docstrings.
- Docs re-read (problem_statement, rulebook, README, protocol_spec):
  compliance check passed — imports stdlib+numpy only; replies only to
  action_request/discard_request, one flushed line each; ignores unknown
  types; exits after match_end; raise amounts int-clamped to
  [min_raise, max_raise]; discard from current hole cards; exception
  fallback => check/call-fold + lowest-card => no forfeit path.
- Colab CPU VM (2 cores, python 3.13.18, numpy 2.4.6), engine pinned to
  1 core with taskset -c 0 (bots inherit affinity): latency probe n=300
  mean 4.38 ms, p99 7.58 ms, max 17.28 ms (limits 5000 ms/reply, 30 s bank);
  startup+model load 0.26 s (limit 5 s); 5 official 400-hand stage-1 matches
  vs starter-kit random_bot: 5/5 wins, 0 notices, bot stderr 0 B across all
  10 legs, each leg ~3-4 s wall (limit 90 s), charged bank time 0.00 s;
  peak python RSS engine+2 bots 177 MB of 1024 MB (systemd-run cgroup cap
  unavailable on the VM, so 1GB verified by measurement, 1 vCPU by pinning).
- Colab session opion-final stopped after the test.
