# Gene Homogeneous MRTA V1.8: Consequence-Aware Direct Assignment

## Motivation

V1.7 Direct Assignment reached a 64-world oracle probe retention of about
96.52%, but its 20-world 97M held-out retention was about 94.98%.

Event-level failure traces showed that the main failure mode was not learned
STOP/WAIT. In the key failures, STOP occurred only after no feasible pair
remained. The earlier assignment had already destroyed a valuable future
task chain or consumed a task that was scarce for another robot.

V1.8 tests the following hypothesis:

> The main V1.7 bottleneck is missing assignment-consequence information in
> the pair observation, not decoder capacity.

## Controlled ablation

Everything below is unchanged from V1.7-T:

- same fixed environment;
- same 256 train / 64 probe cached MILP T* dataset;
- same autoregressive direct pair decoder;
- same learned STOP/WAIT;
- same hidden dimension (8);
- no external Greedy/Hungarian matcher;
- no MILP action labels or imitation;
- same population/archive/HOF sizes;
- same first-1000-generation oracle batch schedule;
- same first-1000-generation mutation-sigma trajectory.

Only the pair observation changes.

## Observation

V1.7 used 8 features:

1. Euclidean distance
2. obstacle-aware path distance
3. service time
4. priority
5. deadline remaining
6. current battery
7. workload
8. competition

V1.8 appends four deterministic consequence features:

9. **Self future reachability**

   Fraction of currently remaining tasks that the same robot could still
   complete after finishing the candidate task, subject to horizon and
   remaining battery.

10. **Self best future time utility**

    Best one-step future time utility available to the same robot after the
    candidate task:

        max_k (1 - F_next(k) / H)

    over future feasible tasks k.

11. **Other-robot opportunity cost**

    For every other robot, including robots currently busy, estimate whether
    the candidate task is a valuable and scarce future option. The feature
    uses the task's future time utility divided by that robot's number of
    feasible remaining options, then takes the maximum over other robots.

    This is a state feature, not an assignment optimization.

12. **Residual battery**

    Battery fraction remaining immediately after the candidate travel.

Therefore:

    observation dimension: 8 -> 12
    hidden dimension:      8 -> 8
    parameters:          116 -> 148

## Exact V1.7 -> V1.8 lift

The V1.7 8xH pair encoder is copied into the first eight rows of the V1.8
12xH encoder. The four new rows are initialized to zero. All pair-decoder and
STOP/WAIT parameters are copied exactly.

Therefore the lifted V1.8 policy initially ignores the new features and
reproduces V1.7 exactly. Evolution may then discover useful nonzero weights
for the new consequence features.

## Smoke

    git pull
    bash tools/run_gene_mrta_v18_mac.sh smoke

Expected sanity behavior:

- all tests pass;
- V1.7 -> V1.8 lift parity tests pass;
- Gen 0 probe best should be at least the lifted V1.7 baseline (about 96.52%)
  when using the finalized V1.7 1000-generation run.

## Long run

    bash tools/run_gene_mrta_v18_mac.sh long

Configuration:

- 1000 generations
- population 256
- archive 32
- HOF 64
- oracle batch: 16 -> 32 -> 64
- probe every 10 generations
- checkpoint every 25 generations
- sigma: 0.25 -> 0.1325

The sigma endpoint reproduces approximately the mutation scale reached by the
V1.7 2000-generation schedule at its 1000-generation stopping point.

## Resume

    bash tools/run_gene_mrta_v18_mac.sh resume <checkpoint.json>

## Held-out comparison

After the run completes:

    bash tools/run_gene_mrta_v18_mac.sh heldout <v18_run_dir>

This evaluates the same 97M held-out worlds and compares:

- V1.8 consequence-aware Direct Assignment
- V1.7 8D Direct Assignment
- V1.6-T-O linear Bid + Greedy
- Hungarian path-time
- proven MILP global T*

The primary question is whether V1.8 reduces catastrophic held-out failures
such as seeds 97000012 and 97000019 while improving mean retention.
