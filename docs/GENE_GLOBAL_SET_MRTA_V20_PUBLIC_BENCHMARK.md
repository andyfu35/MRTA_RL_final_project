# Gene Global-Set MRTA V2.0 — Public Benchmark Transfer

Status: IMPLEMENTED, READY TO RUN  
Branch: `experiment/gene-global-set-mrta-v2`

## Goal

Evaluate the already-trained frozen V2.0 specialists on public instances and
published baselines without retraining or Gene-Bank feedback.

Frozen checkpoint:

`runs/gene_mrta_v20/longrun_1024g_200r_seed200/checkpoint.json`

Frozen Genes:

- Total-Time: `5b448ba2073a90eba8d5`
- On-Time: `ba1efa666ac266786b69`

The benchmark deliberately uses two public tracks because the two specialists
optimize different objectives.

---

## Track A — Total-Time Gene on MinMax mTSP / mTSPLib

Primary external objective:

[
\min \max_r L_r
]

Every salesman starts at the common depot, every city is visited exactly once,
and every route returns to the depot.

This aligns with the public MinMax mTSP benchmark much more closely than the
native V2.0 training worlds. The evaluator therefore scores the frozen Gene
using the public closed-depot makespan, including the final return to depot.

Public instances:

- eil51
- berlin52
- eil76
- rat99

Robot counts:

- m = 2
- m = 5

For each case, both frozen Genes are run, but the Total-Time specialist is the
primary model and the On-Time specialist is only a cross-specialist control.

Task features absent from mTSP are neutralized:

- priority = constant;
- deadline = constant neutral normalized feature;
- service time = 0.

No Gene parameter is changed.

### Public reference semantics

The comparison table follows the ScheduleNet mTSPLib benchmark table:

- CPLEX;
- LKH3;
- OR-Tools;
- ScheduleNet;
- SOM;
- ACO;
- EA.

Only CPLEX values explicitly marked with `*` are treated as proven optimal.

Current proven-optimal cases in the selected subset:

- eil51, m=2: 222.73
- eil76, m=2: 280.85

For those cases:

[
Gap_{OPT} =
100\frac{M_{Gene}-M^*}{M^*}
]

In the final published ScheduleNet/AAMAS benchmark table, non-starred CPLEX
entries are reported as known-best upper bounds, not proven optima. LKH3 is
also reported as a feasible reference.

Therefore non-optimal cases report:

- comparison to every published feasible method;
- gap to the best published feasible result among CPLEX/LKH3, OR-Tools,
  ScheduleNet, SOM, ACO and EA;
- no OPT claim unless the reference is explicitly marked with `*`.

This distinction is mandatory for any paper claim.

### Domain-shift caveat

The frozen V2.0 policy was trained without a return-to-depot requirement.
The public evaluator adds the mandatory final depot return only when scoring
the constructed routes. Therefore this is intentionally a hard zero-shot
transfer test; the Gene was never trained to anticipate the return leg.

---

## Track B — On-Time Gene on TWPC-MRTA

Source:

L. Zhang, J. Zhao, E. Lamon, Y. Wang and X. Hong,
"Energy Efficient Multi-Robot Task Allocation Constrained by Time Window and
Precedence," IEEE Transactions on Automation Science and Engineering,
DOI 10.1109/TASE.2023.3312214.

Public repository:

`zhanglixuan0720/TWPC-MRTA`

Selected public subsets:

- Uniform Distribution RL5:
  - 5 robots
  - 18 tasks
  - map m0
  - samples 0..9
  - public references: MIP, BMRTA/Batch, TePSSI/AuctionO
- Uniform Distribution RL10:
  - 10 robots
  - 36 tasks
  - map m0
  - samples 0..9
  - public references: BMRTA/Batch, TePSSI/AuctionO
  - no MIP solution is published for this selected size

For every exact public instance, the evaluator loads:

- robot initial positions;
- task x/y;
- EST;
- TWL;
- duration;
- precedence matrix;
- published pairwise distance matrix.

The V2.0 5D Task schema remains unchanged:

[
[x, y, priority, deadline, service\_time]
]

Mapping:

- x/y <- public coordinates
- priority <- constant 1 because TWPC-MRTA has no priority field
- deadline <- EST + TWL
- service_time <- DUR

EST and precedence are not added as learned features. They remain external hard
environment constraints. This preserves the frozen 139-parameter Gene.

### Hard constraints enforced

A task may be selected only if:

1. all predecessors have completed;
2. its service starts no earlier than EST;
3. its service starts no earlier than every predecessor's finish time;
4. travel uses the published distance matrix;
5. service finishes no later than EST + TWL.

If no feasible robot-task pair remains, the rollout stops and remaining tasks
count as incomplete.

### Published baselines loaded on the exact same instances

For 5R/18T:

- MIP
- BMRTA / Batch
- TePSSI / AuctionO

For 10R/36T:

- BMRTA / Batch
- TePSSI / AuctionO

The absence of a public MIP file at 10R/36T does not prevent certification of
the primary completion optimum when a feasible public method or the Gene itself
completes all 36 tasks, because 36 is the absolute task-count upper bound.

For every method the public solution provides:

- finished tasks;
- makespan;
- total distance;
- runtime.

The public MIP uses a 120-second Gurobi time limit and lexicographically
prioritizes maximizing assigned/completed tasks before its secondary cost.

### Exact primary optimality statement

The primary On-Time comparison metric is completed feasible tasks.

For an N-task instance:

[
Completed \le N
]

Therefore, if the frozen Gene itself produces a constraint-valid N/N solution,
the primary completion objective is globally optimal by construction:

[
Completed_{Gene}=N \Rightarrow Gap_{completion}=0
]

No solver certificate is needed for this count upper bound.

If Gene < N, compare to the best completed-task count among the three published
baselines. Do not call a sub-N MIP result globally optimal unless an optimality
certificate is available.

Makespan, distance, and runtime are reported as secondary diagnostics because
the TWPC-MRTA paper uses a different secondary objective from V2.0.

---

## Outputs

Each run writes:

- `protocol.json`
- `summary.json`
- `summary.csv`

The first run downloads public text instances and solution files into:

`runs/gene_mrta_v20/public_benchmark_cache`

Later runs reuse the cache.

---

## Commands

Regression tests:

```bash
bash tools/run_gene_mrta_v20_mac.sh tests
```

TT only:

```bash
bash tools/run_gene_mrta_v20_mac.sh public-tt
```

OnTime only:

```bash
bash tools/run_gene_mrta_v20_mac.sh public-ontime
```

Both:

```bash
bash tools/run_gene_mrta_v20_mac.sh public-benchmark
```

Expected log prefixes:

- `V20_PUBLIC_PROTOCOL`
- `V20_PUBLIC_TT`
- `V20_PUBLIC_ONTIME`
- `V20_PUBLIC_RUN_DIR`

---

## Claim rules

Allowed:

- "Gap to proven optimum" only for explicitly proven-optimal public values.
- "Gap to best published feasible baseline" for other mTSPLib cases.
- "Primary completion optimal" for a TWPC instance only when the frozen Gene
  completes all N tasks under all published hard constraints.
- "Zero-shot public benchmark transfer" because no public instance is used for
  training or mutation.

Not allowed:

- calling a non-starred CPLEX/LKH3 upper bound a proven optimum;
- comparing native V2.0 seconds directly to public benchmark seconds;
- claiming TWPC makespan optimality from task-count optimality;
- using public benchmark results to update the Gene Bank and still calling the
  evaluation zero-shot.
