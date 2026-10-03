# V1.10 Evolutionary Mating

## Motivation

V1.9 demonstrated that specialist discovery can succeed while fixed-width
crossover fails to combine capabilities into one robust generalist. V1.10
therefore adds an explicit mating path alongside ordinary mutation.

The policy architecture remains unchanged:

- 12D pair observation
- hidden dimension 8
- 148 parameters
- Direct Assignment
- learned STOP/WAIT

The experiment isolates whether repeated model recombination can accumulate
capabilities without increasing network capacity.

## Population split

Each generation contains 256 new offspring:

[
N = 128_{normal} + 128_{mating}
]

Normal offspring use mutation and parent sampling with the historical
quality exponent:

[
P_{normal}(G_i)
propto Q_i^2
]

Mating offspring use much stronger elite pressure:

[
P_{mating}(G_i)
propto Q_i^{10}
]

A 5 percent uniform component remains to prevent complete reproductive
collapse:

[
P'(i)=0.95P_{Q^{10}}(i)+0.05/|G|
]

## Quality weight without scalarized capability rewards

A Gene has a declared capability set (C_G). For each capability axis (a),
let (B_a) be the best active Gene score on that axis.

The reproductive quality weight is:

[
Q(G)
=
min_{ain C_G}
rac{S_a(G)}{B_a}
]

clipped to ([0,1]).

This is not a weighted reward sum. A multi-capability Gene receives a high
mating probability only when it remains close to the current ceiling on
every capability it claims to carry.

## Mating families

The default mating budget is:

[
32 parent pairs
	imes
4 children/pair
=
128 mating offspring
]

A second parent is preferentially drawn from Genes that add at least one
capability not already carried by the first parent.

Each four-child family samples four distinct recombination operators from:

1. parameter blend
2. hidden-block pick
3. hidden-block blend
4. common-ancestor delta arithmetic
5. TIES-style delta merge
6. DARE-style stochastic delta merge

The common ancestor for delta operators is the frozen V1.8 Gene.

Mating-child mutation is disabled by default in the first experiment so
successful inheritance can be attributed to recombination rather than a
post-merge mutation.

## Functional hidden blocks

The 148-parameter policy is partitioned into eight hidden-unit blocks plus
four global scalar parameters.

A hidden block contains all parameters that belong to one hidden coordinate:

- 12 encoder input weights
- encoder bias
- pair decoder weight
- global-context decoder weight
- robot-context decoder weight
- task-context decoder weight
- STOP decoder weight

Thus a block is inherited as a functional unit rather than by blindly
swapping individual scalars.

## Fixed diverse 100-world mating bank

V1.10 uses a dedicated development namespace beginning at seed 95,000,000.

The builder generates 500 candidate worlds and describes each world using
policy-independent structural features including:

- initial battery mean and spread
- service-time mean and spread
- deadline mean and spread
- priority mean
- task spatial dispersion
- robot-task distance
- path detour
- initial feasible-pair ratio
- initially unreachable-task fraction
- singleton-task fraction
- obstacle area
- path connectivity

The descriptors are standardized and deterministic farthest-point sampling
selects exactly 100 worlds that cover this structural space.

No trained policy score is used when selecting the 100 worlds.

Exact MILP (T^*) is then solved once and cached for all 100 selected worlds.

The development set was frozen after construction. One initially selected
world, seed 95,000,034, could not prove exact optimality within the solver
budget (300 s followed by a 900 s retry; final recorded MIP gap about 0.562).
It was therefore permanently replaced by the nearest unused candidate in the
same standardized descriptor space, seed 95,000,442. The replacement proved
exact optimality with MIP gap 0.0. After this one policy-independent solver
feasibility substitution, the 100-seed list is immutable; later build-bank
runs only reconstruct/verify that frozen set and do not perform dynamic
replacement.

Protected ranges:

- 98,000,000-98,000,099: historical V1.8/V1.9 development evidence
- 99,000,000-99,000,099: untouched V1.10 final benchmark

## Mating evaluation

To control cost, each generation rotates through 25 of the 100 mating worlds
for screening.

All 256 new offspring are screened.

Normal path:
- top 4 candidates per capability axis advance to full 100-world evaluation.

Mating path:
- offspring are ranked by their minimum approximate inheritance retention on
  the 25 screening worlds.
- top 16 mating candidates advance to full 100-world evaluation.

The full 100-world capability vector is:

[
[
T_{mean},
T_{tail10},
C_{continuation},
R_{fleet}
]
]

where:

[
T_{mean}
=
mean_Wleft(T_G(W)/T^*(W)ight)
]

and:

[
T_{tail10}
=
meanleft(
lowest 10 percent ofT_G(W)/T^*(W)
ight)
]

The other two axes retain the V1.9 external continuation and fleet-reserve
metrics.

## Cumulative inheritance

Suppose Parent A carries capability set (C_A) and Parent B carries (C_B).

The child is required to inherit:

[
C_C=C_Acup C_B
]

For every capability (ain C_C), the baseline is the strongest relevant
parent score:

[
B_a
=
max_{
Pin{A,B}:ain C_P
}
S_a(P)
]

The child retention is:

[
R_a(C)
=
rac{S_a(C)}{B_a}
]

A mating is accepted only when:

[
oxed{
R_a(C)ge0.95
quad
orall ain C_C
}
]

There is no averaging that allows improvement on one capability to compensate
for losing another.

A successful child is written to the Hybrid Gene Bank together with:

- complete weights
- parent IDs
- inherited capability set
- recombination operator
- operator parameters
- per-axis scores
- per-axis inheritance retention
- generation

Thus capability accumulation can proceed as:

[
M
ightarrow
MH
ightarrow
MHF
ightarrow
MHFC
]

if every mating passes the external inheritance test.

## Outputs

Each V1.10 run records:

- history.csv
- checkpoint.json
- mating_events.jsonl
- summary.json

mating_events.jsonl stores the full accepted Gene weights, so successful
recombinations remain auditable even if the active reproductive bank is later
pruned for efficiency.

## Protocol

1. Reconstruct/verify the frozen 95M diverse 100-world mating bank.
2. Run implementation tests.
3. Use the latest frozen V1.9 checkpoint as the starting Gene Bank.
4. Run a short V1.10 smoke.
5. Run V1.10 mating evolution.
6. Freeze the V1.10 candidate and procedure.
7. Only then evaluate once on untouched 99M worlds.

The exact oracle never supplies an action label. It only provides the
capability ceiling used for external evaluation.
