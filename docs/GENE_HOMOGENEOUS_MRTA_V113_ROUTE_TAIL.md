# V1.13 Route-Tail Autoregressive Multi-Task Assignment

## Objective

V1.13 changes the allocation semantics from one task per robot per decision
round to an ordered task queue per robot.

The desired output is:

R_i -> [T_i1, T_i2, T_i3, ...]

rather than only:

R_i -> T_ij

V1.12 remains reserved for self-evolving recombination. V1.13 is the
multi-task assignment branch.

## Core decoder change

V1.8-V1.11 used robot-row masking. Once robot i received one task during an
autoregressive decision round, row i was closed.

V1.13 removes robot-row closure.

At decoder step s:

1. score every feasible robot-task append pair;
2. choose the highest logit pair (i,j), unless STOP is higher;
3. close task column j only;
4. update robot i's virtual route tail;
5. recompute all pair observations;
6. continue until STOP, all tasks are assigned, or no feasible append exists.

Therefore the same robot may be selected repeatedly.

## Virtual route-tail state

For every robot i, V1.13 maintains:

- tail_node_i
- tail_position_i
- tail_time_i
- battery_i
- workload_i

If task j is appended, then:

tail_node_i <- task j

tail_position_i <- position(task j)

tail_time_i <- tail_time_i
               + path(tail_i,j)/speed
               + service_j

battery_i <- battery_i
             - path(tail_i,j) * energy_per_distance

workload_i <- workload_i
              + path(tail_i,j)/speed
              + service_j

The next candidate task for the same robot is therefore evaluated from the
end of its already planned route.

## Pair feasibility

Appending task j to robot i is feasible only if:

tail_time_i
+ path(tail_i,j)/speed
+ service_j
<= episode_horizon

and

path(tail_i,j) * energy_per_distance
<= battery_i

Deadlines remain an external performance axis, matching the existing MRTA
environment: a task can be feasible inside the episode horizon while still
being late relative to its own deadline.

## Observation semantics

The 12D V1.8 consequence-aware observation layout is retained.

The meanings are now evaluated from the virtual route tail:

1. Euclidean tail-to-task distance
2. path tail-to-task distance
3. service time
4. priority
5. deadline remaining at that robot's tail time
6. remaining battery
7. planned workload
8. projected competition among robots
9. future reachability after appending the candidate
10. best future time utility after appending
11. cross-robot opportunity cost
12. residual battery after appending

No new policy weights are added.

## Policy size

V1.13 remains:

12D pair observation

hidden_dim = 8

parameter_count = 148

A V1.10 Gene can therefore be lifted exactly into the V1.13 parameter
layout. Its decoder semantics change, however, so old performance is not
assumed to transfer.

## Decoder step normalization

The old decoder could select at most R pairs per round, so its step scalar
was normalized by robot count.

V1.13 can select up to T tasks in one planning round. Therefore:

step_norm = step / number_of_tasks

which keeps the decoder step feature in [0,1].

## Current scope

V1.13 is route-tail append only.

It does not yet insert a new task into the middle of an already planned
route. A later online version may compare:

- append-only queue updates;
- insertion at arbitrary route positions;
- partial replanning of future queues.

## Tests

The structural test creates two robots and five tasks. A deterministic Gene
with STOP disabled must produce:

Robot 0: [T0, T1, T2, T3, T4]

Robot 1: []

This verifies that one robot can receive multiple ordered tasks in a single
planning call and that each later task is evaluated from the previous task's
virtual completion state.

## Commands

Run implementation tests:

    bash tools/run_gene_mrta_v113_route_tail_mac.sh tests

Then run the V1.10 bootstrap smoke:

    bash tools/run_gene_mrta_v113_route_tail_mac.sh smoke

The smoke lifts the strongest currently certified V1.10 active Gene into the
same 148-parameter V1.13 layout and prints route queues on five frozen 95M
development worlds.

The bootstrap smoke is diagnostic only. Because the decoder semantics have
changed, V1.13 will need its own evolution phase before any performance claim.
