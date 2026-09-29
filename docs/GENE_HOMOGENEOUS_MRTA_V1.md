# Gene-based Homogeneous MRTA V1

This branch isolates a first controlled experiment for homogeneous multi-robot task allocation.
It intentionally does **not** train navigation or differential-drive control.

## Research question

Can one shared gene policy learn useful task-allocation behavior for homogeneous robots when evolution preserves three independent capability axes instead of collapsing them into one weighted reward?

## 2D world

- 100 x 100 continuous plane
- 4 homogeneous robots
- 20 homogeneous tasks
- no obstacles, collision, heading, wheel dynamics, or path planner
- robot speed = 2 distance units / second
- task service time = 3 seconds
- episode horizon = 50 seconds
- travel time = Euclidean distance / speed
- after assignment, the simulator stores that task as the robot's projected next-free position and schedules `busy_until = now + distance/speed + service_time`

The simulator is event-based. It does not integrate motion with a small `dt`. Reserved tasks are removed immediately, while the robot cannot bid again until its scheduled finish time.

## Gene observation and output

For each currently free robot-task pair, the shared gene sees four normalized values:

1. `distance_norm`: Euclidean distance / world diagonal
2. `robot_load_norm`: completed task count / fair task share
3. `competition_norm`: fraction of the other robots that are closer to the task
4. `slack_norm`: remaining episode time minus estimated travel+service duration, normalized by episode time

The gene is deliberately small and interpretable:

```text
bid = w_distance * distance_norm
    + w_load * robot_load_norm
    + w_competition * competition_norm
    + w_slack * slack_norm
    + bias
```

The same gene is shared by every robot. The simulator builds all free robot-task bids and performs a greedy global matching by bid.

## Capability axes

Evolution keeps three axes separate:

### 1. Completion

```text
completion = completed_tasks / total_tasks
```

### 2. Efficiency

For every completed assignment, route efficiency is `1 - distance/world_diagonal`. The axis is the mean over completed assignments.

### 3. Balance

Jain's fairness index over completed task counts per robot:

```text
balance = (sum(load_i)^2) / (N * sum(load_i^2))
```

## Gene bank evolution

No weighted scalar reward is used for parent selection.

Each generation:

1. every candidate is evaluated on the same sampled worlds;
2. top genes are stored independently in completion, efficiency, and balance archives;
3. parents are sampled across those archives;
4. genes are crossed and mutated;
5. a small random immigrant fraction preserves exploration.

At the end, all candidates are evaluated on a fixed validation set. A maximin reference gene is reported only as a convenient single candidate for inspection; it is **not** the evolutionary reward.

## Mac quick start

```bash
python3 -m venv .venv-gene
source .venv-gene/bin/activate
python -m pip install --upgrade pip
pip install -r requirements_gene_mrta_v1.txt
export PYTHONPATH="$PWD/src"
pytest -q tests/test_gene_mrta_v1.py
python -m marl2d.gene_mrta_v1.train --generations 80 --population 96 --worlds-per-generation 8 --validation-worlds 64 --seed 7
```

Results are written under `runs/gene_mrta_v1/` as `summary.json` and `history.csv`.

For a very short sanity run first:

```bash
python -m marl2d.gene_mrta_v1.train --generations 5 --population 24 --worlds-per-generation 3 --validation-worlds 12 --seed 7
```
