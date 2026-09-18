# External Reward + ROS2 Distributed Runtime

## Goal

The training core is now separated from three deployment-facing files:

```
marl2d_node
reward.py
config.json
```

The intended Ubuntu deployment workflow is:

1. Build `marl2d_node` once on an Ubuntu machine with ROS2 installed.
2. Edit `config.json` with the four computer IP addresses.
3. Edit `reward.py` whenever a new reward or penalty is required.
4. Copy the same three files to every computer.
5. Start `./marl2d_node` on each computer.

The executable automatically identifies the local computer from the IP list, starts ROS2 publishers/subscribers, exchanges policy versions, checks reward/config hashes, and synchronizes training rounds.

## 1. Reward plugin

The environment no longer has to contain the reward equation.

The deployment file:

```
reward.py
```

is loaded dynamically at process startup.

### Add a new dense reward or penalty

Copy this template:

```python
@reward_term(order=50)
def my_new_reward(ctx: RewardContext) -> np.ndarray:
    out = zeros(ctx)

    # Example:
    # add +0.2 while an alive robot is making positive goal progress
    mask = ctx.alive_before & (ctx.self_progress > 0.0)
    out[mask] = 0.2

    return out
```

Nothing in the simulator or PPO implementation has to be modified.

All functions decorated with `@reward_term` are added together automatically.

Available context includes:

- `old_state`, `new_state`
- `actions`
- `alive_before`, `alive_after`
- `old_goal_distance`, `new_goal_distance`
- `self_progress`
- `team_progress`
- `min_clearance`
- `danger`
- `collision`, `new_death`
- `goal_reached`
- `team_success`
- `both_dead`
- `timeout`
- `steps`
- `params` from `config.json -> reward.params`

### Terminal/event override

For events that must replace the accumulated dense reward, use:

```python
@reward_override(order=100)
def collision_terminal(ctx: RewardContext):
    return override(
        ctx.new_death,
        float(ctx.params["collision_penalty"]),
    )
```

Higher `order` runs later.

The supplied template uses this precedence:

```
dense reward terms
-> collision override
-> timeout override
-> team-success override
```

### Reward synchronization

Every policy ROS2 message contains the SHA256 of `reward.py`.

If another training computer is running a different reward file, the receiver rejects the model instead of silently training under inconsistent objectives.

After changing `reward.py`, restart every training node.

## 2. config.json

The same `config.json` should be copied to every computer.

Important sections:

```json
{
  "reward": {
    "file": "reward.py",
    "params": {}
  },
  "network": {
    "ros_domain_id": 42,
    "validation_node_id": "pc_runner_0",
    "nodes": [
      {
        "id": "pc_runner_0",
        "ip": "192.168.1.101",
        "agent_id": "runner_0",
        "train": true
      }
    ]
  }
}
```

The executable tries to identify the current computer by matching its Ubuntu IPv4 addresses against `network.nodes[].ip`.

This also works on an offline laboratory LAN: the program probes routes to the configured peer IPs and does not require Internet access.

If automatic detection is inconvenient:

```bash
./marl2d_node --node-id pc_runner_0
```

can explicitly select the node.

## 3. Current four-computer template

The supplied config contains four computers:

```
pc_runner_0   runner_0   train=true
pc_runner_1   runner_1   train=true
pc_blocker_0  blocker_0  train=false
pc_blocker_1  blocker_1  train=false
```

The ROS2 transport and membership layer already supports all four computers.

The current simulator/training adapter still implements only the two Runner policies. Therefore the Blocker computers currently start in observer mode and receive policy/status traffic.

This is intentional: Blocker observation/action/reward dynamics have not been implemented yet. The communication layer is generic so adding the Blocker adapter later does not require redesigning model exchange.

## 4. ROS2 topics

Current topics:

```
/marl2d/policy/runner_0
/marl2d/policy/runner_1
/marl2d/status
```

Policy topics use reliable + transient-local QoS so a node that starts slightly later can receive the latest model version.

Each policy message contains:

- node id
- agent id
- policy version
- compressed model state
- model payload SHA256
- reward.py SHA256
- config.json SHA256
- update metrics

The model decoder uses `torch.load(..., weights_only=True)` and enforces an 8 MiB payload limit.

## 5. Version barrier

For policy version `P_k`:

```
all training nodes receive P_k
        |
        v
collect fixed-horizon rollout
        |
        v
each computer updates only its own policy
        |
        v
publish local P_(k+1)
        |
        v
wait until every training agent has P_(k+1)
        |
        v
validation / checkpoint
        |
        v
all trainers publish round_complete(k+1)
        |
        v
round-complete barrier
        |
        v
start the next rollout
```

This prevents one computer from training on a newer model while another is still validating an older version.

## 6. Current distributed Two-Runner computation

Each Runner computer holds its own optimizer and updates only its local policy.

Both computers reconstruct the same frozen joint policy set from ROS2 and run the same fixed-horizon shared-world simulation. With the same config, seed, policy versions, and reward file, the collector is deterministic across nodes.

Formal collection remains:

```
64 worlds x 128 steps
1 PPO epoch
synchronous model commit
fresh rollout after every commit
```

## 7. Ubuntu setup

ROS2 must already be installed.

Open a terminal and source the installed ROS2 distribution:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
```

For development/building, using system ROS2 packages from a venv is easiest with:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

pip install -e .
pip install pytest
python -m pytest -q
```

Before building the standalone executable:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
source .venv/bin/activate

bash tools/build_ubuntu_executable.sh
```

Output:

```
dist/deployment/
├── marl2d_node
├── reward.py
└── config.json
```

Build the binary on the same Ubuntu/ROS2 family used for deployment because ROS2 and Torch include native libraries.

## 8. Copy to all four computers

Copy exactly the same three files to each machine.

Before copying, edit the four IP addresses in `config.json`.

On each Ubuntu computer, check its address with:

```bash
hostname -I
```

Then test the three-file bundle:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd /path/to/deployment

./marl2d_node --check
```

The output shows:

- detected node id
- IP
- agent id
- train/observer mode
- config SHA256
- reward SHA256
- registered reward functions
- ROS domain id

If correct, start:

```bash
./marl2d_node
```

Do this on all four computers.

## 9. Changing rewards later

Typical workflow:

1. Stop all training nodes.
2. Edit one copy of `reward.py`.
3. Add/remove/modify `@reward_term` functions.
4. Copy that exact `reward.py` to all computers.
5. Start `./marl2d_node --check` on each machine.
6. Confirm every machine prints the same reward SHA256.
7. Start training.

No executable rebuild is required for reward changes.

## 10. Checkpoint behavior

Each training computer writes its own local state under:

```
runs/distributed/<node_id>/latest.pt
```

If `training.auto_resume` is enabled, the executable verifies the checkpoint's config/reward hashes before loading it.

Do not enable auto-resume after intentionally changing `reward.py`; that is a new objective and should normally start a new run.
