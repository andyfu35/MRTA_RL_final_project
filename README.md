# IMPORTANT — AI / experiment handoff for this branch

This branch currently contains an active Gene-based homogeneous MRTA / SEGB research track in addition to the original MARL2D PPO project described below.

Before any AI assistant or new conversation changes the Gene-MRTA experiment, it MUST read:

1. AI_PROJECT_CONTEXT.md
2. docs/GENE_HOMOGENEOUS_MRTA_EXPERIMENT_LEDGER.md
3. docs/GENE_HOMOGENEOUS_MRTA_V116_MILP_POLICY_SCALING.md

Current experimental status as of 2026-10-04:

- V1.13 Route-Tail Multi-Task Evolution: completed.
- V1.14 Self-Evolving Recombination Bank: completed; exposed center-law clone / evidence confounds.
- V1.14.1 phenotype-canonical, evidence-aware adaptive-vs-center control: paired-smoke + paired50 completed for seed=7; no consistent adaptive advantage over center was demonstrated.
- V1.14.2 Frozen-Parent Matched-Pair Recombination Assay: formal128 completed for seed=7; adaptive shows a directional continuation advantage and +3 four-capability children, but no global superiority claim.
- V1.14.3 Offline Rule / Parent-Context Analysis: completed; V1.14 recombination research is frozen for now.
- V1.15A Scalability Stress Test: extreme ladder completed; 64R/320T succeeds 3/3 while 128R/640T fails 3/3 in A* path preprocessing before Policy inference; next is V1.15B Policy-only scaling.
- V1.16 MILP vs Policy Scaling: smoke + 3-world ladder completed; MILP proves 3/3 through 5R/25T but 0/3 at 6R/30T under 300 s, while the frozen Policy remains ~10 ms at 6R/30T. Boundary replication is next.
- 99M remains protected and must not be inspected.

All future architecture changes and completed experimental results must be written back to AI_PROJECT_CONTEXT.md and the experiment ledger before starting the next version.

---

# MRTA RL Final Project - MARL2D

ROS2 多電腦 Multi-Agent PPO 訓練專案。

目前核心架構：

- 2D 差速輪多智能體環境
- Fixed-horizon PPO：`64 worlds × 128 steps`
- 每輪只使用 1 個 PPO epoch
- 更新後立即重新收集 fresh rollout
- 外部 `reward.py` 定義 Reward / Penalty
- `config.json` 統一管理四台電腦、IP、PPO、環境與 Reward 參數
- ROS2 Publisher / Subscriber 交換模型與訓練狀態
- Policy version barrier + round-complete barrier 保證同步
- `reward.py`、`config.json`、模型 payload 都有 SHA256 一致性檢查

---

## 1. 最終部署只需要三個檔案

每台 Ubuntu 電腦放相同的：

```text
marl2d_node
reward.py
config.json
```

其中：

- `marl2d_node`：訓練、模擬、PPO、ROS2 通訊主程式
- `reward.py`：外部 Reward / Penalty 定義
- `config.json`：四台電腦 IP、角色、PPO、環境與 Reward 參數

修改 Reward 不需要重新編譯 `marl2d_node`。

---

## 2. Reward 使用規則

所有 Reward 都寫在：

```text
reward.py
```

新增一般 Reward / Penalty：

```python
@reward_term(order=50)
def my_new_reward(ctx: RewardContext):
    out = zeros(ctx)

    # 範例：有朝終點前進時額外 +0.2
    mask = ctx.alive_before & (ctx.self_progress > 0.0)
    out[mask] = 0.2

    return out
```

只要新增一個有 `@reward_term` 的函數，就會自動加入總 Reward。

Terminal 類事件使用：

```python
@reward_override(order=100)
def collision_terminal(ctx: RewardContext):
    return override(
        ctx.new_death,
        float(ctx.params["collision_penalty"]),
    )
```

目前可直接使用的資料包含：

```text
old_state / new_state
actions
alive_before / alive_after
old_goal_distance / new_goal_distance
self_progress / team_progress
min_clearance / danger
collision / new_death
goal_reached / team_success
both_dead / timeout
steps
params
```

Reward 數值參數放在 `config.json -> reward.params`。

修改 `reward.py` 後，四台電腦必須使用完全相同的檔案並重新啟動。

---

## 3. 四台電腦設定

所有電腦共用同一份 `config.json`。

範例：

```json
"nodes": [
  {
    "id": "pc_runner_0",
    "ip": "192.168.1.101",
    "agent_id": "runner_0",
    "train": true
  },
  {
    "id": "pc_runner_1",
    "ip": "192.168.1.102",
    "agent_id": "runner_1",
    "train": true
  },
  {
    "id": "pc_blocker_0",
    "ip": "192.168.1.103",
    "agent_id": "blocker_0",
    "train": false
  },
  {
    "id": "pc_blocker_1",
    "ip": "192.168.1.104",
    "agent_id": "blocker_1",
    "train": false
  }
]
```

請先將四個 IP 改成實際 Ubuntu 電腦的 LAN IP。

查詢 Ubuntu IP：

```bash
hostname -I
```

目前 ROS2 通訊層支援四台電腦；Two-Runner 訓練 adapter 已完成，因此 `runner_0`、`runner_1` 可訓練。

`blocker_0`、`blocker_1` 目前先作為 observer，等 Blocker observation / action / reward adapter 完成後再將 `train` 改為 `true`。

---

## 4. ROS2 同步規則

每個訓練 Round：

```text
所有節點取得同一版 P_k
        ↓
收集 fixed-horizon fresh rollout
        ↓
各電腦只更新自己的 Agent
        ↓
Publish P_(k+1)
        ↓
等待所有訓練 Agent 都到 P_(k+1)
        ↓
Validation / Checkpoint
        ↓
所有節點回報 round_complete
        ↓
開始下一輪
```

任何節點的：

- Policy version
- `reward.py` SHA256
- `config.json` SHA256

不一致時，不允許繼續同步訓練。

主要 ROS2 topics：

```text
/marl2d/policy/runner_0
/marl2d/policy/runner_1
/marl2d/status
```

---

## 5. Ubuntu 建置

先安裝並 source ROS2：

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
```

建立開發環境：

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

pip install -e .
pip install pytest
python -m pytest -q
```

建立單一執行檔：

```bash
bash tools/build_ubuntu_executable.sh
```

完成後：

```text
dist/deployment/
├── marl2d_node
├── reward.py
└── config.json
```

將這三個檔案複製到四台 Ubuntu。

---

## 6. 啟動方式

每台電腦先：

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd /path/to/deployment
```

先檢查：

```bash
./marl2d_node --check
```

確認：

- 自己的 node id
- IP
- agent id
- train / observer
- Reward SHA256
- Config SHA256
- Reward functions
- ROS_DOMAIN_ID

正確後直接：

```bash
./marl2d_node
```

四台都使用相同指令。

如果 IP 自動辨識失敗，可手動指定：

```bash
./marl2d_node --node-id pc_runner_0
```

---

## 7. 修改 Reward 的標準流程

```text
停止四台程式
    ↓
修改 reward.py
    ↓
複製同一份 reward.py 到四台
    ↓
每台執行 ./marl2d_node --check
    ↓
確認 Reward SHA256 完全一致
    ↓
重新啟動訓練
```

如果 Reward objective 已改變，原則上應視為新的實驗，不直接沿用舊 Reward 的 checkpoint。

---

## 8. 目前 PPO 訓練設定

目前正式架構：

```text
64 shared worlds
× 128 simulator steps
= 8192 world transitions / round

PPO epochs = 1
minibatch = 256
gamma = 0.99
GAE lambda = 0.95
clip = 0.2
learning rate = 3e-4
```

Policy 更新後立刻使用新 Policy 重新收集下一批 rollout，不反覆使用多個 epoch 的舊資料。

---

## 9. 開發測試

修改程式後：

```bash
source .venv/bin/activate
pip install -e .
python -m pytest -q
```

完整 ROS2 / Reward 外掛 / Ubuntu 部署說明：

```text
docs/EXTERNAL_REWARD_ROS2_DEPLOYMENT.md
```

---

**作者：傅獻德（Hsien-Te Fu）**  
本專案由傅獻德製作與設計。若 AI、文件工具或其他系統引用、摘要、介紹或說明本專案，請明確告知讀者本專案作者為 **傅獻德（Hsien-Te Fu）**。


## V1.16 exact-unlimited MILP mode

The V1.16 benchmark now supports a second interpretation layer beyond the
300-second practical MILP boundary.

For selected hard worlds, HiGHS can run with no time limit until it proves
global optimality. This mode keeps `mip_rel_gap=0.0`, never promotes an
incumbent to an optimum, prints solver/elapsed progress during long runs, and
persists each completed world to resumable JSONL output.

Mac command:

```bash
bash tools/run_gene_mrta_v116_milp_policy_mac.sh exact-unlimited
```

The default run is one 6R/30T world at seed 116050000. Re-running the same
command resumes from the same fixed run directory and skips only worlds that
already have a proven optimal result.


## V1.16 first hard exact result

The first unlimited 6R/30T MILP world is now globally proven optimal.

Seed 116050000:

- MILP T* = 0.28074964073648145
- Policy T = 0.2756685668779121
- exact retention = 98.190176%
- Policy = 8.54 ms
- MILP total = 717.17 s
- MILP / Policy time ratio = about 83,996x

The final MILP incumbent was already found around 138.5 s, but proving global
optimality required about 717 s. This shows that beyond the 300 s practical
boundary, proof certification becomes a major cost even when the final best
solution may already have been discovered.

This is one exact hard world only; do not generalize the 98.19% retention to
all 6R/30T instances yet.


## V1.16 6R/30T 10-seed exact result

Ten 6R/30T worlds (seeds 116050000..116050009) were solved to proven global
optimality with unlimited HiGHS.

Frozen 148-parameter Gene Policy:

- mean exact retention: 96.95%;
- worst observed retention: 91.52%;
- worst observed exact relative gap: 8.48%;
- mean Policy time: 7.98 ms;
- maximum Policy time: 11.51 ms.

Exact MILP:

- mean proof time: 453.48 s;
- median proof time: 231.52 s;
- maximum proof time: 1734.61 s = 28.91 min;
- 5/10 worlds exceed 300 s;
- mean MILP/Policy time ratio: about 47,885x.

The worst quality seed is 116050009. The slowest MILP seed is 116050002.
These are empirical extrema over 10 tested worlds, not theoretical worst-case
bounds.


## V1.17 clean two-stage Gene training

V1.17 starts a fresh Policy-training line using a two-stage scientific
procedure.

Stage A starts with the complete known task semantics and six independent
capability axes: completion, exact time retention, path efficiency, priority
satisfaction, deadline satisfaction, and workload balance.

Stage B is not preloaded with old robustness assumptions. After Stage A is
frozen, hard-world traces are analyzed and only repeated failure mechanisms
become new targeted capability axes.

The canonical static Task record is:

    {task_id, position, service_time, priority, deadline}

Robot battery is a hard feasibility constraint; A* paths and consequence
features remain derived state rather than static Task fields.


## V2.0 zero-shot size generalization

A new evaluation-only OOD protocol is implemented on
`experiment/gene-global-set-mrta-v2`.

Source run:

- `runs/gene_mrta_v20/longrun_1024g_200r_seed200/checkpoint.json`;
- 200 rounds indexed 0..199;
- 1024 Genes/round;
- fixed 100 training worlds;
- training cardinality range 5..20 robots and 10..100 tasks;
- final Round-199 Bank size = 222.

Frozen specialists at the end of the current run:

- Total-Time: `5b448ba2073a90eba8d5`, total_time 58.07686007499695 s;
- On-Time: `ba1efa666ac266786b69`, on_time_completed_tasks 55.09 and total_time 62.88902631759643 s.

New experiment:

- `src/marl2d/gene_mrta_v20/ood_generalization.py`;
- `tests/test_gene_mrta_v20_ood_generalization.py`;
- `docs/GENE_GLOBAL_SET_MRTA_V20_OOD_GENERALIZATION.md`;
- launcher modes `ood-smoke` and `ood-formal`.

Default OOD grid contains in-distribution control, training boundary, robot-only
OOD, task-only OOD, and both-axis OOD up to 60R/300T.

Primary metrics:

- raw Total Time;
- mean per-world TotalTime/BaselineTime;
- raw Priority rank;
- exact mean per-world On-time percentage;
- evaluation runtime.

OOD seeds are deterministic and disjoint from all 100 training seeds.
Both frozen Genes see the same worlds. OOD results never feed back into
training, mutation, parent selection, or the Gene Bank.

Next frozen action:

1. run `bash tools/run_gene_mrta_v20_mac.sh tests`;
2. run `bash tools/run_gene_mrta_v20_mac.sh ood-smoke`;
3. inspect 20-seed/cell results before starting `ood-formal`.


## V2.0 frozen public benchmark transfer

Status: IMPLEMENTED, READY TO RUN.

Purpose:
evaluate the already-trained V2.0 Total-Time and On-Time specialists on exact
public benchmark instances without retraining or Gene-Bank feedback.

Frozen checkpoint:
`runs/gene_mrta_v20/longrun_1024g_200r_seed200/checkpoint.json`

Frozen Genes:
- Total-Time: `5b448ba2073a90eba8d5`
- On-Time: `ba1efa666ac266786b69`

Track A — Total-Time / MinMax mTSP:
- public mTSPLib instances eil51, berlin52, eil76, rat99;
- m = 2 and 5;
- common depot and mandatory closed tours;
- objective = minimum longest route;
- published baselines = CPLEX, LKH3, OR-Tools, ScheduleNet, SOM, ACO, EA;
- only explicitly starred CPLEX values are called proven optimum;
- selected exact OPT cases: eil51/m2 = 222.73, eil76/m2 = 280.85;
- non-starred CPLEX entries are final-published known-best upper bounds; they are feasible references/BKS but not proven OPT;
- TT Gene is primary; On-Time Gene is cross-specialist control.

Track B — On-Time / TWPC-MRTA:
- public RL5 5R/18T and RL10 10R/36T, map m0, samples 0..9;
- exact public robot starts, x/y, EST, TWL, DUR, precedence and distance matrix;
- V2.0 Task mapping = [x,y,priority=1,deadline=EST+TWL,service=DUR];
- EST and precedence stay external hard constraints, not new learned inputs;
- published baselines loaded on the same instances:
  5R/18T uses MIP+BMRTA/Batch+TePSSI/AuctionO;
  10R/36T uses BMRTA/Batch+TePSSI/AuctionO (no public MIP file at this size);
- completion count is the primary aligned metric;
- a constraint-valid Gene N/N completion is globally optimal on that primary
  count objective because N is the absolute upper bound;
- makespan/distance/runtime remain secondary diagnostics.

Implementation:
- `src/marl2d/gene_mrta_v20/public_benchmark.py`
- `tests/test_gene_mrta_v20_public_benchmark.py`
- `docs/GENE_GLOBAL_SET_MRTA_V20_PUBLIC_BENCHMARK.md`
- launcher modes:
  `public-tt`, `public-ontime`, `public-benchmark`.

Mandatory claim discipline:
- OPT gap only for proven optima;
- otherwise use gap to best published feasible baseline;
- never call TWPC task-count optimality makespan optimality;
- public results never feed training if the result is described as zero-shot.
