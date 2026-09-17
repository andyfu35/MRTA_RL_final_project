# MRTA RL Final Project - MARL2D Prototype

這是一個先不依賴 ROS2 的單機原型，用來驗證未來四台電腦同步訓練的 Multi-Agent PPO 架構。

目前包含：

- 4 個獨立 PPO Agent：`runner_0`, `runner_1`, `blocker_0`, `blocker_1`
- 完全 deterministic 的 2D 差速輪運動學，不使用 MuJoCo 或剛體物理
- 20 m x 20 m 上視角競技場
- 每個 environment reset 都會生成自己的隨機矩形障礙物地圖
- 起點、終點安全區與障礙物間距限制
- 雙輪差速車外觀：車身、左右輪與 heading 都會跟著姿態旋轉
- 同步 round：四個 Agent 都完成一次 PPO update 後才 commit 下一輪
- 異質電腦效能模擬：每個 Agent 可以使用不同 `parallel_envs / rollout_steps / batches`
- 每輪固定相同 `samples_per_update`
- 外部 YAML 修改環境、Reward、PPO、效能 profile 與未來網路地址
- checkpoint、metrics 與 2D GIF 輸出

目前 `network.mode: mock`，因此四個 Worker 都在同一台電腦模擬。之後接 ROS2 時，會保留 Coordinator / Worker / Policy Exchange 的同步介面，只替換 transport 層。

## 1. 安裝

目前以 **Python 3.13** 作為本專案本地驗證環境。

### macOS / Linux

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

如果 PowerShell 禁止啟用虛擬環境：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. 先跑測試

```bash
python -m pytest -q
```

目前測試涵蓋：差速運動學、隨機矩形障礙物、碰撞回退、Observation、PPO、異質採樣、同步 Coordinator、Trainer 與 GIF renderer。

## 3. 最小訓練測試

```bash
python -m marl2d train --config config/default.yaml --rounds 1 --output runs/test1 --device cpu
```

應該會看到類似：

```text
=== COMMITTED ROUND 1 ===
runner_0  samples= 1024 ...
runner_1  samples= 1024 ...
blocker_0 samples= 1024 ...
blocker_1 samples= 1024 ...
```

四個 Agent 每輪都必須收集相同 samples，只有四個都完成才 commit 下一輪。

## 4. 生成 2D GIF

```bash
python -m marl2d render --checkpoint runs/test1/latest.pt --output runs/test1/evaluation.gif --max-steps 160 --device cpu
```

新版 renderer 會顯示：

- 藍色兩輪車：Runner
- 紅色兩輪車：Blocker
- 灰色矩形：隨機障礙物
- 綠色圓形：Goal
- 白色車頭標記：heading

## 5. 地圖與障礙物

預設設定位於 `config/default.yaml`：

```yaml
environment:
  width: 20.0
  height: 20.0
  goal: [18.0, 10.0]

  obstacles:
    count: 14
    min_width: 0.8
    max_width: 1.8
    min_height: 0.8
    max_height: 1.8
    min_spacing: 0.65
    spawn_clearance: 1.8
    goal_clearance: 2.0
```

每個平行 environment 都有自己的隨機地圖。使用相同 seed 時，地圖可完全重現。

障礙物採 axis-aligned rectangle；車子以圓形 footprint 做 collision query，碰撞時該步移動會 rollback，因此不會穿過障礙物。

## 6. 四台不同效能電腦的模擬

```yaml
training:
  samples_per_update: 1024

collection_profiles:
  runner_0:
    parallel_envs: 32
    rollout_steps: 32
    batches: 1
  runner_1:
    parallel_envs: 16
    rollout_steps: 32
    batches: 2
  blocker_0:
    parallel_envs: 8
    rollout_steps: 32
    batches: 4
  blocker_1:
    parallel_envs: 4
    rollout_steps: 32
    batches: 8
```

核心限制：

```text
parallel_envs * rollout_steps * batches = samples_per_update
```

所以目前四台分別是：

```text
32 * 32 * 1 = 1024
16 * 32 * 2 = 1024
 8 * 32 * 4 = 1024
 4 * 32 * 8 = 1024
```

弱電腦甚至可以：

```yaml
parallel_envs: 1
rollout_steps: 32
batches: 32
```

仍然是 1024 samples/update。快電腦先完成後只能等待，不會比其他 Agent 多做 update。

## 7. Reward

Reward 目前保留第一版設定，尚未進入正式調參階段：

```yaml
reward:
  runner:
    goal_bonus: 100.0
    team_progress: 2.0
    self_progress: 0.5
    collision: -2.0
    step: -0.01

  blocker:
    runner_progress: -2.0
    timeout_bonus: 100.0
    goal_failure: -100.0
    collision: -2.0
    step: -0.005
    proximity: 0.02
```

後續會另外討論合作 Runner 與 Blocker 的 Reward 設計，目前先不要把短期訓練結果視為最終行為。

## 8. 未來 ROS2

現在預留：

```yaml
network:
  mode: mock
  coordinator:
    ip: 127.0.0.1
    port: 7400
  agents:
    runner_0: {ip: 127.0.0.1, port: 7401}
    runner_1: {ip: 127.0.0.1, port: 7402}
    blocker_0: {ip: 127.0.0.1, port: 7403}
    blocker_1: {ip: 127.0.0.1, port: 7404}
```

後續目標：

```text
Round k PolicySet
    -> 每台電腦收集固定 samples
    -> 本機 PPO update
    -> 發布 Policy k+1 + READY
    -> Coordinator 等四台 READY
    -> COMMIT Round k+1
```

## 9. 專案結構

```text
config/default.yaml       環境 / Reward / PPO / 效能 profile / future network
src/marl2d/env.py         2D 差速車、隨機矩形地圖、碰撞、Observation、Reward
src/marl2d/policy.py      Actor-Critic network
src/marl2d/ppo.py         PPO update
src/marl2d/worker.py      單一 Agent rollout + local update
src/marl2d/coordinator.py 同步 round barrier
src/marl2d/exchange.py    Mock policy exchange；未來替換 ROS2 transport
src/marl2d/trainer.py     四 Agent 單機假分散式 trainer
src/marl2d/render.py      兩輪車 + 隨機方塊地圖 GIF renderer
src/marl2d/cli.py         train/render CLI
tests/                    自動測試
```

## 10. Experiment 1：Single Runner Navigation（正式基線）

目前已加入單車避障到終點的 PPO 實驗模式。此模式與未來 2v2 分開，目的先驗證單一 policy 是否能在未見過的隨機矩形地圖中穩定導航。

### Python 版本

目前本專案已在 **Python 3.13** 使用以下指令通過本地測試：

```bash
python -m pytest -q
```

macOS / Linux 建議：

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

Windows：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

### Action 與速度限制

PPO 輸出：

```text
[left_wheel, right_wheel] in [-1, 1]
```

每個輪子的 ground speed 線性映射到：

```text
[-2.0, 2.0] m/s
```

因此 `[1, 1]` 對應車體最高直線速度 2 m/s；20 m 橫向距離的理論最低時間為 10 s。

### 多世界 PPO

正式設定在 `config/single_runner.yaml`：

```yaml
training:
  samples_per_update: 8192

collection:
  parallel_envs: 32
  rollout_steps: 256
  batches: 1
```

即：

```text
32 worlds * 256 steps = 8192 samples / PPO update
```

32 個 world 共用同一個 Actor-Critic，但每個 world 使用不同且可重現的 map seed，episode 結束後該 world 立即換下一個 seed。訓練 action 使用 stochastic policy sampling；evaluation 使用 deterministic mean action。

### Reward Ablation

`single_runner_reward.mode` 支援四種模式：

```text
R0 = sparse terminal only
R1 = R0 + progress + time penalty
R2 = R1 + obstacle/wall safety shaping   <- formal baseline
R3 = R2 + small heading shaping
```

R2 非 terminal reward：

```text
R = 5 * (d_prev - d_now)
    - 0.01
    - 0.5 * max(0, (0.5 - clearance) / 0.5)^2
```

Terminal reward 會覆蓋 shaping：

```text
Goal      = +100
Collision = -100 and episode terminates
Timeout   = -20
```

R3 額外加入：

```text
+ 0.02 * cos(heading_error)
```

### 開始訓練

正式 R2 baseline：

```bash
python -m marl2d single-train \
  --config config/single_runner.yaml \
  --reward-mode R2 \
  --rounds 200 \
  --output runs/exp1_r2 \
  --device cpu
```

先做短 smoke test：

```bash
python -m marl2d single-train \
  --config config/single_runner.yaml \
  --reward-mode R2 \
  --rounds 1 \
  --output runs/exp1_smoke \
  --device cpu
```

### Held-out Evaluation

訓練 seed 與 evaluation seed 分開。預設 evaluation 從 10000 開始：

```bash
python -m marl2d single-eval \
  --checkpoint runs/exp1_r2/latest.pt \
  --episodes 100 \
  --seed-start 10000 \
  --output runs/exp1_r2/eval.json \
  --device cpu
```

主要指標：

- `success_rate`
- `collision_rate`
- `mean_time_to_goal_s`
- `mean_path_efficiency`
- `mean_episode_reward`
- `mean_min_clearance_m`

### Reward ablation 實驗

建議使用相同 seed、PPO 超參數與 samples/update，僅改 reward mode：

```bash
python -m marl2d single-train --config config/single_runner.yaml --reward-mode R0 --rounds 200 --output runs/exp1_r0
python -m marl2d single-train --config config/single_runner.yaml --reward-mode R1 --rounds 200 --output runs/exp1_r1
python -m marl2d single-train --config config/single_runner.yaml --reward-mode R2 --rounds 200 --output runs/exp1_r2
python -m marl2d single-train --config config/single_runner.yaml --reward-mode R3 --rounds 200 --output runs/exp1_r3
```

再各自以完全相同的 held-out seed range 評估，才能公平比較 reward shaping 對成功率、碰撞率、時間與路徑效率的影響。
