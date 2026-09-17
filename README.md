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

建議 Python 3.11 或 3.12。

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
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
