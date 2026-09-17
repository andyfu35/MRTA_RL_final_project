# MARL2D Local Prototype

這是一個先不依賴 ROS2 的單機原型，用來驗證：

- 2D 差速輪機器人運動學環境
- 4 個獨立 PPO Agent：`runner_0`, `runner_1`, `blocker_0`, `blocker_1`
- 每個 Agent 使用自己的 policy / value network
- 同步 round：四個 Agent 都完成 update 後才 commit 到下一輪
- 異質電腦效能模擬：每個 Agent 可用不同 `parallel_envs / rollout_steps / batches`
- 但每一輪固定收集相同 `samples_per_update`
- 外部 YAML 修改 reward、環境、PPO 與未來網路 IP
- checkpoint、metrics 與 2D GIF 輸出

目前 `network.mode: mock`，所以四台電腦都在同一個 Python process 中模擬。之後接 ROS2 時，會保留相同的 Coordinator / Worker / Policy Exchange 邏輯，只替換 transport 層。

## 1. Python 版本

建議：Python 3.11 或 3.12。

確認版本：

```bash
python --version
```

若系統使用 `python3`，下面所有 `python` 可改成 `python3`。

## 2. macOS / Linux 安裝

在專案根目錄：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

## 3. Windows PowerShell 安裝

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

如果 PowerShell 禁止啟用虛擬環境，可先在目前 PowerShell 執行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 4. 先跑測試

```bash
python -m pytest -q
```

目前應有 15 個測試，涵蓋環境、PPO、同步 coordinator、worker 與 trainer。

## 5. 最小 Smoke Test：只跑 1 個同步 round

```bash
python -m marl2d train \
  --config config/default.yaml \
  --rounds 1 \
  --output runs/test1 \
  --device cpu
```

Windows PowerShell 建議單行：

```powershell
python -m marl2d train --config config/default.yaml --rounds 1 --output runs/test1 --device cpu
```

你應該看到類似：

```text
=== COMMITTED ROUND 1 ===
runner_0  samples= 1024 ...
runner_1  samples= 1024 ...
blocker_0 samples= 1024 ...
blocker_1 samples= 1024 ...
```

最重要的是四個 Agent 的 `samples` 都必須相同，而且 round 只有全部完成後才 commit。

輸出：

```text
runs/test1/latest.pt
runs/test1/metrics.jsonl
```

## 6. 生成 2D 上視角 GIF

```bash
python -m marl2d render \
  --checkpoint runs/test1/latest.pt \
  --output runs/test1/evaluation.gif \
  --max-steps 160 \
  --device cpu
```

Windows：

```powershell
python -m marl2d render --checkpoint runs/test1/latest.pt --output runs/test1/evaluation.gif --max-steps 160 --device cpu
```

打開：

```text
runs/test1/evaluation.gif
```

即可看到四台車、障礙物與 Goal 的上視角結果。

## 7. 跑 10 個 round

`config/default.yaml` 預設 `rounds: 10`，所以：

```bash
python -m marl2d train --config config/default.yaml --output runs/train10 --device cpu
```

或者直接指定：

```bash
python -m marl2d train --config config/default.yaml --rounds 10 --output runs/train10 --device cpu
```

再生成 GIF：

```bash
python -m marl2d render --checkpoint runs/train10/latest.pt --output runs/train10/evaluation.gif --device cpu
```

## 8. 模擬四台不同效能電腦

`config/default.yaml`：

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

每台的有效採樣量都是：

```text
parallel_envs * rollout_steps * batches = samples_per_update
```

因此目前：

```text
runner_0 : 32 * 32 * 1 = 1024
runner_1 : 16 * 32 * 2 = 1024
blocker_0:  8 * 32 * 4 = 1024
blocker_1:  4 * 32 * 8 = 1024
```

假設未來某台電腦只能一次跑 1 個 world，可以改成：

```yaml
blocker_1:
  parallel_envs: 1
  rollout_steps: 32
  batches: 32
```

仍然是：

```text
1 * 32 * 32 = 1024 samples/update
```

所以快電腦只是更早完成並等待，不會多 update 幾次。

## 9. 修改 Reward

都在：

```text
config/default.yaml
```

例如 Runner：

```yaml
reward:
  runner:
    goal_bonus: 100.0
    team_progress: 2.0
    self_progress: 0.5
    collision: -2.0
    step: -0.01
```

Blocker：

```yaml
  blocker:
    runner_progress: -2.0
    timeout_bonus: 100.0
    goal_failure: -100.0
    collision: -2.0
    step: -0.005
    proximity: 0.02
```

修改 YAML 後不需要改 Python 程式。

## 10. 未來 ROS2 / 四台電腦 IP

現在已經預留：

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

目前 IP 不會真的建立 socket，因為 `mode: mock`。

之後接 ROS2 時預計改成：

```yaml
network:
  mode: ros2
```

並讓四台電腦各跑一個 Worker。同步規則不變：

```text
Round k PolicySet
    -> collect fixed samples
    -> local PPO update
    -> publish Policy k+1 + READY
    -> Coordinator 等四台 READY
    -> COMMIT Round k+1
```

## 11. GPU（之後再用也可以）

目前第一次測試建議：

```text
--device cpu
```

NVIDIA + CUDA PyTorch 安裝正確後可嘗試：

```bash
python -m marl2d train --config config/default.yaml --rounds 10 --output runs/cuda --device cuda
```

Apple Silicon 且 PyTorch MPS 可用時可嘗試：

```bash
python -m marl2d train --config config/default.yaml --rounds 10 --output runs/mps --device mps
```

但這個 2D simulator 本身非常輕，初期 CPU 就足以驗證整體架構。

## 12. 專案結構

```text
config/default.yaml       外部環境 / reward / PPO / network 設定
src/marl2d/env.py         2D 差速車環境、碰撞、觀測、reward
src/marl2d/policy.py      Actor-Critic network
src/marl2d/ppo.py         PPO update
src/marl2d/worker.py      單一 Agent rollout + local update
src/marl2d/coordinator.py 同步 round barrier
src/marl2d/exchange.py    mock policy exchange；未來替換 ROS2 transport
src/marl2d/trainer.py     四 Agent 單機假分散式 trainer
src/marl2d/render.py      2D GIF renderer
src/marl2d/cli.py         train/render CLI
tests/                    自動測試
```

## 13. 第一輪你要回傳給我的東西

請先跑：

```bash
python -m pytest -q
python -m marl2d train --config config/default.yaml --rounds 3 --output runs/local3 --device cpu
python -m marl2d render --checkpoint runs/local3/latest.pt --output runs/local3/evaluation.gif --device cpu
```

把以下貼給我：

1. `pytest` 最後結果
2. 三個 `COMMITTED ROUND` 的 console 輸出
3. 如果可以，傳 `runs/local3/evaluation.gif`
4. 你的 CPU / GPU 型號

下一階段就可以根據你的實際速度決定每台電腦的 `collection_profiles`，再開始接 ROS2。
