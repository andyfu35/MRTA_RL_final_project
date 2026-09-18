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
