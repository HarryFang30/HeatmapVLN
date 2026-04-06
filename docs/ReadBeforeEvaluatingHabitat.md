# InternNav 评估问题排查指南（续）

本文档是 [troubleshooting-guide.md](./troubleshooting-guide.md) 的续篇，记录在评估运行阶段发现的额外问题。

---

## 16. Episode 迭代逻辑 Bug：导航指令与环境不匹配（致命）

### 问题描述

**这是导致 SR 从论文报告的 0.63 暴跌到 0.23 的根本原因**。

在编写独立评估脚本时，错误地使用了"预过滤 episode 列表 + for 循环"的方式来迭代 episode，并从循环变量中获取 episode 的指令。但 `habitat.Env.reset()` 内部有自己的 episode 迭代器（`EpisodeIterator`），每次 `reset()` 都会调用 `next(self._episode_iterator)` 来覆盖 `current_episode`，完全忽略外部设置的 `env.current_episode = episode`。

```python
# 错误的写法（导致指令与环境不匹配）
for ep_idx, episode in enumerate(episodes_to_eval):
    env.current_episode = episode        # ← 会被 reset() 覆盖！
    observations = env.reset()            # ← 实际加载的是 iterator 中的下一个 episode
    
    # 从循环变量获取指令 —— 与实际环境不匹配！
    episode_instruction = episode.instruction.instruction_text
```

`habitat.Env.reset()` 的关键源码：

```python
def reset(self) -> Observations:
    ...
    self._current_episode = next(self._episode_iterator)  # ← 覆盖 current_episode
    self.reconfigure(self._config)
    observations = self.task.reset(episode=self.current_episode)
    ...
```

### 具体影响

- **首次运行（无 progress.json）**：`episodes_to_eval` 与 iterator 顺序一致，指令恰好匹配，结果正确
- **Resume 运行（有 progress.json）**：`episodes_to_eval` 跳过了已完成的 episode，但 iterator 仍从第 1 个 episode 开始。导致：
  - 循环变量 `episode` = 第 N 个未完成的 episode（如第 5 个）
  - `env.reset()` 实际加载 = iterator 中的第 1 个 episode
  - **模型收到 episode #5 的导航指令，却在 episode #1 的环境中导航**
  - 指令和环境完全不匹配，SR 大幅下降

在我们的实际运行中：
- 前 4 个 episode（首次运行）：SR ≈ 1.0（指令匹配，正确）
- 后 176 个 episode（resume）：大部分失败（指令不匹配）
- 加权平均 SR ≈ 0.23（远低于论文的 0.63）

### 解决方案

**核心原则：始终从 `env.current_episode` 获取实际加载的 episode 信息，而不是从外部列表**。

```python
# 正确的写法
seen_episodes = set()

while True:
    observations = env.reset()
    episode = env.current_episode       # ← 获取实际加载的 episode
    scene_id = episode.scene_id.split('/')[-2]
    episode_id = int(episode.episode_id)
    ep_key = (scene_id, episode_id)
    
    if ep_key in seen_episodes:
        break                           # iterator 已循环一圈，所有 episode 遍历完毕
    seen_episodes.add(ep_key)
    
    if ep_key in done_set:
        continue                        # 跳过已完成的 episode
    
    # 从实际 episode 获取指令 —— 保证匹配！
    episode_instruction = episode.instruction.instruction_text
    
    # ... 正常评估逻辑 ...
```

### 原始代码为何正确

原始的 `HabitatVLNEvaluator`（`habitat_vln_evaluator.py`）虽然在 `HabitatEnv.reset()` 中也设置了 `self._env.current_episode = ...`（同样被覆盖），但在获取 episode 信息时使用的是：

```python
# 原始代码（正确）
episode = self.env.get_current_episode()  # ← 获取实际加载的 episode
scene_id = episode.scene_id.split('/')[-2]
episode_instruction = episode.instruction.instruction_text
```

`get_current_episode()` 返回的是 `self._env.current_episode`，即 `reset()` 后实际加载的 episode。因此指令始终与环境匹配。

### 关键教训

1. **永远不要假设 `env.reset()` 会加载你指定的 episode**。Habitat 的 `EpisodeIterator` 有自己的迭代逻辑（分组、shuffle、cycling），`env.current_episode = X` 会被 `reset()` 覆盖
2. **在 `env.reset()` 之后，必须从 `env.current_episode` 读取 episode 元数据**（scene_id、episode_id、instruction 等）
3. **Resume 逻辑不能通过过滤 episode 列表实现**，只能通过在 `reset()` 后检查 `done_set` 来跳过已完成的 episode
4. 这个 bug 在首次运行时不会暴露（因为顺序恰好一致），只有 **resume 时才会出现**，非常隐蔽

### 修复效果

修复后重新运行评估（从零开始），前 7 个 episode 的 SR = **1.0**（7/7 全部成功），与论文报告的 0.63 SR 一致（甚至更好，因为样本量还小）。

---

## Habitat Episode 迭代机制详解

由于这个 bug 涉及 Habitat 的内部迭代机制，这里补充说明其工作原理，方便未来开发参考。

### EpisodeIterator 的关键行为

```python
class EpisodeIterator:
    def __init__(self, episodes, cycle=True, shuffle=False, group_by_scene=True, ...):
        if group_by_scene:
            self.episodes = self._group_scenes(episodes)  # 按 scene 分组排序
        self._iterator = iter(self.episodes)
    
    def __next__(self):
        next_episode = next(self._iterator, None)
        if next_episode is None:
            if not self.cycle:
                raise StopIteration
            self._iterator = iter(self.episodes)  # 循环重置
            next_episode = next(self._iterator)
        return next_episode
```

- `cycle=True`（默认）：遍历完所有 episode 后重新开始，永不抛出 `StopIteration`
- `group_by_scene=True`（默认）：episode 按 scene_id 分组排列
- `shuffle=False`（我们的配置）：不打乱顺序

### env.reset() 的完整流程

```
env.reset()
  ├── self._current_episode = next(self._episode_iterator)  # 获取下一个 episode
  ├── self.reconfigure(self._config)                        # 如果 scene 变了，重新加载
  ├── self.task.reset(episode=self.current_episode)         # 重置 task
  └── self._task.measurements.reset_measures(...)           # 重置所有 metric
```

每次 `reset()` 必定从 iterator 取下一个 episode，外部设置的 `current_episode` 会被覆盖。
