# EXP-18 复现命令（第一视角拓扑热力可视化）

判据见台账 [README.md](README.md) 的 EXP-18 条目。**开跑前把判据再读一遍，跑完只填结果，不改判据。**

全部在**开发机**上跑：渲染与俯视图是纯 CPU（llvmpipe），VO 缓存与预测导出用 **GPU 7/6/5**（≤ 3 卡，台账 §0 第 7 条，不占 0 卡）。
代码不从共享检出跑（§5 第 27 条）：把要跑的 commit 原样 `git archive` 到产物根下的独立目录。

```bash
R=/mnt/afs/liwenhao/agent/370910109
EXP=$R/model/exp18_first_person_viz          # 产物根（导出、指标、图、日志、源码副本）
RENDER=$R/data/exp18_renders                 # C/D/E 新渲染 + 它们的 VO 缓存
SHA=01a76ec                                  # 本次正式运行的代码版本
SRC=$EXP/src_$SHA
```

远程命令一律走 login shell（`CLAUDE.md` §3.1）；后台长任务用 `ssh -n -f … 'bash -lc "setsid nohup … &"'`，
普通 `ssh … &` 会一直挂到任务结束。

---

## 0. 暂存源码

本地（本分支）：

```bash
git archive $SHA | ssh finn_cci_c500 "mkdir -p $SRC && tar -x -C $SRC && git rev-parse $SHA > $SRC/.exp18_git_sha"
```

`.exp18_git_sha` 会被导出与作图写进 manifest。

## 1. 两条流水线（并行）

两条都是 `scripts/exp18/` 下的驱动脚本（与 `$EXP/drivers/` 里实际运行的副本只差一行：仓库版的 `SRC` 缺省取脚本所在检出，
运行副本要求显式传 `SRC`）：

```bash
ssh -n -f finn_cci_c500 'bash -lc "cd $EXP && SRC=$SRC setsid nohup bash $SRC/scripts/exp18/run_gpu_pipeline.sh > logs/gpu_pipeline.log 2>&1 < /dev/null &"'
ssh -n -f finn_cci_c500 'bash -lc "cd $EXP && SRC=$SRC setsid nohup bash $SRC/scripts/exp18/run_cpu_pipeline.sh > logs/cpu_pipeline.log 2>&1 < /dev/null &"'
```

| 流水线 | 步骤 | 工具 | 实测耗时（2026-09-24） |
|---|---|---|---|
| CPU | 选集 → 渲染 C/D/E（8 个 Xvfb + llvmpipe worker）→ finalize 出清单 | `render/select_episodes.py`、`render/run_render.sh`、`render/finalize_clip_lists.py` | C 9 min、D 4.5 min、E 5 min |
| CPU | 俯视图（每场景每层一张正射 RGB + navmesh 掩码 + 房间多边形） | `topdown/run_topdown.sh` | 67 个场景 7 min |
| GPU | A/B 选 clip → 导出 B（858 clip）→ 导出 A（220 clip） | `select_clips_ab.py`、`run_dump.sh` | B 30 min、A 10 min |
| GPU | 等 C/D/E 清单 → 建 VO 缓存 → 导出 | `run_amb3r_cache.sh`、`run_dump.sh` | 约 1 s/帧（C 16357 帧、D 6231、E 5959） |

关键不变量：

- 选集规则全部写死在 `render/select_episodes.py` 与 `select_clips_ab.py`，与台账预注册一一对应；选集清单与每条规则的 sha1 键
  记在 `$RENDER/configs/selection_manifest.json`，每层保留/丢弃明细在 `$EXP/clip_lists/<T>.txt` 的表头与 `<T>_finalize_report.json`。
- finalize 在"选中集缺 clip""没按路线走完"时拒绝出清单（退出码 2），不会让下一名悄悄顶上；任何覆盖参数都要记进台账运行记录。
- 导出**只跑一次**、冻结（MACA 前向不确定，重跑会翻转打平的 logit）。

## 2. 指标与案例

导出全部完成后（`logs/gpu_pipeline.log` 出现 `DONE`）：

```bash
ssh finn_cci_c500 'bash -lc "cd $SRC && PYTHONDONTWRITEBYTECODE=1 $R/envs/qwen25/bin/python -m scripts.exp18.compute_metrics --self-check"'
ssh finn_cci_c500 'bash -lc "cd $SRC && PYTHONDONTWRITEBYTECODE=1 $R/envs/qwen25/bin/python -m scripts.exp18.select_cases"'
```

产物：`$EXP/metrics/{metrics.json, summary.md, slots.parquet, rows.parquet, episodes.parquet, cases.json}`。
`--self-check` 用 `scripts/training/validate.py` 的 `_HeatmapJointMetricAccumulator` 逐层逐臂复算 joint PCK@8，要求逐位相同。
判定（H1/H2/H3）直接写在 `metrics.json` 的 `verdicts` 里，阈值与台账逐字对应。

## 3. 作图

```bash
ssh finn_cci_c500 'bash -lc "cd $SRC && EXP18_ROOT=$EXP bash scripts/exp18/run_figures.sh"'
```

输出 `$EXP/figures/`（中英两版、PDF + 400 dpi PNG + caption），`manifest.json` 记录每张图的来源与 sha256。

**作图口径（用户 2026-09-24 定）**：图里不出现位姿、VO、里程计或位姿臂对比，热力统一叫 affordance map，预测一律画部署设置的输出
（导出里的 `vo` 臂）。GT 位姿臂只用于台账里的 H2 归因。
