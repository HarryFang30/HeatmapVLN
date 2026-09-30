# 服务器迁移说明（2026-09）

把开发机工作区整体搬到新服务器 / 新存储时照这份做。背景约定（开发模式、login shell、safe.directory、测试基线）见
`CLAUDE.md`；上一次迁移（2026-08，`/mnt/afs/lixiaoou/intern/fjl` → 现根）的对照记录在 `CLAUDE.md` §6。

```
W_OLD=/mnt/afs/liwenhao/agent/370910109      # 现工作区根，约 717 GB（2026-09-28 统计）
W_NEW=<新根>                                 # 迁移后填
```

**原则：目录结构原样保留，只换根。** 这样仓库里的写死路径只需一次整体替换（§4.1），磁盘上各种 manifest / 缓存里记录的
相对关系也不会断。

---

## 1. 迁移前（在旧服务器上）

1. **确认没有任务在跑**：网站提交的集群任务会实时读共享检出与数据（台账 §5 第 27 条），开发机上用
   `ps aux | grep -E 'python|torchrun|collect|Xvfb'` 看（`/mnt/afs/lixiaoou/...` 下的僵尸进程不是我们的，台账 §5 第 18 条）。
2. **不在任何 git 仓库里的代码改动**（2026-09-28 状态）：

   | 仓库 | 位置 | 状态 |
   |---|---|---|
   | HeatmapVLN | `W/HeatmapVLN` | 服务器检出 `main` 164c67d、干净；所有实验分支都在 GitHub |
   | rpc（`vla_rpc`） | `W/rpc` | ✅ 已提交推送：HarryFang30/rpc `master` **b35e060**（新增 InferJSON） |
   | VLN-CE 采集器 | `W/habitat/VLN-CE` | ✅ 已提交推送：HarryFang30/VLN-CE `master` **3b0c5c0**（分片采集、MetaX 启动脚本、审计脚本） |
   | AMB3R | `W/amb3r` | ⚠️ **未推**：上游 HengyiWang/amb3r 92c4081 + 8 个文件的 C500 适配 + `memory_bounded_attention.py`。补丁在本地迁移包 `code_patches/amb3r_local.patch`；需要你自己 fork 后推一个分支（如 `metax-c500`） |
   | habitat-lab | `W/habitat/habitat-lab` | 上游 34a4042 + 1 行改动（`habitat_baselines/rl/requirements.txt`），补丁 `code_patches/habitat_lab_local.patch` |
   | habitat-sim | `W/habitat/habitat-sim` | 上游 ee75ba5 + 未跟踪的 `build_glx_vlnce/`（0.1.7 的 GLX 构建产物，见 §4.3） |
   | 认证复刻评测栈 | `W/evaluation_plans/internnav_native_r2r_val_unseen_8gpu_20260802/` | 不在 git 里，本地迁移包 `code_patches/eval_plan_*.tgz` 有整份 |
   | InternNav / CorrectNav / NaVid-VLN-CE | `W/<名>` | 公开仓库的干净克隆（InternNav 7a5c624、CorrectNav f719e7f、NaVid-VLN-CE ce2f804），重新 clone 即可 |

3. **`W/.ssh/`**（ssh 配置与密钥目录，本说明没有读取其内容）：自己单独搬，**不要**放进任何仓库、迁移包或共享目录。
4. **本地迁移包**（2026-09-27 下载到个人电脑 `~/HeatmapVLN_checkpoints/`，逐文件 sha256 校验过）：最终权重、7 个消融权重、
   上表所有补丁/代码包、两个环境的依赖清单（`envs/*_pip_freeze.txt`、`*_conda_explicit.txt`）。它是**保险**，正式迁移仍按 §2 搬服务器上的原件。

---

## 2. 目录清单与处置

四类：**A 只此一份**（丢了就没了）→ 必须搬、搬完校验；**B 能重算但贵** → 搬；**C 能重新下载** → 网络好就重新下载，
否则一起搬；**D 可丢**。

| 目录 | 大小 | 类 | 内容 / 处置 |
|---|---|---|---|
| `model/` | 24.6 GB | **A** | 全部实验 checkpoint、评测结果、台账 §4 引用的全部产物（含 EXP-18/19）。整目录搬，搬完用 §3 的 sha256 清单核对 |
| `data/` | 453 GB | **A/B** | 见 §2.1 细分 |
| `r2r_panoramic_data_v2/` | 102.4 GB | **B** | Stage1/2 训练数据（26 场景 5000 clip）。可用 VLN-CE 采集器重采（8 worker 约数小时），但训练与缓存都以它为准，**搬** |
| `teacher_sidecars/` | 28.5 GB | **B** | `stage2_native_dataset`：`scripts/evaluation/collect_internnav_teacher_sidecar.py` 在已采数据上跑 InternNav 得到的教师标签（旧的全景适配器 Stage2 线用，`run_stage2_pano_adapter_8gpu_mxc500_launcher.sh`）。重算要整轮 GPU 推理，搬 |
| `habitat/` | 59.6 GB | **C**（场景）+ **A**（补丁/构建） | `VLN-CE/data/scene_datasets/mp3d`（21 GB，90 场景）与 `hm3d`（28 GB，800 场景）受数据集许可约束，可按你们的许可重新下载；`VLN-CE/data/datasets`（278 MB，R2R / ScaleVLN episode）；`habitat-lab`、`habitat-sim`（含 `build_glx_vlnce/`）、`VLN-CE` 代码。**建议整目录搬**，省去重新编 habitat-sim 0.1.7 GLX |
| `envs/` | 15.8 GB | **B** | `qwen25`（Py3.12，MetaX torch 2.8）、`vlnce`（Py3.8，habitat 0.1.7）、`scalevln`、`maca-pytorch2.8-py312-3.3.0.2-x86_64.tar.xz`（2.6 GB，**MetaX 专用 torch/triton/flash-attn/xformers 的 wheel 包，必须搬**）。见 §4.3 |
| `InternNav-Model/` | 15.6 GB | **C** | InternVLA-N1 发布权重（慢系统 Qwen2.5-VL-7B + 快系统 NextDiT），全程冻结。可重新下载，或直接搬 |
| `amb3r/` | 10.3 GB | **A**（代码补丁）+ **C**（权重） | `checkpoints/DA3NESTED-GIANT-LARGE`（6.3 GB，**唯一被用到的 AMB3R 权重**）、`checkpoints/amb3r.pt`（3.9 GB，我们的代码不用）、`checkpoints/deps/utils3d-*.whl`；代码有本地补丁（§1 表）。整目录搬最省事 |
| `tools/` | 0.7 GB | **A** | `x11_headless_bundle_ubuntu22_20260801_v4`（Habitat 无头渲染必需，见 `docs/ops/README_HEADLESS_HABITAT_XVFB_XKB_LLVMPIPE.md`）；v1/v3 是旧版，可丢 |
| `evaluation_plans/` | < 0.1 GB | **A** | 认证复刻评测栈（native 62.5% golden 参照） |
| `HeatmapVLN_native_lock_bd5ead1/` | < 0.1 GB | **A** | native 锁定副本（不是 git 仓库），搬 |
| `runtime_config/`、`runtime_home/`、`staging/`、`training_plans/`、`r2r_collect_tools/` | < 0.1 GB | **A** | 小文件，整目录搬 |
| `evaluations/`、`evaluation/` | 1.8 GB / 0.5 GB | **A**（历史） | 7 月的早期评测输出，体积小，搬 |
| `HeatmapVLN/`、`InternNav/`、`CorrectNav/`、`NaVid-VLN-CE/`、`rpc/` | < 0.5 GB | **C** | 从 GitHub 重新 clone 到 §1 表里的 commit；HeatmapVLN 按 §4.1 改路径 |
| `tmp/`（2.9 GB）、`_codex_staging/`、`.codex_tmp/`、`.codex_backup/`、`.trash/`、`.pytest_cache/`、`cache/`、`tensorboard_sessions/`、`tensorlog/`、`.tos/`、`.cache/`、`.home/`、`.conda_home/` | ~4 GB | **D** | 临时与缓存，可丢（丢之前自己扫一眼 `tmp/`） |

### 2.1 `data/` 细分（453 GB）

| 子目录 | 大小 | 类 | 内容 / 处置 |
|---|---|---|---|
| `heatmap_randomwalk_train_v1/` | **420 GB** | **B** | 随机游走全景（61 场景 / 6000 clip，四视角都有深度）：历史头预训练（单视角 v1、位姿适配）与 EXP-02/04 探针的数据。可用 VLN-CE heatmap 采集器重采但很慢。**不打算从头重训历史头、也不重跑 EXP-02/04 时，可以只放冷存储**；这是整个迁移里唯一可以商量的大头 |
| `heatmap_system1_dagger_v1/` | 16.6 GB | **A** | DAgger 采集（10868 个 episode tar，EXP-12/13/14/16/17 的数据），重采要跑闭环策略，搬 |
| `exp18_renders/` | 11.7 GB | **B** | EXP-18 的 C/D/E 新渲染 + 它们的 VO 缓存（渲染 20 min 可重做，VO 缓存要 3 卡约 2.5 h），搬 |
| `amb3r_endpoint_v3_full_r2r/` | 0.1 GB | **A** | R2R v2 的 AMB3R VO 位姿缓存，Stage1/2 训练与所有 VO 臂评测都读它；重建要 8 卡约 11 h（且 GPU 间有已知的小差异），**必须搬** |
| `heatmap_randomwalk_amb3r_endpoint_cache_v2_4gpu/`、`heatmap_randomwalk_amb3r_causal_cache_v1/` | 0.1 GB | **A** | 随机游走数据的 VO 缓存，小，搬 |
| `candidate_support_audit_v1/`、`_v2/`、`candidate_continuation*` | 4.5 GB | **A**（历史） | 8 月候选支持审计的产物，搬 |
| `heatmap_system1_training_v1/`、`r2r_sft_balanced500_v1/`、`val_heatmap/` | 0.2 GB | **A** | 小，搬 |
| `*_smoke_*`、`*_preflight*`、`*dev_smoke*` | ~0 | **D** | 冒烟/预检产物，可丢 |

按"全搬"算约 717 GB；把 `heatmap_randomwalk_train_v1/` 放冷存储、丢掉 D 类后，新服务器上的热数据约 **290 GB**。

---

## 3. 传输与校验

- **同一集群 / 能挂载同一 AFS**：`rsync -aH --partial --info=progress2 W_OLD/<目录>/ W_NEW/<目录>/`，逐目录搬，断了重跑续传。
- **跨网络**：先搬 A 类，再 B 类；C 类（InternNav-Model、DA3、MP3D/HM3D）优先在新环境重新下载。本机到开发机的链路只有约 1 MB/s，
  不适合走个人电脑中转大文件。
- **校验 A 类**：搬之前在旧服务器上生成清单，搬完在新服务器上核对：

  ```bash
  cd $W_OLD && find model evaluation_plans HeatmapVLN_native_lock_bd5ead1 -type f -print0 | xargs -0 sha256sum > /tmp/sha_A.txt
  cd $W_NEW && sha256sum -c /tmp/sha_A.txt | grep -v ': OK$'     # 无输出 = 全部一致
  ```
  `data/` 与 `r2r_panoramic_data_v2/` 文件太多，用 `find … | wc -l` 与 `du -s` 对总数，再抽样核对 sha256。

---

## 4. 新服务器搭建

### 4.1 根目录与写死路径

仓库里 `configs/`、`run_*_mxc500.sh`、`scripts/`、`docs/` 写死了工作区根（2026-09-28 在 EXP-18 分支上统计：**117 个文件 / 353 处**；
`main` 上更少）。**在本地改、提交、推送，服务器只 pull**（`CLAUDE.md` §1）：

```bash
OLD=/mnt/afs/liwenhao/agent/370910109; NEW=<新根>
git grep -l "$OLD" | xargs sed -i '' "s#$OLD#$NEW#g"      # macOS sed；Linux 去掉 ''
git grep -n "$OLD"                                          # 应无输出
```

然后在 `CLAUDE.md` §6 追加一条新的迁移对照（旧 → 新）。注意：

- **`/opt/maca-3.3.0`**（30 个文件）与 **`/opt/conda`** 不在工作区下，新机器若装在别处要单独替换。
- **磁盘上的历史产物**（各 `manifest.json`、`*_report.json`、EXP-18 导出里的 `clip_dir`）记录的是旧绝对路径，**不要改**，它们是历史记录。
  需要重新读这些产物的工具都有覆盖参数（例如 EXP-18 作图的 `--clip-root`、`EXP18_ROOT`）；AMB3R 缓存按"相对数据根的路径"匹配，搬家不受影响。
- 旧旧根 `/mnt/afs/lixiaoou/intern/fjl` 仍出现在两个环境的 pip 元数据里（vlnce 的可编辑 habitat-lab、qwen25 的 MetaX wheel 来源路径）。
  这只是安装时记录的来源：2026-09-28 实测 `habitat` 从现根导入。新机器重建环境时用新路径。

### 4.2 平台

- 加速卡 **MetaX C500（MACA 3.3.0，`/opt/maca-3.3.0`）**，Ubuntu 22.04。换成别的卡（例如 NVIDIA）不是"迁移"而是"移植"：
  qwen25 的 torch/triton/flash-attn/xformers 都是 MetaX 构建，AMB3R 的补丁也是为"无 xformers"写的。
- 远程命令一律 login shell（`CLAUDE.md` §3.1，否则 `MACA_PATH` 为空）。
- Habitat 渲染走 CPU（Xvfb + Mesa llvmpipe），用 `tools/x11_headless_bundle_ubuntu22_20260801_v4`，不依赖 GPU 驱动。

### 4.3 Python 环境

**新根与旧根的相对布局相同、系统同为 Ubuntu 22.04 + MACA 3.3.0 时**，最省事的是整目录搬 `envs/`，然后修 conda 前缀：

```bash
# 在旧机器上打包（conda-pack 会改写前缀；装在 /opt/conda 下）
/opt/conda/bin/conda pack -p $W_OLD/envs/qwen25 -o qwen25.tar.gz
/opt/conda/bin/conda pack -p $W_OLD/envs/vlnce  -o vlnce.tar.gz
# 新机器：解到 $W_NEW/envs/<名>/ 后运行 bin/conda-unpack
```

（vlnce 里的 habitat-lab 是可编辑安装、habitat-sim 是 egg 安装，conda-pack 可能拒绝打包；那就直接 `rsync` 整个 env 目录，再把
`site-packages` 里的 `*.egg-link` / `easy-install.pth` 指向新的 `habitat/habitat-lab`。）

**要重建时**：
- `qwen25`：Python 3.12.13；先装 `envs/maca-pytorch2.8-py312-3.3.0.2-x86_64.tar.xz` 里的 MetaX wheel（torch 2.8.0+metax3.3.0.2、
  triton 3.0.0、flash_attn 2.6.3、xformers 0.0.22、apex、torchvision、torchaudio），再按 `qwen25_pip_freeze.txt` 装其余（transformers 4.51.0、numpy 1.26.4 等）。
- `vlnce`：Python 3.8.20；habitat-sim 0.1.7（GLX 构建，源码 `habitat/habitat-sim` ee75ba5、构建目录 `build_glx_vlnce/`），habitat-lab 0.1.7 可编辑安装
  （`habitat/habitat-lab` 34a4042 + 补丁），其余按 `vlnce_pip_freeze.txt`（numpy 1.23.5、cv2 4.13、PIL 10.4）。
- 依赖清单在本地迁移包 `~/HeatmapVLN_checkpoints/envs/`。

### 4.4 权重（最小集合）

| 用途 | 文件 |
|---|---|
| 最终方法（部署 / 主表 / EXP-18/19） | `model/output_past_plan_action_refine_v2_8gpu/run_20260829_115642/checkpoints/best.pth`（历史头 + 未来头 + 桥） |
| 冻结底座 | `InternNav-Model/`（慢系统 + 快系统，`INTERNNAV_MODEL_PATH`） |
| 部署里程计 | `amb3r/checkpoints/DA3NESTED-GIANT-LARGE/`（AMB3R 唯一加载的权重） |
| 消融重跑 | `ablation_stage3/{exp08,exp09a,exp09b,exp09c}`、v1 `best_deployment_full.pth`、Stage1/Stage2 `best.pth`（路径见台账 §4） |

`model/` 整目录搬了，这些自然都在。

---

## 5. 验收（新服务器上依次做，全过才算迁完）

1. **测试基线**：`CLAUDE.md` §4 的全量命令，`main` 上应为 **1004 passed / 1 skipped / 1 collection error**（唯一已知错误
   `test_stage3_dataloader_order.py`）；EXP-18 分支合入后多出 `tests/test_exp18_geometry.py`（125 个，纯 numpy）。
   只在新机器上挂的测试先怀疑平台/路径（`CLAUDE.md` §5.2），冷 AFS 下的已知抖动见 `CLAUDE.md` §4。
2. **导入与平台**：`envs/qwen25/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`；
   `envs/vlnce/bin/python -c "import habitat, habitat_sim; print(habitat.__file__)"`（路径应在新根下）。
3. **数值复现（CPU，几分钟）**：从新根重跑 EXP-18 指标，结果必须与已存的 `model/exp18_first_person_viz/metrics/metrics.json` 相同：
   `EXP18_ROOT=$W_NEW/model/exp18_first_person_viz EXP18_METRICS_DIR=/tmp/exp18_metrics_check python -m scripts.exp18.compute_metrics --self-check`，
   再比较 `/tmp/exp18_metrics_check/metrics.json` 与已存文件的 `tiers` / `verdicts`（自检要求与 `validate.py` 逐位一致）。
4. **GPU 前向（单卡，~10 min）**：用 `scripts/exp18/run_dump.sh` 对 B 层 3 个 clip 重新导出，与已存导出比较：标签、位姿逐位相同，
   预测差在 MACA 噪声内（gated ≤ ~2e-4，logit ≤ ~0.125）。
5. **AMB3R（单卡，~5 min）**：用 `scripts/exp18/run_amb3r_cache.sh` 对 `JmbYfDe2QKZ/clip_000427`、`clip_300453` 重建缓存，
   与 `data/amb3r_endpoint_v3_full_r2r` 比较：帧号一致，位姿中位差 mm 级（已知 clip_000427 的第 12 帧有 36° 的系统差，见 EXP-18 探针记录）。
6. **Habitat 渲染（CPU，~1 min）**：`TIER=C NUM_WORKERS=1` 跑 `scripts/exp18/render/run_render.sh` 的冒烟（输出到临时根），
   chunk 格式与 `r2r_panoramic_data_v2` 一致。
7. **闭环复现（网站提交，8 卡）**：按 `docs/experiments/exp07-seed1337-submission.md` 的提交物跑 v2 权重的 val_unseen 全量评测
   （`scripts/run_ppa_stage2_r2r_val_unseen_8gpu_mxc500.sh`，`PPA_EVAL_CHECKPOINT` / `PPA_EVAL_CONFIG` / `PPA_EVAL_OUTPUT_ROOT` /
   `PPA_EVAL_PROTOCOL_SEED`，路径换成新根），SR 应与台账 §4 基线一致（种子 42：62.81%；种子 1337：61.17%）。

---

## 6. 已知坑（迁移相关）

- 路径：见 §4.1；**不要在服务器上改文件或提交**，改路径在本地做。
- `git` 报 `dubious ownership`：每条 git 命令带 `-c safe.directory=<仓库>`（`CLAUDE.md` §3.2）。
- 过期 `.pyc` 会让 traceback 显示旧路径：迁完先清 `__pycache__`（`CLAUDE.md` §3.3）。
- AFS 冷缓存 / 小文件元数据慢时，Python 导入和 Xvfb 启动会慢到小时级：运行时缓存放本地 `/tmp`（EXP-19 的 `EXP19_RUNTIME_ROOT` 做法）。
- 远程 `pgrep -f <pattern>` 会匹配到发起它的那条 ssh 命令本身，等待循环里用 `| grep -v pgrep` 或 pid 文件。
