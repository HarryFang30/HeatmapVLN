# EXP-03 结果：部署头也是位姿投影器，外观只买到可见性

判据见台账 [EXP-03](README.md#exp-03-部署头本身是否也只看位姿)（2026-09-04 预注册，开跑前 commit）。
本次为**纯评测、无训练**（`evaluation_only: true`）：加载部署头
`model/exp03_deployment_head/head_v2_best.pth`（`initial_head_hash=e00d5058…`），
在 R2R v2 的 **val 划分（4 个场景 / 858 clips，与 22 个训练场景不相交）** 上跑 7 个条件 ×
400 个样本，位姿全部来自 `data/amb3r_endpoint_v3_full_r2r` 的 **AMB3R VO 缓存**。
开发机单卡（GPU 7），2026-09-05 13:25 → 14:17，约 51 分钟。

## 1. 七个条件

指标口径与 EXP-02 一致：主指标 `joint_pck8`，可见性看 AUPRC / F1。

| 干预 | joint_pck4 | **joint_pck8** | Δpck8 | 恢复率 | vis AUPRC | vis F1 | 像素误差中位 |
|---|---|---|---|---|---|---|---|
| 无干预（基准） | 0.8347 | **0.9308** | — | 1.000 | 0.9957 | 0.9561 | 1.414 |
| **history-shuffle**（历史图像整体倒序） | 0.8285 | **0.9216** | **−0.92pt** | 0.990 | 0.9956 | 0.9555 | 1.414 |
| current-shuffle（换成别的样本的当前帧） | 0.8334 | 0.9272 | −0.36pt | 0.996 | 0.9959 | 0.9557 | 1.414 |
| blank-images（图像全黑） | 0.7186 | 0.8537 | −7.71pt | 0.917 | **0.9598** | **0.8015** | 2.828 |
| **zero-pose**（位姿换成常量） | 0.3316 | **0.5100** | **−42.08pt** | 0.548 | 0.7814 | 0.8421 | 8.000 |
| **pose-conflict**（位姿整体错位一格） | 0.5651 | **0.7665** | −16.43pt | 0.823 | 0.9191 | 0.8892 | 3.162 |
| **pose-conflict-shifted-target**（位姿与标签同步错位） | 0.8327 | **0.9259** | −0.49pt | **0.995** | 0.9955 | 0.9573 | 1.414 |

（400 个样本、3049 个可见历史槽位、`num_history=8`、seed 42。）

## 2. 判定：✅ 支持假设

| 判据 | 支持线 | 否定线 | 实测 | 判定 |
|---|---|---|---|---|
| `pose-conflict-shifted-target` 恢复率 | ≥0.95 | <0.85 | **0.9947** | ✅ |
| `history-shuffle` 相对基准变化 | \|Δ\| ≤ 2pt | 掉 > 5pt | **−0.92pt** | ✅ |

两项都过 ⇒ **支持**。按预注册，"§3.3 保守口径确立，可以写进论文"。

**假设本身是"部署头与探针行为一致"，它成立。** 走完 Stage1/Stage2 的部署权重没有凭空长出
视觉定位能力——这与 EXP-02 在探针上的结论一致，但两者是**不同的问题**：
EXP-02 回答"可识别性"（这个头**能不能**从视觉定位），EXP-03 回答"归因"
（部署的这个头**实际上**靠什么定位）。现在两问都有答案。

## 3. 最干净的两行仍然是错位对

把位姿整体错位一格，`joint_pck8` 从 0.9308 崩到 **0.7665**；
**把标签也同步错位一格，立刻回到 0.9259（基准的 99.5%）**。

也就是说：**某个槽位的输出只取决于送进该槽位的位姿，与该槽位放的是哪张历史图像无关。**
history-shuffle 只掉 0.92pt 是同一结论的另一面——把八张历史图整体倒序，
输出几乎不动，因为位姿没动。

## 4. 外观唯一买到的东西：可见性

blank-images 是唯一能把外观的贡献单独拎出来的干预（图像全黑、位姿保留）：

| 量 | 基准 | blank-images | 变化 |
|---|---|---|---|
| joint_pck8（定位） | 0.9308 | 0.8537 | −7.71pt |
| 像素误差中位 | 1.414 | 2.828 | ×2（1 格 → 2 格）|
| **visibility F1** | 0.9561 | **0.8015** | **−15.5pt** |
| **visibility AUPRC** | 0.9957 | **0.9598** | **−3.6pt** |
| visible_view_accuracy | 0.9718 | 0.9114 | −6.0pt |

**可见性的相对损失是定位的两倍**。这与 EXP-02 的第四节结论方向一致：
外观在这个头里稳定贡献的是"这个历史点现在还看不看得见"，不是"它在哪"。

## 5. 一个必须自己解释掉的数：zero-pose 还剩 0.5100

位姿被清成常量之后 `joint_pck8` 仍有 **0.5100**，看上去像"没有位姿也能定位一半"。
它**不是**视觉定位的证据，理由在同一张表里：

- **history-shuffle 只掉 0.92pt**——图像被打乱到完全错配，输出几乎不动；
- **blank-images 只掉 7.71pt**（定位口径）——图像整个拿掉，定位仍有 0.8537。

图像换掉、打乱、拿掉都动不了多少，说明剩下的 0.5100 不可能来自图像。
更合理的解释是**位置先验**：历史路点的标签分布本身极度偏斜（同一部署 checkpoint 在
512 对验证上的逐视角样本数是 back 82.27% / front 0.27%，见 EXP-10 条目），
一个"历史大致在我身后某个典型距离"的常数猜测本来就能拿到可观的 PCK@8。

**边界**：本次没有测这个先验基线（没有跑"位姿清零 + 图像清零"的双清条件），
所以"0.5100 来自位置先验"是**解释，不是测量**。要坐实它需要一个额外条件，
本次没做，也不写进论文。

## 6. 边界

沿用预注册的一条，新增两条：

① **结论只对"部署头 + VO 位姿 + R2R 分布"成立。** 它与 EXP-02（从零探针、真值位姿、
random-walk 数据）**不样本匹配**——两张表各自读，**绝对值不能交叉比较**。
本报告里出现的所有数只与本表内部比较；EXP-02 的 0.882 与本次的 0.9308 之间没有可比性。

② **0.5100 的来源未测**（§5）。"位置先验"是解释性说法，缺一个双清对照。

③ **单种子。** 本次只跑 seed 42。EXP-02 的结论是两个种子都支持才写下的；
本次是评测（不是训练），种子只影响 400 个样本的抽样，但**"评测抽样噪声"本身没有被量化**。
判据的两个数分别以 4.5pt 和 1.1pt 的余量过线，抽样噪声不太可能翻转它们，但这没有被测量。

## 7. 复现与产物

```bash
R=/mnt/afs/liwenhao/agent/370910109
OUT=$R/model/exp03_deployment_head/probe_full_v2

export HEATMAP_DATA_ROOT=$R/r2r_panoramic_data_v2/train
export SINGLE_VIEW_HM_OUT_DIR=$OUT
export SINGLE_VIEW_HM_TB_DIR=$OUT/tensorboard
export INTERNNAV_MODEL_PATH=$R/InternNav-Model
export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1

cd $R/HeatmapVLN
$R/envs/qwen25/bin/python scripts/tools/diagnose_heatmap_shortcuts.py \
  --mode full --architecture internnav_single_view \
  --config configs/train_heatmap_internnav_single_view_8gpu.yaml \
  --data-root $R/r2r_panoramic_data_v2/train \
  --amb3r-pose-cache-root $R/data/amb3r_endpoint_v3_full_r2r \
  --head-checkpoint $R/model/exp03_deployment_head/head_v2_best.pth \
  --output-dir $OUT --device cuda:0 --num-history 8 --val-samples 400 --seed 42
```

| 产物 | 路径 |
|---|---|
| 七条件报告 | `model/exp03_deployment_head/probe_full_v2/full/report.json` |
| 运行日志 | `model/exp03_deployment_head/probe_full_v2/run.log` |
| 输入的 head-only 权重 | `model/exp03_deployment_head/head_v2_best.pth` |

两处与配置有关的注意事项：

- `--config` **必须**是 `configs/train_heatmap_internnav_single_view_8gpu.yaml`——
  部署头就是按这份 config 的结构抽出来的（见 `head_v2_best.pth` 的 `provenance`），
  换一份会键名对不上。
- 该 config 有 4 个 `$VAR` 占位（`HEATMAP_DATA_ROOT` / `SINGLE_VIEW_HM_OUT_DIR` /
  `SINGLE_VIEW_HM_TB_DIR` / `INTERNNAV_MODEL_PATH`），漏导会在加载时才炸（§5 第 11 条）。
- eval-only 模式下脚本仍会先构建一遍 **train** 数据集（22 场景 / 4142 clips，约 2 分钟）
  再构建 val（4 场景 / 858 clips）。**评测只用 val**，日志开头的 `split=train` 不是错。
