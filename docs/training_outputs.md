# 训练产物目录说明

每次训练都会在 `log.out_dir` 下创建一个独立的 `run_时间戳/` 目录，并维护一个 `latest` 软链接指向最近一次运行。

目录结构如下：

```text
run_YYYYMMDD_HHMMSS/
  manifest/
    config.yaml
    args.json
    git.json
    env.json
    summary.json
  logs/
    train.log
    metrics.jsonl
  checkpoints/
    epoch_001.pth
    epoch_002.pth
    best.pth
    latest.pth
  visualizations/
    train/
    val/
  plots/
    curves.png
    history.json
  tensorboard/
```

各目录用途：

- `manifest/`: 固化本次实验的配置、命令行参数、git 状态、环境信息和最终摘要。
- `logs/train.log`: 训练主日志。
- `logs/metrics.jsonl`: 结构化指标流，适合后处理和实验对比。
- `checkpoints/`: epoch 检查点，以及 `best.pth` 和 `latest.pth`。
- `visualizations/train/` 与 `visualizations/val/`: 训练/验证阶段导出的热力图可视化。
- `plots/`: 训练曲线图和 epoch 级历史指标。
- `tensorboard/`: 当前 run 对应的 TensorBoard 事件文件。

兼容入口：

- 主入口仍然是当前 run 下的 `tensorboard/`
- 如果配置了 `log.tensorboard_dir`，会在该目录下维护 `latest`
- 同时会在 `/root/tf-logs/latest` 下保留一个根级入口，方便通过 autodl 端口实时监控

常用查看方式：

- 看最近一次运行：`log.out_dir/latest`
- 看文本日志：`logs/train.log`
- 看最佳模型：`checkpoints/best.pth`
- 看结构化指标：`logs/metrics.jsonl`
- 启动 TensorBoard：`tensorboard --logdir <run_dir>/tensorboard`
- autodl 端口监控：`tensorboard --logdir /root/tf-logs/latest`
