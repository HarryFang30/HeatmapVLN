#!/usr/bin/env python3
"""
持续监控指定 GPU：当某张卡的利用率与显存占用同时低于阈值，且该状态连续超过设定时长时，
通过 configs/train_config.yaml 中的飞书 Webhook 发送提醒（注明卡号）。

用法示例:
  python scripts/monitor_gpu_idle.py
  python scripts/monitor_gpu_idle.py --config configs/train_config.yaml --util-max 5 --mem-max-mib 1024
  python scripts/monitor_gpu_idle.py --gpus 0,1 --duration-sec 60 --interval-sec 5

未指定 --gpus 时，默认监控配置文件里 gpu.devices；若仍无，则监控本机所有可见 GPU。
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    print("需要 PyYAML: pip install PyYAML", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("monitor_gpu_idle")


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_webhook_url(cfg: dict) -> str:
    notify = (cfg.get("log") or {}).get("notify") or {}
    url = (notify.get("webhook_url") or "").strip()
    if not url:
        raise ValueError("配置中 log.notify.webhook_url 为空，无法发送飞书消息")
    return url


def get_default_gpu_indices(cfg: dict) -> Optional[List[int]]:
    gpu = cfg.get("gpu") or {}
    devices = gpu.get("devices")
    if isinstance(devices, list) and devices:
        return [int(x) for x in devices]
    return None


def query_nvidia_smi() -> List[Tuple[int, float, float, float]]:
    """
    返回 [(index, util_percent, mem_used_mib, mem_total_mib), ...]
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE, timeout=30)
    except FileNotFoundError:
        logger.error("未找到 nvidia-smi，请确认已安装 NVIDIA 驱动")
        sys.exit(2)
    except subprocess.CalledProcessError as e:
        logger.error("nvidia-smi 执行失败: %s", e.stderr or e)
        sys.exit(2)

    rows: List[Tuple[int, float, float, float]] = []
    for line in out.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            util = float(parts[1])
            mem_used = float(parts[2])
            mem_total = float(parts[3])
        except ValueError:
            continue
        rows.append((idx, util, mem_used, mem_total))
    return rows


def send_feishu_text(webhook_url: str, text: str) -> bool:
    body = {"msg_type": "text", "content": {"text": text}}
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))
            if result.get("code") == 0 or result.get("StatusCode") == 0:
                return True
            logger.warning("飞书返回异常: %s", result)
            return False
    except urllib.error.URLError as e:
        logger.warning("发送飞书失败: %s", e)
        return False


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_cfg = repo_root / "configs" / "train_config.yaml"

    parser = argparse.ArgumentParser(description="GPU 空闲监控 + 飞书通知")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help="YAML 配置路径（读取 log.notify.webhook_url 与 gpu.devices）",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help="要监控的 GPU 编号，逗号分隔，如 0,1；留空则使用配置文件 gpu.devices 或全部卡",
    )
    parser.add_argument(
        "--util-max",
        type=float,
        default=10.0,
        help="利用率低于该值（%%）视为空闲，默认 10",
    )
    parser.add_argument(
        "--mem-max-mib",
        type=float,
        default=512.0,
        help="显存已占用低于该值（MiB）视为空闲，默认 512",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=60.0,
        help="空闲状态需连续保持的秒数后才发通知，默认 60",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=5.0,
        help="采样间隔（秒），默认 5",
    )
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    if not cfg_path.is_file():
        logger.error("配置文件不存在: %s", cfg_path)
        sys.exit(1)

    cfg = load_config(cfg_path)
    webhook_url = get_webhook_url(cfg)

    if args.gpus.strip():
        watch = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    else:
        watch = get_default_gpu_indices(cfg)
        if watch is None:
            watch = []  # 表示全部

    util_max = args.util_max
    mem_max = args.mem_max_mib
    need_sec = args.duration_sec
    interval = max(1.0, args.interval_sec)

    # gpu_index -> (idle_since_monotonic or None, already_notified_this_streak)
    state: Dict[int, Tuple[Optional[float], bool]] = {}

    hostname = socket.gethostname()
    logger.info(
        "开始监控 | 主机=%s | util<=%.1f%% mem_used<=%.0fMiB 持续>=%.0fs | 间隔=%.1fs | 配置=%s",
        hostname,
        util_max,
        mem_max,
        need_sec,
        interval,
        cfg_path,
    )
    if watch:
        logger.info("监控 GPU 索引: %s", watch)
    else:
        logger.info("监控本机全部可见 GPU")

    while True:
        stats = {r[0]: r for r in query_nvidia_smi()}
        if watch:
            indices = watch
        else:
            indices = sorted(stats.keys())

        now = time.monotonic()
        to_notify: List[Tuple[int, float, float]] = []

        for idx in indices:
            if idx not in stats:
                logger.warning("未找到 GPU %s 的 nvidia-smi 数据，跳过", idx)
                continue
            _, util, mem_used, mem_total = stats[idx]
            is_idle = util <= util_max and mem_used <= mem_max

            prev = state.get(idx, (None, False))
            idle_since, notified = prev

            if not is_idle:
                state[idx] = (None, False)
                continue

            if idle_since is None:
                state[idx] = (now, False)
                continue

            elapsed = now - idle_since
            if elapsed >= need_sec and not notified:
                to_notify.append((idx, util, mem_used))
                state[idx] = (idle_since, True)

        if to_notify:
            lines = [
                f"【GPU 空闲告警】主机 {hostname}",
                f"条件: 利用率≤{util_max:.1f}% 且 显存占用≤{mem_max:.0f}MiB，已连续超过 {need_sec:.0f} 秒。",
                f"空闲 GPU 编号: {', '.join(str(i) for i, _, _ in sorted(to_notify))}",
                "",
            ]
            for idx, util, mem_used in sorted(to_notify):
                total = stats[idx][3]
                lines.append(
                    f"GPU {idx}: 利用率 {util:.1f}%, 显存 {mem_used:.0f}/{total:.0f} MiB"
                )
            lines.append("")
            lines.append(f"配置: {cfg_path}")
            text_plain = "\n".join(lines)
            if send_feishu_text(webhook_url, text_plain):
                logger.info("已发送飞书通知: %s", [i for i, _, _ in to_notify])
            else:
                # 发送失败则允许下次重试
                for idx, _, _ in to_notify:
                    s = state.get(idx)
                    if s:
                        state[idx] = (s[0], False)

        time.sleep(interval)


if __name__ == "__main__":
    main()
