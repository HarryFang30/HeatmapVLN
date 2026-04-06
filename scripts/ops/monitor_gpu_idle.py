#!/usr/bin/env python3
"""
持续监控指定 GPU：当某张卡的利用率与显存占用同时低于阈值，且该状态连续超过设定时长时，
通过 configs/train_config_internnav.yaml 中的飞书 Webhook 发送提醒（注明卡号）。

用法示例:
  python scripts/ops/monitor_gpu_idle.py
  python scripts/ops/monitor_gpu_idle.py --util-max 5 --mem-max-ratio 0.05
  python scripts/ops/monitor_gpu_idle.py --gpus 0,1 --duration-sec 60 --interval-sec 5
  python scripts/ops/monitor_gpu_idle.py --ignore-compute-procs   # 仅看 util+显存比例（易误判训练中停顿）
  python scripts/ops/monitor_gpu_idle.py --no-auto-occupy
  python scripts/ops/monitor_gpu_idle.py --occupy-script /workspace/HeatmapVLN/scripts/train.py --occupy-args "--mem max --util 35"

未指定 --gpus 时，默认监控本机 nvidia-smi 可见的全部 GPU；仅 Webhook 从配置文件读取。

空闲判定（默认）: 利用率低 + 已用显存占总额比例低 + 无「训练类计算进程」占用足够显存。
compute-apps 仅统计进程名匹配白名单正则的条目（默认 python/torchrun/deepspeed 等），
系统或其它常驻进程占显存不计入，避免挡掉「狙击别人训练脚本」的判定。

飞书发送成功后，可选自动执行占卡脚本（默认 /workspace/HeatmapVLN/scripts/train.py，即 --gpu 逗号列表），
已在占用的 GPU 不会重复拉起进程。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# 默认：已用显存 / 总显存 ≤ 该比例视为「显存侧」空闲（如 0.10 = 10%）
DEFAULT_MEM_MAX_RATIO = 0.10
# 某 GPU 上「匹配训练类进程名」的 compute-app 显存合计超过该值（MiB）则认为有训练在占用
DEFAULT_COMPUTE_PROC_MIN_MIB = 128.0
# 仅将进程名匹配该正则的 compute-app 计入（nvidia-smi 的 process_name，不含路径）
DEFAULT_COMPUTE_PROC_NAME_REGEX = (
    r"(?i)(python|torchrun|deepspeed|accelerate|mpiexec|mpirun|horovod|pt_elastic|"
    r"pt_main_thread|ray::)"
)

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
OccupyChild = Tuple[subprocess.Popen, FrozenSet[int], Path]


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_webhook_url(cfg: dict) -> str:
    notify = (cfg.get("log") or {}).get("notify") or {}
    url = (notify.get("webhook_url") or "").strip()
    if not url:
        raise ValueError("配置中 log.notify.webhook_url 为空，无法发送飞书消息")
    return url


def query_nvidia_smi() -> Tuple[List[Tuple[int, float, float, float]], Dict[int, str]]:
    """
    返回:
      rows: [(index, util_percent, mem_used_mib, mem_total_mib), ...]
      index_to_uuid: 用于关联 compute-apps
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
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
    index_to_uuid: Dict[int, str] = {}
    for line in out.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
            uuid = parts[1]
            util = float(parts[2])
            mem_used = float(parts[3])
            mem_total = float(parts[4])
        except ValueError:
            continue
        rows.append((idx, util, mem_used, mem_total))
        index_to_uuid[idx] = uuid
    return rows, index_to_uuid


def query_compute_app_mem_mib_by_uuid_filtered(
    name_pattern: Optional[re.Pattern[str]],
) -> Dict[str, float]:
    """
    按 GPU UUID 汇总 used_gpu_memory（MiB）。
    name_pattern 为 None 时计入所有 compute-app；否则仅计入 process_name 能匹配 pattern 的条目。
    """
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if r.returncode != 0:
            return {}
        out = r.stdout or ""
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return {}
    sums: Dict[str, float] = {}
    for line in out.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        uuid, proc_name, mem_s = parts[0], parts[1], parts[2]
        if name_pattern is not None and not name_pattern.search(proc_name):
            continue
        try:
            mem = float(mem_s)
        except ValueError:
            continue
        sums[uuid] = sums.get(uuid, 0.0) + mem
    return sums


def compute_proc_mem_mib_by_index(
    index_to_uuid: Dict[int, str],
    name_pattern: Optional[re.Pattern[str]],
) -> Dict[int, float]:
    by_uuid = query_compute_app_mem_mib_by_uuid_filtered(name_pattern)
    by_index: Dict[int, float] = {}
    for idx, uuid in index_to_uuid.items():
        v = by_uuid.get(uuid, 0.0)
        if v > 0:
            by_index[idx] = v
    return by_index


def reap_occupy_processes(
    running: List[OccupyChild],
) -> Set[int]:
    """移除已结束的占卡子进程，并返回异常退出的 GPU。"""
    alive: List[OccupyChild] = []
    failed_gpus: Set[int] = set()
    for p, gpus, log_path in running:
        if p.poll() is None:
            alive.append((p, gpus, log_path))
        else:
            log_fn = logger.info if p.returncode == 0 else logger.warning
            log_fn(
                "占卡进程已结束 pid=%s gpus=%s code=%s log=%s",
                p.pid,
                sorted(gpus),
                p.returncode,
                log_path,
            )
            if p.returncode != 0:
                failed_gpus.update(gpus)
    running[:] = alive
    return failed_gpus


def gpus_covered_by_running(
    running: List[OccupyChild],
) -> Set[int]:
    out: Set[int] = set()
    for _, gpus, _ in running:
        out |= set(gpus)
    return out


def launch_occupy_script(
    script: Path,
    gpu_ids: List[int],
    extra_args: str,
    log_dir: Path,
) -> Optional[Tuple[subprocess.Popen, Path]]:
    """后台启动占卡脚本，返回 (Popen, 日志路径)；失败返回 None。"""
    gpu_csv = ",".join(str(i) for i in gpu_ids)
    cmd: List[str] = [sys.executable, str(script), "--gpu", gpu_csv]
    if extra_args.strip():
        cmd.extend(shlex.split(extra_args))
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"occupy_gpu_{'_'.join(str(i) for i in gpu_ids)}_{stamp}.log"
    try:
        # 占卡脚本按 nvidia-smi 的物理 GPU 编号传 --gpu；若继承父进程的
        # CUDA_VISIBLE_DEVICES，PyTorch 可见卡会重编号，导致与物理号不一致。
        child_env = os.environ.copy()
        child_env.pop("CUDA_VISIBLE_DEVICES", None)
        log_path.write_text(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] cmd={' '.join(shlex.quote(c) for c in cmd)}\n",
            encoding="utf-8",
        )
        # 新会话，避免监控进程信号误伤子进程；输出写入日志，便于排查占卡失败
        with open(log_path, "a", encoding="utf-8") as log_fp:
            p = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=child_env,
            )
        logger.info(
            "已启动占卡: pid=%s cmd=%s log=%s",
            p.pid,
            " ".join(shlex.quote(c) for c in cmd),
            log_path,
        )
        return p, log_path
    except OSError as e:
        logger.error("启动占卡脚本失败: %s", e)
        return None


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
    repo_root = Path(__file__).resolve().parent.parent.parent
    default_cfg = repo_root / "configs" / "train_config_internnav.yaml"

    parser = argparse.ArgumentParser(description="GPU 空闲监控 + 飞书通知")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help="YAML 配置路径（仅读取 log.notify.webhook_url）",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help="要监控的 GPU 编号，逗号分隔，如 0,1；留空则监控本机全部可见 GPU",
    )
    parser.add_argument(
        "--util-max",
        type=float,
        default=10.0,
        help="利用率低于该值（%%）视为空闲，默认 10",
    )
    parser.add_argument(
        "--mem-max-ratio",
        type=float,
        default=DEFAULT_MEM_MAX_RATIO,
        help="已用显存/总显存 ≤ 该值视为显存侧空闲，默认 0.10（即 10%%）",
    )
    parser.add_argument(
        "--ignore-compute-procs",
        action="store_true",
        help="不检查 nvidia-smi 计算进程（仅 util+显存比例；训练中 util 低时更易误判）",
    )
    parser.add_argument(
        "--count-all-compute-procs",
        action="store_true",
        help="compute-apps 不按进程名过滤（系统/驱动占显存也会计入，易与「只盯训练」冲突）",
    )
    parser.add_argument(
        "--compute-proc-name-regex",
        type=str,
        default=DEFAULT_COMPUTE_PROC_NAME_REGEX,
        help="仅统计进程名(process_name)匹配该正则的显存；默认匹配 python/torchrun/deepspeed 等",
    )
    parser.add_argument(
        "--compute-proc-min-mib",
        type=float,
        default=DEFAULT_COMPUTE_PROC_MIN_MIB,
        help="匹配上述正则的 compute-app 显存合计超过该值（MiB）则视为有训练占用，默认 128",
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
    parser.add_argument(
        "--auto-occupy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="飞书发送成功后是否启动占卡脚本（默认开启，可用 --no-auto-occupy 关闭）",
    )
    parser.add_argument(
        "--occupy-script",
        type=Path,
        default=repo_root / "scripts" / "train.py",
        help="占卡脚本路径（需支持 --gpu 0,1 这类参数），默认 <repo>/scripts/train.py",
    )
    parser.add_argument(
        "--occupy-args",
        type=str,
        default="",
        help='传给占卡脚本的额外参数（shell 分词），例如: --mem max --util 40',
    )
    parser.add_argument(
        "--occupy-log-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "monitor_gpu_idle",
        help="占卡脚本日志目录，默认 /tmp/monitor_gpu_idle",
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
        watch = []  # 空列表表示监控全部可见 GPU

    util_max = args.util_max
    mem_max_ratio = args.mem_max_ratio
    if not (0 < mem_max_ratio <= 1.0):
        logger.error("--mem-max-ratio 应在 (0,1] 内，例如 0.1 表示 10%%")
        sys.exit(1)
    check_compute_procs = not args.ignore_compute_procs
    compute_proc_min_mib = args.compute_proc_min_mib
    proc_name_pattern: Optional[re.Pattern[str]] = None
    if check_compute_procs:
        if args.count_all_compute_procs:
            proc_name_pattern = None
        else:
            try:
                proc_name_pattern = re.compile(args.compute_proc_name_regex)
            except re.error as e:
                logger.error("无效正则 --compute-proc-name-regex: %s", e)
                sys.exit(1)
    need_sec = args.duration_sec
    interval = max(1.0, args.interval_sec)

    # gpu_index -> (idle_since_monotonic or None, already_notified_this_streak)
    state: Dict[int, Tuple[Optional[float], bool]] = {}
    occupy_children: List[OccupyChild] = []

    occupy_script = args.occupy_script.resolve()
    auto_occupy = args.auto_occupy
    occupy_extra = args.occupy_args
    occupy_log_dir = args.occupy_log_dir.resolve()
    occupy_retry_delay_sec = max(interval * 2.0, 15.0)

    hostname = socket.gethostname()
    logger.info(
        "开始监控 | 主机=%s | util<=%.1f%% mem_used/total<=%.1f%% 持续>=%.0fs | 间隔=%.1fs | 计算进程=%s | 配置=%s",
        hostname,
        util_max,
        mem_max_ratio * 100.0,
        need_sec,
        interval,
        (
            "开启(训练类进程显存合计>%gMiB)"
            % compute_proc_min_mib
            if check_compute_procs and proc_name_pattern is not None
            else (
                "开启(全部进程显存合计>%gMiB)" % compute_proc_min_mib
                if check_compute_procs
                else "关闭"
            )
        ),
        cfg_path,
    )
    if check_compute_procs and proc_name_pattern is not None:
        logger.info("compute-app 进程名正则: %s", args.compute_proc_name_regex)
    if watch:
        logger.info("监控 GPU 索引: %s", watch)
    else:
        logger.info("监控本机全部可见 GPU")

    if auto_occupy:
        if occupy_script.is_file():
            logger.info("占卡脚本: %s（飞书成功后将启动空闲 GPU）", occupy_script)
            logger.info("占卡日志目录: %s", occupy_log_dir)
        else:
            logger.warning("占卡脚本不存在，将跳过自动占卡: %s", occupy_script)

    while True:
        failed_gpus = reap_occupy_processes(occupy_children)

        gpu_rows, index_to_uuid = query_nvidia_smi()
        stats = {r[0]: r for r in gpu_rows}
        if watch:
            indices = watch
        else:
            indices = sorted(stats.keys())

        proc_mem_by_idx: Dict[int, float] = {}
        if check_compute_procs:
            proc_mem_by_idx = compute_proc_mem_mib_by_index(index_to_uuid, proc_name_pattern)

        now = time.monotonic()
        if failed_gpus:
            for idx in sorted(failed_gpus):
                state[idx] = (now - need_sec + occupy_retry_delay_sec, False)
            logger.warning(
                "以下 GPU 的占卡进程异常退出，将在约 %.0f 秒后重试: %s",
                occupy_retry_delay_sec,
                sorted(failed_gpus),
            )
        to_notify: List[Tuple[int, float, float]] = []

        for idx in indices:
            if idx not in stats:
                logger.warning("未找到 GPU %s 的 nvidia-smi 数据，跳过", idx)
                continue
            _, util, mem_used, mem_total = stats[idx]
            mem_ratio = (mem_used / mem_total) if mem_total > 0 else 1.0
            mem_ok = mem_ratio <= mem_max_ratio
            util_ok = util <= util_max
            proc_ok = True
            if check_compute_procs:
                proc_sum = proc_mem_by_idx.get(idx, 0.0)
                proc_ok = proc_sum <= compute_proc_min_mib
            is_idle = util_ok and mem_ok and proc_ok

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
            if not check_compute_procs:
                cond_proc = "（未启用计算进程检测）"
            elif proc_name_pattern is None:
                cond_proc = f"且全部 compute-app 显存合计≤{compute_proc_min_mib:.0f}MiB"
            else:
                cond_proc = (
                    f"且训练相关进程显存合计≤{compute_proc_min_mib:.0f}MiB（系统/其它进程名不计）"
                )
            lines = [
                f"【GPU 空闲告警】主机 {hostname}",
                f"条件: 利用率≤{util_max:.1f}% 且 已用显存/总显存≤{mem_max_ratio * 100:.1f}% {cond_proc}，已连续超过 {need_sec:.0f} 秒。",
                f"空闲 GPU 编号: {', '.join(str(i) for i, _, _ in sorted(to_notify))}",
                "",
            ]
            for idx, util, mem_used in sorted(to_notify):
                total = stats[idx][3]
                pct = (100.0 * mem_used / total) if total > 0 else 0.0
                pm = proc_mem_by_idx.get(idx, 0.0) if check_compute_procs else 0.0
                if check_compute_procs and proc_name_pattern is not None:
                    extra = f", 训练类进程显存合计 {pm:.0f} MiB"
                elif check_compute_procs:
                    extra = f", 全部 compute-app 显存合计 {pm:.0f} MiB"
                else:
                    extra = ""
                lines.append(
                    f"GPU {idx}: 利用率 {util:.1f}%, 显存 {mem_used:.0f}/{total:.0f} MiB ({pct:.1f}%){extra}"
                )
            lines.append("")
            lines.append(f"配置: {cfg_path}")
            text_plain = "\n".join(lines)
            if send_feishu_text(webhook_url, text_plain):
                logger.info("已发送飞书通知: %s", [i for i, _, _ in to_notify])
                if auto_occupy and occupy_script.is_file():
                    failed_gpus = reap_occupy_processes(occupy_children)
                    if failed_gpus:
                        for idx in sorted(failed_gpus):
                            state[idx] = (now - need_sec + occupy_retry_delay_sec, False)
                        logger.warning(
                            "以下 GPU 的占卡进程异常退出，将在约 %.0f 秒后重试: %s",
                            occupy_retry_delay_sec,
                            sorted(failed_gpus),
                        )
                    want = {i for i, _, _ in to_notify}
                    busy = gpus_covered_by_running(occupy_children)
                    to_launch = sorted(want - busy)
                    if to_launch:
                        for gpu_id in to_launch:
                            launched = launch_occupy_script(
                                occupy_script,
                                [gpu_id],
                                occupy_extra,
                                occupy_log_dir,
                            )
                            if launched is not None:
                                p, log_path = launched
                                occupy_children.append((p, frozenset([gpu_id]), log_path))
                            else:
                                state[gpu_id] = (now - need_sec + occupy_retry_delay_sec, False)
                    if want <= busy:
                        logger.info(
                            "空闲 GPU 已在占卡进程中，跳过拉起: %s",
                            sorted(want),
                        )
            else:
                # 发送失败则允许下次重试
                for idx, _, _ in to_notify:
                    s = state.get(idx)
                    if s:
                        state[idx] = (s[0], False)

        time.sleep(interval)


if __name__ == "__main__":
    main()
