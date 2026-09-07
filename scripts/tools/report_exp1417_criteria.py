"""Read the EXP-14 / EXP-17 decision files out against their pre-registered criteria.

The thresholds below are transcribed from docs/experiments/README.md, which fixed
them before any of these arms produced a number.  The point of running the
read-out through code is that the verdict is mechanical: no cell is decided by
which number happened to look good.

Usage:
  report_exp1417_criteria.py --exp14a-firsttoken A.json --exp14b-firsttoken B.json \
      --exp17b-gen C.json [--exp17a-gen D.json] [--exp14b-gen E.json] [--exp14a-gen F.json] \
      [--exp17b-noise G.json] [--exp17a-noise H.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SUPPORT, NEGATE, UNMEASURED = "✅ 支持", "❌ 否定", "⚠️ 没测出来"


def _f(value: Any) -> str:
    if value is None:
        return "  n/a "
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _pt(value: float | None) -> str:
    return "  n/a " if value is None else f"{value * 100:+.2f}pt"


def _load(path: Path | None) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path else None


def _row(rows: list, name: str, value: Any, rule: str, verdict: str) -> None:
    rows.append((name, _f(value) if not isinstance(value, str) else value, rule, verdict))


def _print(title: str, rows: list) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    width = max(len(r[0]) for r in rows) if rows else 10
    for name, value, rule, verdict in rows:
        print(f"{name:<{width}}  {value:>12}   {rule:<46} {verdict}")


def exp14(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Stop and turn are judged separately and never traded off (ledger EXP-14)."""
    rows: list = []
    recall, false_alarm = a["stop_recall"], a["stop_false_alarm"]
    preservation, turn_a = a["normal_preservation"], a["recovery_turn_accuracy"]
    turn_b = b["recovery_turn_accuracy"]
    normal_fa = (a.get("stop_false_alarm_by_source") or {}).get("dagger_normal")

    if recall < 0.20 or false_alarm > 0.05:
        stop_verdict = NEGATE
    elif recall >= 0.50 and false_alarm <= 0.02 and preservation >= 0.90:
        stop_verdict = SUPPORT
    else:
        stop_verdict = UNMEASURED

    delta_turn = turn_a - turn_b
    if delta_turn <= 0.02 or a["normal_preservation"] < 0.75:
        turn_verdict = NEGATE
    elif delta_turn >= 0.10 and turn_a >= 0.50 and preservation >= 0.90:
        turn_verdict = SUPPORT
    else:
        turn_verdict = UNMEASURED

    _row(rows, "stop_recall (memory)", recall, "支持 ≥ 0.50 / 否定 < 0.20", "")
    _row(rows, "stop_false_alarm (memory)", false_alarm, "支持 ≤ 0.02 / 否定 > 0.05", "")
    _row(rows, "normal_preservation (memory)", preservation, "支持 ≥ 0.90 / 否定 < 0.75", "")
    _row(rows, "→ 停的判定", "", "三条同时成立才支持", stop_verdict)
    _row(rows, "recovery_turn_accuracy (memory)", turn_a, "支持臂内 ≥ 0.50", "")
    _row(rows, "recovery_turn_accuracy (constant)", turn_b, "对照", "")
    _row(rows, "Δ_turn = memory − constant", _pt(delta_turn), "支持 ≥ +10pt / 否定 ≤ +2pt", "")
    _row(rows, "→ 转向的判定（13-B 判据）", "", "三条同时成立才支持", turn_verdict)
    _row(rows, "stop_false_alarm · dagger_normal", normal_fa, "14-C 准入 ≤ 0.002",
         "" if normal_fa is None else ("✅ 过" if normal_fa <= 0.002 else "❌ 不过"))
    _print("EXP-14 判据（首 token 读数，memory 臂为治疗臂）", rows)

    gate = {(SUPPORT, SUPPORT): "跑 14-C，两类决定一起测",
            (SUPPORT, UNMEASURED): "跑 14-C，论文只写恢复",
            (SUPPORT, NEGATE): "跑 14-C，论文只写恢复"}
    if turn_verdict == SUPPORT and stop_verdict == SUPPORT:
        decision = "跑 14-C，两类决定一起测"
    elif turn_verdict == SUPPORT:
        decision = "跑 14-C，论文只写恢复；停作为发现保留"
    elif stop_verdict == SUPPORT:
        decision = "跑 14-C，论文只写停"
    else:
        decision = "停。记为「决策层微调在两类决定上都未证实」"
    print(f"\n总门：转向 {turn_verdict} × 停 {stop_verdict} → {decision}")
    return decision


def exp17(b17: dict, a17: dict | None, b14: dict | None, a14: dict | None,
          noise_b: dict | None, noise_a: dict | None) -> None:
    rows: list = []
    nat = b17["passes"]["natural"]
    prefix = nat.get("prefix") or {}
    view_macro = prefix.get("slot_view_macro_acc")
    progress_macro = prefix.get("progress_macro_acc")

    v1a = SUPPORT if (view_macro or 0) >= 0.90 else NEGATE
    _row(rows, "1a 方位宏准确率 (C3 自然)", view_macro, "支持 ≥ 0.90 / 否定 < 0.90", v1a)

    v1c = SUPPORT if (progress_macro or 0) >= 0.62 else (NEGATE if (progress_macro or 0) <= 0.52 else UNMEASURED)
    _row(rows, "1c 进度宏准确率 (C3 自然)", progress_macro, "支持 ≥ 0.62 / 否定 ≤ 0.52", v1c)

    no_pose = (b17["passes"].get("no_pose") or {}).get("prefix") or {}
    view_nopose = no_pose.get("slot_view_macro_acc")
    v1d = SUPPORT if (view_nopose or 0) >= 0.45 else (NEGATE if (view_nopose or 0) <= 0.30 else UNMEASURED)
    _row(rows, "1d 不给位姿的方位宏准确率", view_nopose, "支持 ≥ 0.45 / 否定 ≤ 0.30", v1d)

    ph = b17["passes"].get("placeholder")
    if ph:
        d_stop = nat["stop_recall"] - ph["stop_recall"]
        d_turn = nat["recovery_turn_accuracy"] - ph["recovery_turn_accuracy"]
        best = max(d_stop, d_turn)
        v2a = SUPPORT if best >= 0.05 else (f"{NEGATE}（记为装饰）" if (d_stop <= 0.02 and d_turn <= 0.02) else UNMEASURED)
        _row(rows, "2a 自然−占位 · stop_recall", _pt(d_stop), "支持任一 ≥ +5pt", "")
        _row(rows, "2a 自然−占位 · turn_accuracy", _pt(d_turn), "两项都 ≤ +2pt → 装饰", v2a)

    assoc = b17.get("natural_association") or {}
    rel = assoc.get("stop_vs_progress (relevant)") or {}
    spec = assoc.get("stop_vs_slot_views (specificity)") or {}
    rd, lo = rel.get("risk_difference"), (rel.get("ci95") or [None, None])[0]
    rd_spec = spec.get("risk_difference")
    gap = None if (rd is None or rd_spec is None) else rd - rd_spec
    if rd is None:
        v2b = UNMEASURED
    elif rd <= 0.05:
        v2b = NEGATE
    elif rd >= 0.20 and lo is not None and lo > 0 and gap is not None and gap >= 0.10:
        v2b = SUPPORT
    else:
        v2b = UNMEASURED
    _row(rows, "2b 停 vs 进度 RD", _pt(rd), f"支持 ≥ +20pt 且 CI 下界 {_pt(lo)} > 0", "")
    _row(rows, "2b 停 vs 方位 RD（特异性）", _pt(rd_spec), "相关成分要高出它 ≥ 10pt", "")
    _row(rows, "2b 相关 − 特异性", _pt(gap), "支持 ≥ +10pt / 否定 RD ≤ +5pt", v2b)

    if b14:
        base = b14["passes"]["natural"]
        d_turn3a = nat["recovery_turn_accuracy"] - base["recovery_turn_accuracy"]
        v3a = SUPPORT if d_turn3a >= 0.05 else (NEGATE if d_turn3a <= 0.02 else UNMEASURED)
        _row(rows, "3a 转向 C3 − exp14b", _pt(d_turn3a), "支持 ≥ +5pt / 否定 ≤ +2pt（预期落否定）", v3a)
        d_stop3b = nat["stop_recall"] - base["stop_recall"]
        fa_ok = nat["stop_false_alarm"] <= 0.02
        v3b = (SUPPORT if (d_stop3b >= 0.05 and fa_ok) else (NEGATE if d_stop3b <= 0.02 else UNMEASURED))
        _row(rows, "3b 停 C3 − exp14b", _pt(d_stop3b), "支持 ≥ +5pt 且误报 ≤ 0.02", "")
        _row(rows, "3b C3 stop_false_alarm", nat["stop_false_alarm"], "≤ 0.02", v3b)

    pres, nonpix, fa_norm = nat["preservation_generated"], nat["nonpixel_on_normal"], nat["stop_false_alarm_normal"]
    v4 = SUPPORT if (pres >= 0.98 and nonpix <= 0.005 and fa_norm <= 0.002) else NEGATE
    _row(rows, "4 preservation_generated", pres, "≥ 0.98", "")
    _row(rows, "4 nonpixel_on_normal", nonpix, "≤ 0.005", "")
    _row(rows, "4 stop_false_alarm_normal", fa_norm, "≤ 0.002", v4)
    _print("EXP-17 判据（生成式读数，C3 = exp17b）", rows)

    extra: list = []
    if a17:
        na = a17["passes"]["natural"]
        _row(extra, "C3 − C1 · turn", _pt(nat["recovery_turn_accuracy"] - na["recovery_turn_accuracy"]), "报告，不作承重", "")
        _row(extra, "C3 − C1 · stop_recall", _pt(nat["stop_recall"] - na["stop_recall"]), "报告，不作承重", "")
    if a14 and b14:
        _row(extra, "exp14a − exp14b · stop_recall (生成)",
             _pt(a14["passes"]["natural"]["stop_recall"] - b14["passes"]["natural"]["stop_recall"]), "同一口径", "")
    for tag, clean, noisy in (("C3", b17, noise_b), ("C1", a17, noise_a)):
        if clean and noisy:
            c, n = clean["passes"]["natural"], noisy["passes"]["natural"]
            _row(extra, f"{tag} 位姿噪声 · stop_recall", _pt(n["stop_recall"] - c["stop_recall"]), "掉 ≥ 5pt 则先训 exp17c", "")
            _row(extra, f"{tag} 位姿噪声 · turn", _pt(n["recovery_turn_accuracy"] - c["recovery_turn_accuracy"]), "掉 ≥ 5pt 则先训 exp17c", "")
    if extra:
        _print("附带读数（无判据）", extra)


def main() -> int:
    p = argparse.ArgumentParser()
    for name in ("exp14a-firsttoken", "exp14b-firsttoken", "exp17b-gen", "exp17a-gen",
                 "exp14b-gen", "exp14a-gen", "exp17b-noise", "exp17a-noise"):
        p.add_argument(f"--{name}", type=Path, default=None)
    args = p.parse_args()
    a14ft, b14ft = _load(args.exp14a_firsttoken), _load(args.exp14b_firsttoken)
    if a14ft and b14ft:
        exp14(a14ft, b14ft)
    b17 = _load(args.exp17b_gen)
    if b17:
        exp17(b17, _load(args.exp17a_gen), _load(args.exp14b_gen), _load(args.exp14a_gen),
              _load(args.exp17b_noise), _load(args.exp17a_noise))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
