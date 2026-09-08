"""便宜检查先行：轨迹留出、重复性、无量化平滑；失败即停止后续批量实验。"""
import argparse
import json
from pathlib import Path
import sys


def check_split(directory, calibration_count, offset, count):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from qvla.baseline_contract import sample_identity
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rows = {s["file"]: s for s in manifest["samples"]}
    paths = sorted(directory.glob("sample-*.npz"))
    a, b = paths[:calibration_count], paths[offset:offset + count]
    if len(a) != calibration_count or len(b) != count or not a or not b:
        raise ValueError("样本数量不足")
    def trajectories(values):
        return {(rows[p.name]["suite"], rows[p.name]["episode"]) for p in values}
    if trajectories(a) & trajectories(b):
        raise ValueError("校准和探针存在同轨迹泄漏；不能只按帧 offset 留出")
    if set(sample_identity(a).values()) & set(sample_identity(b).values()):
        raise ValueError("不同文件名下存在重复样本")
    return {"calibration_frames": len(a), "calibration_trajectories": len(trajectories(a)),
            "heldout_frames": len(b), "heldout_trajectories": len(trajectories(b)),
            "scope": "独立校准/诊断轨迹；不是论文完整 512 轨迹校准的证明"}


def check_metrics(path, threshold):
    values = json.loads(path.read_text(encoding="utf-8"))
    if not values:
        raise ValueError("指标为空")
    errors = [x["normalized_action"]["mse"] for x in values]
    import math
    if any(not math.isfinite(v) or v > threshold for v in errors):
        raise ValueError(f"控制用例动作 MSE 超限：max={max(errors)} threshold={threshold}；先排错，不进行批量测试")
    return {"max_normalized_action_mse": max(errors), "threshold": threshold,
            "note": "工程排错阈值，不代表成功率或论文等价证明"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    split = sub.add_parser("split")
    split.add_argument("--samples", type=Path, required=True)
    split.add_argument("--calibration-count", type=int, default=8)
    split.add_argument("--offset", type=int, default=64)
    split.add_argument("--count", type=int, default=8)
    control = sub.add_parser("control")
    control.add_argument("--metrics", type=Path, required=True)
    control.add_argument("--threshold", type=float, required=True)
    args = p.parse_args()
    result = check_split(args.samples, args.calibration_count, args.offset, args.count) if args.mode == "split" else check_metrics(args.metrics, args.threshold)
    print(json.dumps({"status": "PASS", **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
