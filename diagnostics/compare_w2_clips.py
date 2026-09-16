"""Check exact state/seed pairing for W2 language-only clipped versus no-clip rollouts."""
import argparse
import json
import re
from pathlib import Path


def read_run(path):
    if (path / "exit-code.txt").read_text(encoding="utf-8").strip() != "0":
        raise ValueError(f"Evaluation failed: {path}")
    logs = list(path.glob("EVAL-*.txt"))
    if len(logs) != 1:
        raise ValueError("Expected exactly one evaluation log")
    text = logs[0].read_text(encoding="utf-8")
    if "Episode error:" in text:
        raise ValueError("Episode exception present")
    manifests = [json.loads(x) for x in re.findall(r"^EPISODE_MANIFEST (.+)$", text, re.M)]
    successes = [x == "True" for x in re.findall(r"^Success: (True|False)$", text, re.M)]
    if len(manifests) != 10 or len(successes) != 10:
        raise ValueError("Expected ten episodes")
    keys = [(x["task_id"], x["init_state_index"]) for x in manifests]
    if set(keys) != {(i, 0) for i in range(10)} or len(set(keys)) != 10:
        raise ValueError("Task/state mismatch or duplicate")
    line = re.findall(r"^\[official-quant\] candidate=(.+)$", (path / "console.log").read_text(encoding="utf-8"), re.M)
    if len(line) != 1:
        raise ValueError("Expected one candidate metadata line")
    candidate = json.loads(line[0])
    if candidate["scope"] != "language" or candidate["applied_targets"] != 224:
        raise ValueError("Unexpected quantization target scope")
    return {key: (manifest, success) for key, manifest, success in zip(keys, manifests, successes)}, candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clipped", type=Path, required=True)
    parser.add_argument("--no-clip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clipped, cmeta = read_run(args.clipped)
    no_clip, nmeta = read_run(args.no_clip)
    if any(clipped[k][0] != no_clip[k][0] for k in clipped):
        raise ValueError("A task has different initial-state hashes or seeds")
    if cmeta["name"] != "profile" or cmeta["removed_language_clips"] != 0:
        raise ValueError("Clipped profile metadata mismatch")
    if nmeta["name"] != "w2-no-clip-primary-g64" or nmeta["removed_language_clips"] != 160:
        raise ValueError("No-clip candidate metadata mismatch")
    if cmeta["profiles"][0]["sha256"] != nmeta["profiles"][0]["sha256"]:
        raise ValueError("The language W2 base profile differs")
    by_task = [{"task_id": task, "initial_state_sha256": clipped[(task, 0)][0]["init_state_sha256"],
                "clipped_success": clipped[(task, 0)][1], "no_clip_success": no_clip[(task, 0)][1]}
               for task in range(10)]
    result = {"paired_manifests_equal": True, "same_w2_base_profile": True,
              "language_target_count": 224, "removed_language_clips_in_no_clip": 160,
              "clipped_successes": sum(x["clipped_success"] for x in by_task),
              "no_clip_successes": sum(x["no_clip_success"] for x in by_task),
              "by_task": by_task,
              "limit": "Single state per task. A 0-vs-1 difference does not establish the preferable clipping policy; the vision G64 candidate is inactive in language-only scope."}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "paired_w2_clip_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("clipped_successes", "no_clip_successes", "paired_manifests_equal")}))


if __name__ == "__main__":
    main()
