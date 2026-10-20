"""Audit selected multi-suite samples, keeping offline trajectories distinct from simulator resets."""
from pathlib import Path
from .common import digest, read_json
from .config import require


def audit_manifests(config):
    aliases = {"training": {"training", "peft_train", "student_state_train"},
               "calibration": {"calibration", "quant_calibration"}}
    identities = {"training": set(), "calibration": set()}
    contents = {"training": set(), "calibration": set()}
    evaluation = {(s["name"], "simulator_reset", task, reset)
                  for s in config["evaluation"]["suites"] for task in s.get("task_ids", range(10))
                  for reset in range(s["reset_offset"], s["reset_offset"] + s["resets_per_task"])}
    for role in identities:
        for record in config.get("data", {}).get(role, []):
            manifest_path = Path(record["manifest"])
            manifest = read_json(manifest_path)
            samples = manifest.get("samples", [])
            require(samples, "Empty selected data manifest")
            for sample in samples:
                actual_role = sample.get("role", sample.get("trajectory_role", manifest.get("role")))
                require(actual_role in aliases[role], "Forbidden/unverified sample role; do not relabel holdout")
                require(sample.get("suite", record["suite"]) == record["suite"], "Sample suite differs from selected suite")
                namespace = record["namespace"]
                if namespace == "simulator_reset":
                    key = (record["suite"], namespace, sample["task_id"], sample["init_state_index"])
                    require(key[2] in record["task_ids"] and key[3] in record["resets"], "Manifest exceeds declared reset scope")
                else:
                    identity = sample.get("trajectory_id", sample.get("dataset_order_index", sample.get("episode")))
                    require(identity is not None, "Offline trajectory identity required")
                    key = (record["suite"], namespace, str(identity))
                require(key not in evaluation, "Actual sample overlaps evaluation")
                identities[role].add(key)
                expected = sample.get("sample_sha256", sample.get("sha256"))
                name = sample.get("file", sample.get("name"))
                require(name and expected, "Sample path and content SHA required")
                path = (manifest_path.parent / name).resolve()
                require(manifest_path.parent.resolve() in path.parents, "Sample escaped manifest root")
                require(digest(path) == expected, "Actual sample content SHA mismatch")
                fingerprint = sample.get("observation_sha256", expected)
                require(fingerprint not in contents[role], "Duplicate observations inflate selected sample budget")
                contents[role].add(fingerprint)
    require(not identities["training"] & identities["calibration"], "Training/calibration trajectory overlap")
    require(not contents["training"] & contents["calibration"], "Training/calibration content overlap")
    return {role: {"trajectories": len(identities[role]), "unique_contents": len(contents[role])} for role in identities}
