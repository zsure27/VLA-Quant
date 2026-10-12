"""Versioned paired/upstream command and actual trace audit; archived auditor unchanged."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qvla.extended_peft import sha
from qvla.on_policy_capture import observation_hash
from qvla.paired_metrics import comparison


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def normalized_trace_state(raw, stats):
    """Mirror the frozen OFT BOUNDS_Q99 proprio transform, then NPZ float32 storage."""
    value = np.asarray(raw)
    q01, q99 = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    mask = np.asarray(stats.get("mask", np.ones_like(q01, dtype=bool)), dtype=bool)
    if value.shape != (8,) or q01.shape != (8,) or q99.shape != (8,) or mask.shape != (8,) or not np.isfinite(value).all():
        raise ValueError("Malformed raw trace or frozen proprio statistics")
    return np.clip(np.where(mask, 2 * (value - q01) / (q99 - q01 + 1e-8) - 1, value), -1, 1).astype(np.float32)


def audit_case_protocol(directory, model_id, offset, count, protocol, model_seed, env_seed):
    if (directory / "exit-code.txt").read_text().strip() != "0": raise ValueError("Evaluation failed")
    command = json.loads((directory / "invocation.json").read_text())["command"]
    def flag(key): return command[command.index(key) + 1]
    for key, value in (("--model-id", model_id), ("--initial-state-offset", str(offset)),
                       ("--num_trials_per_task", str(count)), ("--seed", str(model_seed)), ("--env-seed", str(env_seed)),
                       ("--seed-protocol", protocol), ("--task_suite_name", "libero_spatial")):
        if flag(key) != value: raise ValueError(f"Actual command differs: {model_id} {key}")
    if flag("--weight-bits") != "4" or flag("--activation-bits") != "16" or flag("--awq-scope") != ("none" if model_id == "BF16" else "all"):
        raise ValueError("Wrong actual baseline quantization scope/bits")
    if "--trace-observations" not in command or "--trace-actions" not in command:
        raise ValueError("Both traces must really be enabled")
    statistics = Path(flag("--pretrained_checkpoint")) / "dataset_statistics.json"
    norm = json.loads(statistics.read_text(encoding="utf-8"))["libero_spatial_no_noops"]["proprio"]
    logs = list(directory.glob("EVAL-*.txt"))
    if len(logs) != 1: raise ValueError("Expected exactly one evaluation log")
    log = logs[0].read_text()
    manifests = [json.loads(m.group(1)) for m in re.finditer(r"EPISODE_MANIFEST (\{.*\})", log)]
    successes = [m.group(1) == "True" for m in re.finditer(r"Success: (True|False)", log)]
    expected = {(t, r) for t in range(10) for r in range(offset, offset + count)}
    keys = [(m["task_id"], m["init_state_index"], m["init_state_sha256"], m["env_seed"], m["model_seed"], m["protocol"]) for m in manifests]
    if len(keys) != len(expected) or len(successes) != len(expected) or len(set(keys)) != len(expected) or {(k[0], k[1]) for k in keys} != expected:
        raise ValueError("Completed episode count or exact task/reset coverage mismatch")
    if any(k[3:] != (env_seed, model_seed, protocol) for k in keys):
        raise ValueError("Manifest protocol/seed differs from actual command")
    if re.search(r"Episode error:|Traceback \(most recent call last\)", log):
        raise ValueError("Program error in completed evaluation")
    events = read_lines(directory / "on-policy-events.jsonl")
    queries = read_lines(directory / "policy-queries.jsonl")
    eq = [r for r in events if r["record_type"] == "query"]
    starts = [r for r in events if r["record_type"] == "episode_start"]
    ends = [r for r in events if r["record_type"] == "episode_end"]
    if ({r["episode_serial"] for r in starts} != set(range(len(keys))) or len(starts) != len(keys)
            or len(ends) != len(keys) or [r["episode_serial"] for r in ends] != list(range(len(keys)))
            or [r["success"] for r in ends] != successes or any(r["aborted"] for r in ends)):
        raise ValueError("Episode trace incomplete/aborted or success mapping differs")
    if len(eq) != len(queries) or not eq: raise ValueError("Action/observation query traces mismatch")
    first, next_query = {}, {i: 0 for i in range(len(keys))}
    for row, query in zip(eq, queries):
        serial = row["episode_serial"]
        if serial not in next_query or row["query_in_episode"] != next_query[serial] or row["chunk_execution_steps"] != 8:
            raise ValueError("Non-contiguous query or wrong chunk execution semantics")
        if row["executed_step_start"] != row["query_in_episode"] * 8:
            raise ValueError("Re-inference must follow eight executed steps")
        next_query[serial] += 1
        action = np.asarray(query["raw_policy_chunk"])
        if query["finite"] is not True or action.shape != (8, 7) or not np.isfinite(action).all():
            raise ValueError("Nonfinite/non-7D/non-eight-step action")
        source = directory / row["file"]
        if not source.resolve().is_relative_to(directory.resolve()) or sha(source) != row["file_sha256"]:
            raise ValueError("Observation file missing or SHA mismatch")
        with np.load(source, allow_pickle=False) as sample:
            fingerprint = observation_hash(sample["image"], sample["wrist_image"], sample["state"], str(sample["instruction"].item()))
            if not np.array_equal(sample["student_action"], action.astype(np.float32)):
                raise ValueError("Saved observation action differs from execution trace")
            if (row["task"] != query["task"] or row["task"] != str(sample["instruction"].item())
                    or query.get("state_space") != "raw_proprio_before_get_action"
                    or not np.array_equal(sample["state"], normalized_trace_state(query["state"], norm))):
                raise ValueError("Trace task/proprio differs from saved policy observation")
        if fingerprint != row["observation_sha256"]: raise ValueError("Observation fingerprint mismatch")
        if row["query_in_episode"] == 0: first[serial] = fingerprint
    if set(first) != set(range(len(keys))) or any(r["queries"] != next_query[r["episode_serial"]] for r in ends):
        raise ValueError("Missing first observation or query count mismatch")
    return {"keys": keys, "success": successes, "first": first, "query_count": len(eq),
            "sha256": {p.name: sha(p) for p in (logs[0], directory / "invocation.json",
                        directory / "policy-queries.jsonl", directory / "on-policy-events.jsonl", statistics)}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('paired', 'upstream', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise ValueError('Never overwrite an audit')
    results = {}
    for protocol, root, seed, env_seed in [('paired', a.paired, 0, 1), ('upstream', a.upstream, 7, 0)]:
        results[protocol] = {c: audit_case_protocol(root/c, c, 0, 5, protocol, seed, env_seed) for c in ('BF16', 'A0')}
    output = {'gate': 'PASS_PROTOCOL_TRACE_AUDIT', 'episodes_per_case': 50,
              'classification': 'Same-task development diagnostic; seed and stream conventions jointly differ',
              'auditor_sha256': sha(Path(__file__)), 'protocols': {}, 'cross_protocol': {},
              'scope': 'Fixed W4 profile/implementation; no author recipe claim',
              'limits': '50/reset0-4 is not the full 500 benchmark; cross-protocol label changes are not same-observation rescue/break'}
    for protocol, cases in results.items():
        b, w = cases['BF16'], cases['A0']
        if b['keys'] != w['keys']: raise ValueError('Within-protocol task/reset/init/seed keys differ')
        matched = b['first'] == w['first']
        rows = [{'task_id': key[0], 'init_state_index': key[1],
                 'BF16': b['success'][i], 'A0': w['success'][i]} for i, key in enumerate(b['keys'])]
        output['protocols'][protocol] = {'first_observations_match': matched,
            'changed_first_episode_serials': [i for i in b['first'] if b['first'][i] != w['first'][i]],
            'cases': {c: {'successes': sum(r['success']), 'queries': r['query_count'], 'sha256': r['sha256']} for c, r in cases.items()},
            'strict_paired_comparison': comparison(rows, 'A0', 'BF16') if matched else None,
            'rows': rows}
    for c in ('BF16', 'A0'):
        b, u = results['paired'][c], results['upstream'][c]
        if [k[:3] for k in b['keys']] != [k[:3] for k in u['keys']]: raise ValueError('Cross-protocol initial state mismatch')
        output['cross_protocol'][c] = {'changed_first_observations': sum(b['first'][i] != u['first'][i] for i in b['first']),
            'gains_on_matched_reset_keys': sum(not x and y for x,y in zip(b['success'],u['success'])),
            'losses_on_matched_reset_keys': sum(x and not y for x,y in zip(b['success'],u['success'])),
            'success_count_delta_upstream_minus_paired': sum(u['success'])-sum(b['success'])}
    if not output['protocols']['paired']['first_observations_match']: raise ValueError('Primary paired input gate failed')
    output['interpretation'] = 'Protocol changed observable inputs; measured score effect only on this diagnostic slice. Remaining paper gap not identified; unknown author AWQ recipe and full-cohort effects remain.'
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'gate': output['gate'], 'protocols': {k:{'first_match':v['first_observations_match'], 'successes':{c:r['successes'] for c,r in v['cases'].items()}} for k,v in output['protocols'].items()}, 'cross_protocol':output['cross_protocol']}))


if __name__ == '__main__': main()

