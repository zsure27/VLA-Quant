#!/usr/bin/env bash
set -Eeuo pipefail
OFFSET=${1:-0}; COUNT=${2:-5}
[[ "$OFFSET" =~ ^[0-9]+$ && "$COUNT" =~ ^[0-9]+$ ]] || exit 2
(( COUNT > 0 && OFFSET + COUNT <= 50 )) || exit 2
ROOT=${VLA_EXPERIMENT_ROOT:-/root/autodl-tmp/qvla-repro}
REPO=${VLA_REPO:-/root/autodl-tmp/VLA-Quant-p2c-20260925}
BASE=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w2.pt
W4=$ROOT/artifacts/awq-spatial-20260912-163735-1136/profiles/w4.pt
G64=$ROOT/artifacts/awq-primary-group-20260913-213127-3342/profiles/w2-g64.pt
LORA_STATE=${VLA_LORA_STATE:?VLA_LORA_STATE must name the exact trained Recovery LoRA checkpoint}
OFT_ROOT=${VLA_OFT_ROOT:-$ROOT/overlays/awq-p0-stage-20260923/oft}
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=$ROOT/eval/p2c-lora-closed-loop-pilot-${OFFSET}-$((OFFSET+COUNT-1))-$STAMP-$$
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 3
for path in "$REPO/qvla/run_eval_official_quant.py" "$REPO/qvla/recovery_lora.py" "$OFT_ROOT/experiments/robot/libero/run_libero_eval.py" "$LORA_STATE" "$BASE" "$W4" "$G64"; do test -s "$path"; done
mkdir -p "$OUT"
printf '%s\n' "$OUT" > "$ROOT/eval/LATEST_P2C_LORA.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit-code.txt"' EXIT
source /root/miniconda3/bin/activate /root/miniconda3/envs/qvla-oft
export PYTHONPATH="$REPO:$OFT_ROOT:$ROOT/src/LIBERO"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 WANDB_MODE=disabled MUJOCO_GL=egl
export TF_NUM_INTEROP_THREADS=2 TF_NUM_INTRAOP_THREADS=4 TOKENIZERS_PARALLELISM=false
python - "$OUT/preregistered-protocol.json" "$OFFSET" "$COUNT" "$LORA_STATE" <<'PY'
import hashlib,json,pathlib,sys
out,offset,count,state=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),pathlib.Path(sys.argv[4])
payload={"schema_version":"1.0","question":"Does exact-12L rank8 Recovery LoRA trained by end-to-end action distillation improve paired closed-loop success?","status":"PREREGISTERED_BEFORE_EXECUTION","suite":"libero_spatial","initial_state_offset":offset,"trials_per_task":count,"task_count":10,"episodes_per_config":count*10,"seed":0,"env_seed":0,"seed_protocol":"paired","configs":{"C0":{"w4_layers":"8-15,20-23","adapter":False},"C1":{"w4_layers":"8-15,20-23","adapter":"rank8 Recovery LoRA"},"C2":{"w4_layers":"8-15,18-23","adapter":False}},"case_directories":{"C0":"C0-12L","C1":"C1-12L-lora","C2":"C2-14L"},"adapter_state":{"path":str(state),"sha256":hashlib.sha256(state.read_bytes()).hexdigest(),"bytes":state.stat().st_size},"classification":"historical/development closed-loop evidence; not final blind evaluation"}
pathlib.Path(out).write_text(json.dumps(payload,indent=2)+"\n")
PY
sha256sum "$REPO/qvla/run_eval_official_quant.py" "$REPO/qvla/recovery_lora.py" \
  "$REPO/diagnostics/awq_interventions.py" "$OFT_ROOT/experiments/robot/libero/run_libero_eval.py" \
  "$BASE" "$G64" "$W4" "$LORA_STATE" > "$OUT/CONTRACT_SHA256SUMS.txt"
common=(python "$REPO/qvla/run_eval_official_quant.py" --method awq --weight-bits 2 --activation-bits 16
 --pretrained_checkpoint "$ROOT/models/openvla-7b-oft-finetuned-libero-spatial"
 --profile "$BASE" --awq-primary-group64-profile "$G64" --awq-w4-profile "$W4"
 --official-root "$ROOT/src/official-quantization" --task_suite_name libero_spatial
 --num_trials_per_task "$COUNT" --initial-state-offset "$OFFSET" --libero_root "$ROOT/src/LIBERO"
 --seed 0 --env-seed 0 --seed-protocol paired --trace-actions --awq-scope all
 --awq-candidate w2-attention-primary-g64-stage-w4)
run_config(){
  local name=$1 layers=$2; shift 2
  mkdir -p "$OUT/$name"
  local command=("${common[@]}" --awq-w4-layers "$layers" --local_log_dir "$OUT/$name" "$@")
  printf '%q ' "${command[@]}" > "$OUT/$name/command.txt"; printf '\n' >> "$OUT/$name/command.txt"
  "${command[@]}" 2>&1 | tee "$OUT/$name/console.log"
  printf '0\n' > "$OUT/$name/exit-code.txt"
}
run_config C1-12L-lora '8,9,10,11,12,13,14,15,20,21,22,23' --awq-recovery-lora-state "$LORA_STATE"
run_config C0-12L '8,9,10,11,12,13,14,15,20,21,22,23'
run_config C2-14L '8,9,10,11,12,13,14,15,18,19,20,21,22,23'
printf '{"status":"EXECUTION_COMPLETE","configs":["C0-12L","C1-12L-lora","C2-14L"]}\n' > "$OUT/complete.json"
