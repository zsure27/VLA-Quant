"""Additional closure triggers: audited task completion or confirmed 30m idle GPU."""
import hashlib
from pathlib import Path


def completion_verified(receipt, auth, root):
    if not isinstance(receipt,dict):return False
    expected=auth.get('task_manifest_sha256')
    if not expected or receipt.get('task_manifest_sha256')!=expected:return False
    if receipt.get('round_id')!=root.name or receipt.get('gate')!='PASS_ALL_REQUESTED_TASK_WORK_AUDITED':return False
    if receipt.get('all_requested_work_resolved') is not True or receipt.get('raw_audit_passed') is not True:return False
    try:
        file=root/receipt['recovery_index_relative_path']
        if file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(root.resolve()):return False
        # Reject symlinked parent directories as well as a symlinked file.
        if any(p.is_symlink() for p in [file,*file.parents] if p!=root.parent):return False
        return hashlib.sha256(file.read_bytes()).hexdigest()==receipt['recovery_index_sha256']
    except (KeyError,TypeError,OSError,ValueError):return False


def idle_transition(since, now_monotonic, *, gpu_present, observation_valid,
                    compute_processes_present, owned_work_present,
                    memory_used_mib, utilization_percent):
    # Unknown/no-GPU readings reset continuity; a one-time 0% reading is not idle.
    idle=(gpu_present and observation_valid and not compute_processes_present
          and not owned_work_present and 0<=memory_used_mib<=64
          and utilization_percent==0)
    if not idle:return None,False
    if since is None or since>now_monotonic:since=now_monotonic
    return since,now_monotonic-since>=1800
