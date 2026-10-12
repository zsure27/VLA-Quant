"""Current-instance resource admission before the first real GPU stage."""
import argparse,json,shutil,socket,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qvla.b5_contract import candidate_contract,require_baseline_cpu,save
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--materials',type=Path,required=True);p.add_argument('--expected-hostname',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import torch
    if socket.gethostname()!=a.expected_hostname or not torch.cuda.is_available():raise ValueError('Actual confirmed instance with user-enabled GPU required')
    m=json.loads(a.materials.read_text());candidate_contract(m);require_baseline_cpu(m)
    if shutil.disk_usage(Path(m['checkpoint']).parent).free<12*1024**3:raise ValueError('Need at least 12GiB persistent free and 6GiB archive reserve; do not delete unique data')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:raise ValueError('GPU busy; recover actual existing plan before starting')
    save(a.output,{'gate':'PASS_B5_GPU_START_RESOURCES','hostname':socket.gethostname(),'gpu':torch.cuda.get_device_name(0),'free_bytes':shutil.disk_usage(Path(m['checkpoint']).parent).free})
if __name__=='__main__':main()
