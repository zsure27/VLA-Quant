"""Join both frozen collectors by whole episodes without outcome selection."""
import argparse,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qvla.b5_contract import VERSION,sha,save,audit_capture,source_hashes
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--captures',type=Path,nargs=2,required=True)
    p.add_argument('--role',choices=('router_train','router_dev'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Never overwrite joined data')
    rows=[];episodes=[];policies=set();inputs=[];norm=None
    for d in a.captures:
        r=json.loads((d/'manifest.json').read_text())
        if r.get('version')!=VERSION or r.get('role')!=a.role or r.get('state_space')!='raw_proprio_before_get_action' or r['source_sha256']!=source_hashes():raise ValueError('Capture role/source differs')
        policies.add(r['policy']);inputs.append({'path':str(d),'sha256':sha(d/'manifest.json')})
        if norm is not None and norm!=r['proprio_norm_stats']:raise ValueError('Collector normalization differs')
        norm=r['proprio_norm_stats']
        episodes.extend(r['episodes']);rows.extend((d,row) for row in r['samples'])
    if policies!={'fixed-e0','fixed-e1'}:raise ValueError('Require both independent frozen collectors')
    a.output.mkdir(parents=True)
    for d,row in rows:
        if sha(d/row['file'])!=row['sha256']:raise ValueError('Capture content changed')
        shutil.copyfile(d/row['file'],a.output/row['file'])
    save(a.output/'manifest.json',{'version':VERSION,'role':a.role,'state_space':'raw_proprio_before_get_action','proprio_norm_stats':norm,'samples':[r for _,r in rows],'episodes':episodes,'sources':inputs,'source_sha256':source_hashes(),'selection':'five uniform query indices per full episode; no outcome selection'})
    audit_capture(a.output,a.role)
if __name__=='__main__':main()
