"""Version unstarted B5 continuations to avoid audit output collisions."""
import copy,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'results/experiments/p5-contextual-routing/20261012-b5-spatial-preparation/plans'
NEW=ROOT/'results/experiments/p5-contextual-routing/20261012-091-b5-pilot-v2/continuation-plans-v3'
REMOTE='/root/autodl-tmp/qvla-repro/control/b5-pilot-091-20261012-continuation-v3'
DATA='/root/autodl-tmp/qvla-repro/backups/experiments/p5-contextual-routing/20261012-091-b5-pilot-v2'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 NEW.mkdir(exist_ok=False);records=[]
 for name in ('reset-10-19','reset-20-29','reset-30-39','reset-40-49'):
  oldpath=OLD/(name+'-plan.json');old=json.loads(oldpath.read_text());new=copy.deepcopy(old)
  new['plan_id']=old['plan_id']+'-continuation-v3'
  new['control_root']=REMOTE+'/plans/control'
  for stage in new['stages']:
   if stage['id'] in ('verify-prior','audit-phase'):
    c=stage['command'];i=c.index('--output');c[i+1]=DATA+'/stage-records/'+name+'-'+stage['id']+'-v3'
   if name=='reset-10-19' and stage['id']=='verify-prior':
    c=stage['command'];i=c.index('--references');c.insert(i,DATA+'/first50/random')
    i=c.index('--require');c[i+1]=DATA+'/first50-all-controls-gate.json'
  for a,b in zip(old['stages'],new['stages']):
   if a['id'] not in ('verify-prior','audit-phase'):assert a==b,'GPU stage command changed'
  new['registration_reason']='Unique wrapper audit output directories; next10 prior gate includes random control. GPU commands/model/data/success-independent continuation unchanged.'
  target=NEW/(name+'-plan.json');target.write_text(json.dumps(new,indent=2)+'\n')
  records.append({'name':name,'old_file_sha256':digest(oldpath),'new_file_sha256':digest(target),'plan_id':new['plan_id'],'remote_path':REMOTE+'/plans/'+target.name,'gpu_commands_exactly_preserved':True})
 outputs=[]
 for p in [OLD/'first50-plan.json']+list(NEW.glob('*plan.json')):
  for s in json.loads(p.read_text())['stages']:
   c=s['command'];outputs.append(c[c.index('--output')+1])
 assert len(outputs)==len(set(outputs)),'Wrapper outputs collide'
 (NEW/'registration.json').write_text(json.dumps({'reason':'Discovered duplicate stage-records/verify-prior and audit-phase across old plans, rejected by frozen run_locked_peft_stage.py. Old plans/source/current first50 retained.','execution_commit':'eb196183a3b1e5d012ea6dbaed6ee3297ee9ce5d','records':records,'gate':'PASS_CONTINUATION_GPU_COMMAND_AND_OUTPUT_PATH_CONTRACT','wrapper_outputs_unique':len(outputs)},indent=2)+'\n')
 print(json.dumps({'gate':'PASS_CONTINUATION_GPU_COMMAND_AND_OUTPUT_PATH_CONTRACT','plans':len(records),'wrapper_outputs_unique':len(outputs)}))
if __name__=='__main__':main()

