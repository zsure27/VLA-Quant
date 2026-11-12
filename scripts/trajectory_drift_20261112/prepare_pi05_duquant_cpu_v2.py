"""Generate original DuQuant block128 rotation on CPU in a new private prep dir.

Only device and selected block size change in upstream AST. No frozen source
or previous blocked receipt is overwritten. No full-model compatibility claim.
"""
import argparse,ast,datetime,hashlib,json,os,pathlib,pickle,sys
import torch
import torch.nn.functional as F

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--original-generator',type=pathlib.Path,required=True);a=p.parse_args()
r,out=a.root.resolve(),a.output.resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(7)
assert not torch.cuda.is_available()
body=a.original_generator.read_bytes();expected='ba28805a63bf18cee2bcc23ab042b1c38a84ad15'
gitsha=lambda b:hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
if gitsha(body)!=expected and gitsha(body.rstrip(b'\n'))==expected:body=body.rstrip(b'\n')
assert gitsha(body)==expected,'Original generator Git blob mismatch'
(out/'upstream-get_rot.py').write_bytes(body)
tree=ast.parse(body);device=next(x for x in tree.body if isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Name) and x.targets[0].id=='device');device.value=ast.Constant(value='cpu')
loop=next(x for x in tree.body if isinstance(x,ast.For));loop.iter=ast.List(elts=[ast.Constant(value=128)],ctx=ast.Load());ast.fix_missing_locations(tree)
os.chdir(out);exec(compile(tree,'pinned-get_rot[cpu,block128]','exec'),{'__name__':'__main__'})
R=pickle.load((out/'Rot.pkl').open('rb'))[128]
assert R.shape==(128,128) and torch.allclose(R.T@R,torch.eye(128),atol=2e-6,rtol=2e-6)
src=r/'sources/DuQuant-archive-v1';sys.path.insert(0,str(src));from quantize.quantizer import UniformAffineQuantizer
kwargs=dict(n_bits=4,dynamic_method='per_token',quant_method='duquant',rotate=True,block_size=128,max_rotation_step=4,permutation_times=1)
q=UniformAffineQuantizer(**kwargs);x=torch.randn(16,128);xr=q.init_duquant(x.clone());basis=q.init_duquant(torch.eye(128));w=torch.randn(9,128)
assert torch.allclose(F.linear(xr,w@basis),F.linear(x,w),atol=1e-4,rtol=1e-4)
q.register_duquant_params();torch.save(q.state_dict(),out/'toy-state.pt');q2=UniformAffineQuantizer(**kwargs);q2.copy_duquant_params(q);q2.register_duquant_params();q2.load_state_dict(torch.load(out/'toy-state.pt',weights_only=True))
assert torch.equal(q(x.clone()),q2(x.clone()))
values={}
for bits in [4,8]:
 q.change_n_bits(bits);y=q(x.clone());assert torch.isfinite(y).all();values[str(bits)]=float((y-xr).square().mean())
save(out/'result.json',dict(gate='PASS_ORIGINAL_DUQUANT_BLOCK128_ROTATION_RELOAD_CPU',original_repository='Hsu1023/DuQuant',original_commit='d56cfc6fe97c34c0eb100fec82fe439865905679',generator_git_blob=expected,generator_sha256=sha(out/'upstream-get_rot.py'),cpu_only_generator_changes=['device=cpu','only n=128'],rotation_asset_sha256=sha(out/'Rot.pkl'),upstream_quantizer_sha256=sha(src/'quantize/quantizer.py'),toy_rotation_steps=4,paired_transform_identity=True,activation_toy_mse=values,save_reload_bitwise=True,gpu_model_admitted=False,full_model_adapter_complete=False,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
print('PASS_ORIGINAL_DUQUANT_BLOCK128_ROTATION_RELOAD_CPU')
