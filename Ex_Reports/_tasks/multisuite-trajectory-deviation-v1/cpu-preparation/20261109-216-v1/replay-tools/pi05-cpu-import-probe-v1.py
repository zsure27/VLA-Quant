import os,json,pathlib
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['JAX_PLATFORMS']='cpu';os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
root=pathlib.Path.cwd();r={'gate':'IN_PROGRESS_CPU_IMPORTS','gpu_admitted':False}
try:
 import torch,jax
 import openpi.transforms as transforms
 import openpi.models_pytorch.preprocessing_pytorch as preprocessing
 r.update(torch_version=torch.__version__,jax_version=jax.__version__,jax_backend=jax.default_backend(),jax_devices=[str(x) for x in jax.devices()],openpi_transforms_file=transforms.__file__,pytorch_preprocessing_file=preprocessing.__file__)
 assert r['jax_backend']=='cpu'; assert torch.isfinite(torch.eye(2)@torch.ones(2)).all()
 r['gate']='PASS_CPU_IMPORTS_NO_FULL_MODEL_LOAD'
except BaseException as e:r.update(gate='FAIL_CPU_IMPORTS',error_type=type(e).__name__,error=str(e));raise
finally:(root/'checks/20261109-216-cpu-import-probe-v1.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r))
