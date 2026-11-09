import importlib.util,dataclasses,json,pathlib,torch
from openpi.training import config
root=pathlib.Path('/root/autodl-tmp/trajectory-drift-20261107'); out=root/'checks/20261109-pi05-conversion-mapping-v1.json'
assert not out.exists()
spec=importlib.util.spec_from_file_location('official_converter',root/'sources/openpi/examples/convert_jax_model_to_pytorch.py'); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
original=torch.nn.Module.load_state_dict
class MappingComplete(BaseException):pass
def checked(self,state,*a,**kw):
 result=original(self,state,*a,**kw)
 if self.__class__.__name__!='PI0Pytorch':return result
 targets=self.state_dict(keep_vars=True); aliases=[];bad=[]
 for name in result.missing_keys:
  value=targets[name]; matches=[k for k in state if k in targets and targets[k].data_ptr()==value.data_ptr()]
  if matches:aliases.append(dict(missing=name,mapped_aliases=matches))
  else:bad.append(name)
 report=dict(gate='PASS_PI05_CONVERSION_PARAMETER_MAPPING' if not bad and not result.unexpected_keys else 'FAIL_PI05_CONVERSION_PARAMETER_MAPPING',mapped_keys=len(state),missing_unmapped=bad,unexpected=result.unexpected_keys,tied_aliases=aliases)
 out.write_text(json.dumps(report,indent=2)); print(report)
 if bad or result.unexpected_keys:raise RuntimeError('Unmapped conversion parameters')
 raise MappingComplete()
torch.nn.Module.load_state_dict=checked
try:mod.convert_pi0_checkpoint(str(root/'models/pi05_libero'),'bfloat16',str(root/'UNUSED_NO_WRITE'),dataclasses.replace(config.get_config('pi05_libero').model,pytorch_compile_mode=None))
except MappingComplete:pass
else:raise RuntimeError('Mapping audit hook did not run')
