import runpy
from experiments.robot import openvla_utils as utils
utils.update_auto_map=lambda _:None
utils.check_model_logic_mismatch=lambda _:None
runpy.run_module('qvla.calibrate_official_quant',run_name='__main__')
