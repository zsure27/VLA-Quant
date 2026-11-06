"""DyQ Sec.3-inspired paired analysis, independent of model backends and CUDA.

Pose error at query t -> NEXT query's action response, same-precision branches.
Native pi0.5/GR00T collectors are NOT supplied by this CPU analysis module.
"""
from __future__ import annotations
import numpy as np
from qvla.trajectory_diagnostics_v1 import finite, rotation_difference


def measure_pair(reference,perturbed,position_ref,position_pert,quat_ref,quat_pert,
                 next_action_ref,next_action_pert,scales,query_index,next_query_index,
                 noise_floor=1e-8):
    keys=('model','precision','suite','task_id','reset_index','checkpoint_sha256',
          'quantizer_sha256','normalization_sha256','model_seed','env_seed',
          'matched_noise_sha256','execute_horizon','query_timebase','coordinate_frame')
    for k in keys:
        if k not in reference or reference[k]!=perturbed.get(k):
            raise ValueError('Unmatched SAME-precision branch identity: '+k)
    if reference.get('perturbed') is not False or perturbed.get('perturbed') is not True:
        raise ValueError('Expected an unperturbed and a perturbed closed-loop branch')
    if reference['model'] not in ('pi05','groot_n15') or reference['execute_horizon']!={'pi05':5,'groot_n15':1}[reference['model']]:
        raise ValueError('Native execution horizon differs from this preregistered protocol')
    if next_query_index!=query_index+1:raise ValueError('Must pair error t with action response t+1')
    if not np.isfinite(noise_floor) or noise_floor<=0:raise ValueError('Positive finite repeatability floor required')
    h=reference['execute_horizon']
    a,b=finite(next_action_ref),finite(next_action_pert)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=7 or a.shape[0]<h:
        raise ValueError('Matching controller-space action chunks with executed prefix required')
    s=finite(scales['action_channels'],(6,))
    spatial_scales=finite([scales['translation_m'],scales['rotation_rad']],(2,))
    if np.any(s<=0) or np.any(spatial_scales<=0):
        raise ValueError('Positive frozen, per-backbone calibration scales required')
    eT=float(np.linalg.norm(finite(position_pert,(3,))-finite(position_ref,(3,))))
    vector=rotation_difference(quat_pert,quat_ref);eR=float(np.linalg.norm(vector))
    epsT=eT/scales['translation_m'];epsR=eR/scales['rotation_rad']
    delta=(b[:h,:6]-a[:h,:6])/s
    dT=float(np.sqrt(np.mean(np.sum(delta[:,:3]**2,axis=1))))
    dR=float(np.sqrt(np.mean(np.sum(delta[:,3:]**2,axis=1))))
    dAll=float(np.sqrt(np.mean(np.sum(delta**2,axis=1))))
    return dict(query=query_index,next_query=next_query_index,epsilon_T=epsT,epsilon_R=epsR,
        delta_A_T_next=dT,delta_A_R_next=dR,delta_A_all_next=dAll,
        sensitivity_T=None if epsT<=noise_floor else dAll/epsT,
        sensitivity_R=None if epsR<=noise_floor else dAll/epsR,
        translation_error_m=eT,rotation_error_rad=eR,rotation_log_world_rad=vector.tolist(),
        interpretation='same-precision perturbed versus unperturbed; NOT quantized versus FP action subtraction')


def complete_case_ids(rows,precisions,times):
    """Fixed pair set across shown times AND precisions, with risk-set accounting."""
    index={}
    for row in rows:
        key=(row['case_id'],row['precision'],row['tau'])
        if key in index:raise ValueError('Duplicate case/precision/time record')
        index[key]=row
    seen={r['case_id'] for r in rows}
    complete={case for case in seen if all((case,p,t) in index for p in precisions for t in times)}
    risk={p:{str(t):sum((case,p,t) in index for case in seen) for t in times} for p in precisions}
    return complete,risk


def reflected_kde(points,xgrid,ygrid,bandwidth):
    """Positive-domain reflected 2D Gaussian KDE; bandwidth fixed across panels."""
    a=finite(points)
    if a.ndim!=2 or a.shape[1]!=2 or not len(a) or np.any(a<0):raise ValueError('Nonnegative Nx2 points required')
    hx,hy=bandwidth
    if min(hx,hy)<=0:raise ValueError('Positive frozen bandwidth required')
    xx,yy=np.meshgrid(xgrid,ygrid);z=np.zeros(xx.shape)
    for x,y in a:
        kx=np.exp(-.5*((xx-x)/hx)**2)+np.exp(-.5*((xx+x)/hx)**2)
        ky=np.exp(-.5*((yy-y)/hy)**2)+np.exp(-.5*((yy+y)/hy)**2)
        z+=kx*ky
    return z/(len(a)*2*np.pi*hx*hy)


def scalar_summary(values):
    a=finite(values)
    if a.ndim!=1 or not len(a):raise ValueError('Nonempty scalar cohort required')
    return dict(n=len(a),mean=float(a.mean()),std=float(a.std()),median=float(np.median(a)),
                p95=float(np.percentile(a,95)))
