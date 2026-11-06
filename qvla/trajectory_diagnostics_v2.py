"""Matched-input W/A decomposition and event/chunk metrics, CPU only.

These are descriptive diagnostics, not proof of behavioral modes, safety or cause.
v1 geometry/capture remains byte-identical. All six command scales are external,
positive, frozen calibration scales; gripper is never mixed into motion geometry.
"""
from __future__ import annotations
import numpy as np
from qvla.trajectory_diagnostics_v1 import finite


def command_array(value):
    a=finite(value)
    if a.ndim < 2 or a.shape[-1]!=7:
        raise ValueError("Expected ... x H x 7 command array")
    return a


def factorial_errors(fp, weight_only, activation_only, joint, input_hashes, coordinate_ids):
    """Exact inclusion-exclusion on four SAME-input outputs, not a norm sum.

Same weight recipe and same smoothing coordinate are required. With differing
AWQ/SQ transforms, this contrast is not accepted as a W/A factorial experiment.
"""
    arrays=[command_array(x) for x in (fp,weight_only,activation_only,joint)]
    if any(a.shape!=arrays[0].shape for a in arrays):raise ValueError("Output shape mismatch")
    if len(input_hashes)!=4 or not input_hashes[0] or len(set(input_hashes))!=1:
        raise ValueError("Four counterfactuals must share the exact query input")
    if len(coordinate_ids)!=4 or not coordinate_ids[0] or len(set(coordinate_ids))!=1:
        raise ValueError("Four counterfactuals need one verified equivalent coordinate")
    f,w,a,wa=arrays
    dw,da,interaction=w-f,a-f,wa-w-a+f
    return {"weight":dw,"activation":da,"interaction":interaction,"total":wa-f}


def boundary_projection(student, teacher, scales, minimum_change=1e-5):
    """Project error onto adjacent BF16 CHUNK command change (offline surrogate).

Not a physical contact boundary and not an online detector. Angle 0..180 retains
direction; zero reference changes/errors are masked, never called zero degrees.
"""
    s,t=command_array(student),command_array(teacher)
    if s.shape!=t.shape or s.ndim!=3:raise ValueError("Expected matching QxHx7 chunks")
    scale=finite(scales,(6,))
    if np.any(scale<=0) or minimum_change<=0:raise ValueError("Positive frozen scales/change floor required")
    direction=np.diff(t[...,:6],axis=1)/scale
    error=(s[:,:-1,:6]-t[:,:-1,:6])/scale
    norm=np.linalg.norm(direction,axis=-1)
    valid=norm>minimum_change
    unit=np.divide(direction,norm[...,None],out=np.zeros_like(direction),where=valid[...,None])
    signed=np.sum(error*unit,axis=-1)
    perp=np.linalg.norm(error-signed[...,None]*unit,axis=-1)
    error_norm=np.linalg.norm(error,axis=-1)
    angle=np.full(norm.shape,np.nan)
    angle_valid=valid & (error_norm>1e-12)
    cosine=np.divide(signed,error_norm,out=np.zeros_like(signed),where=angle_valid)
    angle[angle_valid]=np.degrees(np.arccos(np.clip(cosine[angle_valid],-1,1)))
    return dict(signed_parallel=np.where(valid,signed,np.nan),perpendicular=np.where(valid,perp,np.nan),
                angle_deg=angle,valid=valid,reference_change_norm=norm)


def first_close_command(commands, close_value=1.):
    """First observed open->close controller command, not verified grasp/contact."""
    a=command_array(commands)
    if a.ndim!=2:raise ValueError("Expected Tx7 controller commands")
    indices=np.flatnonzero((a[1:,-1]==close_value)&(a[:-1,-1]!=close_value))
    return int(indices[0]+1) if len(indices) else None


def event_aligned(traces, event_indices, before=16, after=24):
    if len(traces)!=len(event_indices) or before<0 or after<0:raise ValueError("Invalid event alignment")
    grid=np.arange(-before,after+1)
    data=np.full((len(traces),len(grid)),np.nan)
    no_event=0
    for i,(trace,event) in enumerate(zip(traces,event_indices)):
        a=finite(trace)
        if a.ndim!=1:raise ValueError("Expected scalar traces")
        if event is None:
            no_event+=1;continue
        if type(event) is not int or event<0 or event>=len(a):raise ValueError("Invalid event index")
        idx=grid+event;mask=(idx>=0)&(idx<len(a))
        data[i,mask]=a[idx[mask]]
    return dict(offsets=grid,values=data,counts=np.sum(np.isfinite(data),axis=0),no_event=no_event)


def activation_summary(values, calibration_absmax, clipped_reconstruction=None):
    """Stats over supplied sampled elements; elements are not independent states."""
    a=finite(values).ravel()
    if not len(a) or not np.isfinite(calibration_absmax) or calibration_absmax<=0:
        raise ValueError("Nonempty finite activation values and frozen range required")
    mean=float(a.mean());var=float(np.mean((a-mean)**2))
    out={"sampled_elements":len(a),"mean":mean,"variance":var,
         "abs_p99_9":float(np.percentile(np.abs(a),99.9)),
         "out_of_calibration_range_fraction":float(np.mean(np.abs(a)>calibration_absmax)),
         "skewness":float(np.mean((a-mean)**3)/var**1.5) if var>0 else None,
         "excess_kurtosis":float(np.mean((a-mean)**4)/var**2-3) if var>0 else None}
    if clipped_reconstruction is not None:
        q=finite(clipped_reconstruction).ravel()
        if q.shape!=a.shape:raise ValueError("Activation pair shape mismatch")
        out["paired_mse"]=float(np.mean((a-q)**2))
    return out


def demo_symmetric_qdq(value,bits,range_abs):
    """Explicit NumPy toy QDQ for unit tests/figures, NOT an OFT quantizer backend."""
    a=finite(value)
    if bits not in (4,8) or range_abs<=0:raise ValueError("Demo QDQ supports A4/A8 only")
    qmax=2**(bits-1)-1;scale=range_abs/qmax
    return np.clip(np.rint(a/scale),-qmax,qmax)*scale
