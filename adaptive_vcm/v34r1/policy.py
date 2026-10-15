"""Feasible learned proposals plus explicitly unlearned measured filter trials."""
import numpy as np


def shortlist(prediction,mask,tolerance,*,calibration=None,max_learned=3,extras=()):
    rate=np.asarray(prediction['rate'],float)
    risk=np.asarray(prediction['risk'],float)
    logits=np.asarray(prediction['logits'],float)
    active=np.asarray(mask,bool)
    tolerance=np.asarray(tolerance,float)
    if (risk.shape!=(len(rate),3) or logits.shape!=rate.shape or active.shape!=(3,) or not active.any()
        or tolerance.shape!=(3,) or max_learned<0):
        raise ValueError('invalid shortlist inputs')
    bounds=risk+(np.asarray(calibration['risk'],float) if calibration else 0)
    feasible=np.isfinite(bounds[:,active]).all(1)&np.isfinite(rate)&(rate>0)&np.isfinite(logits)
    feasible &= (bounds[:,active]<=tolerance[active]).all(1)
    score=rate+.25*np.maximum(risk[:,active],0).mean(1)
    ordered=[int(np.argmax(logits)),*np.argsort(score,kind='stable').tolist()]
    choices=[]
    for action in ordered:
        if action and feasible[action] and action not in choices and len(choices)<max_learned:
            choices.append(action)
    explicit=[int(i) for i in extras]
    if any(i<len(rate) for i in explicit) or len(set(explicit))!=len(explicit):
        raise ValueError('extra profiles must be distinct unlearned action IDs')
    return [0,*choices,*explicit]


def select_measured(rates,costs,mask,limits,*,budget=1.,quality_weight=0.):
    rate=np.asarray(rates,float)
    values=np.asarray(costs,float)
    active=np.asarray(mask,bool)
    limits=np.asarray(limits,float)
    if (rate.ndim!=1 or len(rate)<1 or values.shape!=(len(rate),3) or active.shape!=(3,) or not active.any()
        or limits.shape!=(3,) or not np.isfinite(limits).all() or np.any(limits<0)
        or not np.isfinite(budget) or budget<1 or not 0<=quality_weight<=1):
        raise ValueError('invalid measured source-guard table')
    if not np.isfinite(values[0,active]).all() or rate[0]!=1:
        raise ValueError('identity must have finite source costs and rate ratio one')
    excess=values-values[0]
    feasible=np.isfinite(values[:,active]).all(1)&np.isfinite(rate)&(rate>0)&(rate<=budget)
    feasible &= (excess[:,active]<=limits[active]).all(1)
    normalized=np.maximum(excess[:,active],0)/np.maximum(limits[active],1e-8)
    objective=rate+quality_weight*np.nan_to_num(normalized,nan=1e12).mean(1)
    choice=int(np.argmin(np.where(feasible,objective,np.inf)))
    return choice,dict(rejected=np.flatnonzero(~feasible).tolist(),objective=objective.tolist(),
        rate=rate.tolist(),excess=[[float(v) if np.isfinite(v) else None for v in row] for row in excess],
        mask=active.tolist(),limits=limits.tolist(),budget=float(budget),quality_weight=float(quality_weight),
        scope='source-only task consistency; actual measured packet bytes; no GT accuracy guarantee')
