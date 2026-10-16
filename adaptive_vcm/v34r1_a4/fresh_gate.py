"""Prospective fresh-DEV resource decision, explicitly separate from actual BD."""
import math


def fresh_rd_gate(anchor,selected,log_ratios,scales,limit):
    if not math.isfinite(limit) or limit>=0: raise ValueError('finite negative frozen resource limit required')
    utility={}
    for codec in ('h264','h265'):
        scale=scales[codec]; paid=log_ratios[codec]
        if not math.isfinite(scale) or scale<=0 or not math.isfinite(paid) or paid>1e-12:
            raise ValueError('positive frozen scales and nonincreasing actual paid-rate ratios required')
        for curve in (anchor[codec],selected[codec]):
            if (len(curve['rate'])!=4 or len(curve['quality'])!=4 or
                not all(math.isfinite(v) and v>0 for v in curve['rate']) or
                not all(math.isfinite(v) and 0<=v<=1 for v in curve['quality'])):
                raise ValueError('complete finite raw four-point curves required')
        drop=sum(a-b for a,b in zip(anchor[codec]['quality'],selected[codec]['quality']))/4
        utility[codec]=paid+scale*drop
    return dict(objective_is_bd_rate=False,utility=utility,frozen_limit=limit,
                eligible_for_test=all(v<limit-1e-12 for v in utility.values()))
