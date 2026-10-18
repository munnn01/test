"""Pure shared-profile choice and FIT-only valid-curve selection."""
import math


def choose_shared(rates,costs,weight,actions):
    if (len(rates)!=4 or len(costs)!=4 or not actions or actions[0]!=0 or len(set(actions))!=len(actions)
        or not math.isfinite(weight) or weight<0 or any(type(a) is not int or a<0 for a in actions)
        or any(len(r)!=len(actions) or r[0]!=1 or any(not math.isfinite(x) or x<=0 for x in r) for r in rates)
        or any(len(c)!=len(actions) for c in costs)):
        raise ValueError('aligned four-QP tables with identity and finite lambda required')
    if any(all(x is None for x in c) for c in costs):
        return 0,dict(empty_source_reference=True,shared_across_qps=[30,35,40,45],objective_is_bd_rate=False)
    if any(x is None or not math.isfinite(x) or not 0<=x<=1 for c in costs for x in c):raise ValueError('complete finite source costs required')
    delta=[[x-c[0] for x in c] for c in costs]
    objective=[sum(math.log(rates[q][i])+weight*delta[q][i] for q in range(4))/4 for i in range(len(actions))]
    eligible=[i for i in range(len(actions)) if all(rates[q][i]<=1 for q in range(4))]
    chosen=min(eligible,key=lambda i:(objective[i],actions[i]))
    return actions[chosen],dict(empty_source_reference=False,shared_across_qps=[30,35,40,45],relative_source_hota_cost=delta,
        objective=objective,eligible_actions=[actions[i] for i in eligible],selected_action=actions[chosen],objective_is_bd_rate=False)


def select_valid_fit(anchor,candidates):
    r,q=anchor['rate'],anchor['quality']
    if len(r)!=4 or len(q)!=4 or any(not math.isfinite(x) for x in r+q) or min(r)<=0 or r[0]<=r[-1] or q[0]<=q[-1]:
        return dict(selected=None,endpoint_scale=None,reason='invalid fixed FIT endpoint span')
    scale=math.log(r[0]/r[-1])/(q[0]-q[-1]);eligible=[];audits=[]
    for c in candidates:
        if len(c['quality'])!=4 or not math.isfinite(c['log_rate_ratio']) or any(not math.isfinite(x) for x in c['quality']):raise ValueError('finite FIT candidate required')
        utility=c['log_rate_ratio']+scale*sum(a-b for a,b in zip(q,c['quality']))/4
        record=dict(c,utility=utility);audits.append(record)
        if c['bd_valid'] and utility<0:eligible.append(record)
    best=min(eligible,key=lambda c:(c['utility'],c['weight'])) if eligible else None
    return dict(selected=best,endpoint_scale=scale,candidates=audits,scope='H264 FIT only; valid BD and negative endpoint utility; no CAL selection')
