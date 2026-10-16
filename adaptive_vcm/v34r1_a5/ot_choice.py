"""Causal source-HOTA/paid-rate decision, with no GT inputs."""
import math


def choose_lagrangian(rates,costs,weight,actions):
    if (not math.isfinite(weight) or weight<0 or not len(rates) or len(costs)!=len(rates) or len(actions)!=len(rates)
        or rates[0]!=1 or actions[0]!=0 or len(set(actions))!=len(actions)
        or any(type(a) is not int or a<0 for a in actions)
        or not all(math.isfinite(r) and r>0 for r in rates)):
        raise ValueError('aligned paid ratios, identity first and nonnegative finite lambda required')
    if all(v is None for v in costs):
        return 0,dict(empty_source_reference=True,objective_is_bd_rate=False)
    if not all(v is not None and math.isfinite(v) and 0<=v<=1 for v in costs):
        raise ValueError('complete finite source-HOTA costs required')
    delta=[v-costs[0] for v in costs]
    objective=[math.log(r)+weight*d for r,d in zip(rates,delta)]
    available=[i for i,r in enumerate(rates) if r<=1]
    chosen=min(available,key=lambda i:(objective[i],actions[i]))
    return chosen,dict(empty_source_reference=False,objective_is_bd_rate=False,objective=objective,
        relative_source_hota_cost=delta,eligible_paid_indices=available,selected_action=actions[chosen])
