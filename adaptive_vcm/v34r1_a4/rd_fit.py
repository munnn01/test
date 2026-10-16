"""FIT-only endpoint-scaled Lagrangian diagnostic, explicitly NOT BD-rate.

Select before reading calibration outcomes. Interior nonmonotonicity is retained;
it still invalidates actual BD-rate in the independent scientific evaluator.
"""
import math

CODECS=('h264','h265')


def select_rd_fit(anchor_curves,candidates):
    scales={}; base=[]
    for codec in CODECS:
        a=anchor_curves[codec]; rate=a['rate']; quality=a['quality']
        if (len(rate)!=4 or len(quality)!=4 or
            not all(math.isfinite(v) and v>0 for v in rate) or
            not all(math.isfinite(v) and 0<=v<=1 for v in quality) or
            not rate[0]>rate[-1] or not quality[0]>quality[-1]):
            raise ValueError('finite four-point tables with positive endpoint spans required')
        scales[codec]=math.log(rate[0]/rate[-1])/(quality[0]-quality[-1]); base.extend(quality)
    weights=[scales[c] for c in CODECS for _ in range(4)]; scored=[]; seen=set()
    for row in candidates:
        threshold=row['threshold']; quality=row['quality']; lograte=row['log_rate_ratio']
        if (not math.isfinite(threshold) or threshold<0 or threshold in seen or len(quality)!=8 or
            not all(math.isfinite(v) and 0<=v<=1 for v in quality) or
            not math.isfinite(lograte) or lograte>1e-12):
            raise ValueError('unique finite threshold and complete paid-rate/quality grid required')
        seen.add(threshold)
        utility=lograte+sum(w*(a-b) for w,a,b in zip(weights,base,quality))/8
        scored.append(dict(row,utility=utility))
    accepted=[r for r in scored if r['utility']<0]
    return dict(objective_is_bd_rate=False,endpoint_scales=scales,candidates=scored,
                selected=min(accepted,key=lambda r:(r['utility'],r['threshold'])) if accepted else None)
