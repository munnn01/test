"""Pure contracts: deterministic fresh cohorts and a GT-free paid-byte selector."""
import hashlib
import math

QPS=(30,35,40,45)
CODECS=('h264','h265')


def fresh_cohorts(images,excluded,*,dev=128,test=512):
    if dev<2 or dev%2 or test<1: raise ValueError('positive even DEV and positive TEST required')
    ids=[int(r['id']) for r in images]
    if len(ids)!=len(set(ids)): raise ValueError('duplicate image IDs')
    ranked=sorted((dict(r) for r in images if int(r['id']) not in excluded),
                  key=lambda r:hashlib.sha256(('v34r1/A3/20261016/'+str(r['id'])).encode()).hexdigest())
    if len(ranked)<dev+test: raise ValueError('insufficient fresh images; no source substitution')
    return dict(fit=ranked[:dev//2],validation=ranked[dev//2:dev],test=ranked[dev:dev+test])


def choose(rows,proxy,threshold):
    """Read only profile, paid bytes and one declared source proxy. Identity wins ties."""
    if not math.isfinite(threshold) or threshold<0: raise ValueError('invalid source-proxy limit')
    ids=[r['profile'] for r in rows]
    if not rows or len(ids)!=len(set(ids)) or 0 not in ids: raise ValueError('unique profiles including identity required')
    anchor=next(r for r in rows if r['profile']==0)
    if not math.isfinite(anchor['bytes']) or anchor['bytes']<=0: raise ValueError('invalid paid anchor bytes')
    ref=anchor[proxy]
    if ref is None or not math.isfinite(ref): return 0
    accepted=[r for r in rows if r[proxy] is not None and math.isfinite(r[proxy])
              and math.isfinite(r['bytes']) and 0<r['bytes']<=anchor['bytes'] and r[proxy]-ref<=threshold+1e-12]
    return min(accepted,key=lambda r:(r['bytes'],r['profile']))['profile']


def audit_grid(rows,image_ids):
    ids=list(image_ids)
    if len(ids)!=len(set(ids)): raise ValueError('duplicate source IDs')
    expected={(i,c,q,p) for i in ids for c in CODECS for q in QPS for p in range(21)}
    actual=[(r['image_id'],r['codec'],r['qp'],r['profile']) for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=expected: raise ValueError('incomplete/duplicate/drifted A3 four-QP grid')
    return dict(images=len(ids),rows=len(actual),profiles=21,qps=list(QPS),codecs=list(CODECS))


def fit_threshold(candidates,anchor_quality):
    """FIT-only: AP mean drop <=.002 and every-condition drop <=.01; maximize rate saving."""
    if not candidates or not anchor_quality or not all(math.isfinite(v) for v in anchor_quality):
        raise ValueError('finite fit metric tables required')
    accepted=[]
    for r in candidates:
        quality=r['quality']; rate=r['log_rate_ratio']
        if len(quality)!=len(anchor_quality) or not all(math.isfinite(v) for v in quality) or not math.isfinite(rate):
            raise ValueError('fit metric grid drift')
        drop=[a-b for a,b in zip(anchor_quality,quality)]
        if max(drop)<=.01+1e-12 and sum(drop)/len(drop)<=.002+1e-12: accepted.append(r)
    return min(accepted,key=lambda r:(r['log_rate_ratio'],r['threshold'])) if accepted else None
