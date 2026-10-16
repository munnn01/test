"""Pure cohort separation and conditional sample-expansion contracts."""
import math

REQUIRED_SCALE_BLOCKS=('grid','policy_lock','source_provenance','paid_header',
    'independent_decode','source_only','no_substitution','measurement_reproducibility')


def audit_ot21(rows,sequences,*,codec=None):
    """Require every paid action for every declared64-frame source condition."""
    codecs=('h264','h265') if codec is None else (codec,)
    if not sequences or any(c not in ('h264','h265') for c in codecs):raise ValueError('declared physical sources/codecs required')
    expected={(n,t,c,q,a) for n in sequences for t in (1,17,33,49) for c in codecs for q in (30,35,40,45) for a in range(21)}
    seen=set()
    for r in rows:
        k=(r['sequence'],r['start'],r['codec'],r['qp'],r['action'])
        if (k not in expected or k in seen or r['split']!=sequences[r['sequence']]
            or r['metadata_bytes']!=40 or r['elementary_bytes']<=0 or r['bytes']!=r['elementary_bytes']+40):
            raise ValueError('duplicate, partial, mislabeled or unpaid OT bank')
        seen.add(k)
    if seen!=expected:raise ValueError('complete all21 four-QP OT grid required')
    return len(seen)


def cohort_folds(groups,*,expected_size=64,forbidden=()):
    if len(groups)!=3 or type(expected_size) is not int or expected_size<1: raise ValueError('three complete cohorts required')
    used=set(); blocked=set(forbidden); clean={}
    for name,ids in sorted(groups.items()):
        ids=list(ids)
        if (len(ids)!=expected_size or any(type(i) is not int for i in ids) or len(set(ids))!=len(ids)
            or set(ids)&(used|blocked)): raise ValueError('disjoint image cohorts excluding pilot/TEST required')
        clean[name]=sorted(ids); used.update(ids)
    return [dict(cohort=name,heldout_ids=ids,train_ids=sorted(used-set(ids))) for name,ids in clean.items()]


def scale4_gate(comparisons,blocks):
    reasons=[]
    if any(blocks.get(k) is not True for k in REQUIRED_SCALE_BLOCKS) or any(v is not True for v in blocks.values()):
        reasons.append('incomplete or failed block/provenance checks')
    for codec in ('h264','h265'):
        r=comparisons.get(codec,{})
        if (r.get('valid') is not True or r.get('qps')!=[30,35,40,45] or r.get('primary_interpolator')!='PCHIP'
            or r.get('threshold_pct')!=-15. or r.get('invalid_reason') is not None):
            reasons.append(codec+': invalid four-point comparison');continue
        value=r.get('pchip_bd_rate_pct')
        if type(value) not in (int,float) or not math.isfinite(value) or not value < -15.:
            reasons.append(codec+': point BD is not strictly below -15%')
    return dict(eligible_for_scale4=not reasons,reasons=reasons,threshold_pct=-15.,
        point_gate_only=True,does_not_establish_confidence_confirmation=True)
