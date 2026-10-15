"""Shape-preserving task BD-rate and a strict six-endpoint -15% report."""
from __future__ import annotations

import math
from collections import Counter
import numpy as np
from scipy.interpolate import PchipInterpolator
from . import QPS

ENDPOINTS = tuple(f'{task}/{codec}' for task in ('ar', 'od', 'ot') for codec in ('h264', 'h265'))
TARGET = -15.0


def _below_target(value):
    # Treat roundoff at the exact threshold as unresolved, never a success.
    return value < TARGET and not math.isclose(value, TARGET, rel_tol=0, abs_tol=64*np.finfo(float).eps*abs(TARGET))


def _fit(curve):
    rate = np.asarray(curve['rate'], dtype=float)
    quality = np.asarray(curve['quality'], dtype=float)
    if rate.shape != (4,) or quality.shape != (4,):
        raise ValueError('exactly four paired rate/quality points are required')
    if not np.isfinite(rate).all() or not np.isfinite(quality).all() or np.any(rate <= 0):
        raise ValueError('rates must be positive and all curve points finite')
    order = np.argsort(rate, kind='stable')
    if np.any(np.diff(quality[order]) < -1e-12):
        raise ValueError('quality is not monotone with rate')
    unique = np.unique(quality)
    if len(unique) < 3:
        raise ValueError('at least three distinct quality values are required')
    # Equal-quality points are a plateau; the lowest actual rate dominates.
    log_rate = np.array([np.log(rate[quality == value].min()) for value in unique])
    return unique, PchipInterpolator(unique, log_rate, extrapolate=False)


def assess_curve(anchor, adaptive, *, qps, ci=None):
    if tuple(qps) != QPS:
        raise ValueError('V34r1 requires the frozen four-QP grid [30,35,40,45]')
    result = dict(qps=list(QPS), primary_interpolator='PCHIP', threshold_pct=TARGET,
                  valid=False, pchip_bd_rate_pct=None, point_passes=False,
                  threshold_confirmed=False, ci=ci, invalid_reason=None)
    try:
        qa, a = _fit(anchor)
        qb, b = _fit(adaptive)
        low, high = float(max(qa.min(), qb.min())), float(min(qa.max(), qb.max()))
        if high-low <= 1e-9:
            raise ValueError('no nontrivial common quality interval; extrapolation is prohibited')
        value = float(np.expm1((b.integrate(low, high)-a.integrate(low, high))/(high-low))*100)
        if not math.isfinite(value):
            raise ValueError('nonfinite integrated BD-rate')
        result.update(valid=True, pchip_bd_rate_pct=value, common_quality_interval=[low, high],
                      point_passes=_below_target(value))
        if ci:
            hi = ci.get('hi')
            interval_valid = (hi is not None and math.isfinite(hi)
                              and ci.get('finite_fraction', 0) >= .95
                              and ci.get('draws', 0) >= 2000
                              and ci.get('familywise_endpoints') == len(ENDPOINTS)
                              and ci.get('invalid_draw_policy') == 'worst-case-unbounded'
                              and ci.get('independent_unit') in ('video', 'image', 'sequence')
                              and ci.get('independent_count', 0) >= 2
                              and ci.get('one_sided_confidence', 0) >= 1-.05/len(ENDPOINTS))
            result['threshold_confirmed'] = bool(result['point_passes'] and interval_valid and _below_target(hi))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        result['invalid_reason'] = str(error)
    return result


def assess_all(endpoints):
    missing = [name for name in ENDPOINTS if name not in endpoints]
    invalid = [name for name in ENDPOINTS if name in endpoints and not endpoints[name].get('valid', False)]
    complete = not missing and not invalid
    return dict(required_endpoints=list(ENDPOINTS), missing_endpoints=missing, invalid_endpoints=invalid,
                all_point_targets=bool(complete and all(endpoints[name].get('point_passes', False) for name in ENDPOINTS)),
                all_thresholds_confirmed=bool(complete and all(endpoints[name].get('threshold_confirmed', False) for name in ENDPOINTS)),
                threshold_pct=TARGET, endpoints=endpoints)


def paired_ar_ci(rows,codec,*,draws=5000,seed=303401):
    """Paired video bootstrap; all four QPs and both arms share every resample."""
    subset=[r for r in rows if r['codec']==codec and r['arm'] in ('anchor','adaptive')]
    ids=sorted({r['id'] for r in subset}); lookup={(r['id'],r['qp'],r['arm']):r for r in subset}
    if len(ids)<2 or draws<1 or len(lookup)!=len(ids)*8: raise ValueError('complete paired AR videos required')
    rates={arm:np.array([[lookup[(i,q,arm)]['bpp'] for q in QPS] for i in ids]) for arm in ('anchor','adaptive')}
    correct={arm:np.array([[lookup[(i,q,arm)]['correct'] for q in QPS] for i in ids]) for arm in rates}
    rng=np.random.default_rng(seed); values=[]; invalid=Counter()
    for _ in range(draws):
        indices=rng.integers(0,len(ids),len(ids))
        curves={arm:dict(rate=rates[arm][indices].mean(0),quality=correct[arm][indices].mean(0)) for arm in rates}
        result=assess_curve(curves['anchor'],curves['adaptive'],qps=QPS)
        if result['valid']: values.append(result['pchip_bd_rate_pct'])
        else: invalid[result['invalid_reason']]+=1
    alpha=.05/len(ENDPOINTS)
    # Undefined draws are unknown outcomes, not missing-at-random samples.
    # Pay their entire mass in each adverse tail before taking quantiles.
    missing=draws-len(values)
    lower=float(np.quantile([*values,*([-np.inf]*missing)],alpha,method='lower'))
    upper=float(np.quantile([*values,*([np.inf]*missing)],1-alpha,method='higher'))
    return dict(lo=lower if math.isfinite(lower) else None,
                hi=upper if math.isfinite(upper) else None,
                draws=draws,seed=seed,finite_fraction=len(values)/draws,invalid_reasons=dict(invalid),
                familywise_endpoints=len(ENDPOINTS),independent_unit='video',independent_count=len(ids),
                one_sided_confidence=1-alpha,
                invalid_draw_policy='worst-case-unbounded',
                scope='paired video percentile bootstrap; undefined draws pay worst-case unbounded tail mass; no population claim beyond source sampling assumptions')
