"""Conservative four-QP BD-rate. Invalid curves remain explicitly unavailable."""
import numpy as np
from scipy.interpolate import PchipInterpolator


def bd_rate(reference_rates, reference_quality, candidate_rates, candidate_quality):
    curves = []
    for rates, quality in ((reference_rates, reference_quality), (candidate_rates, candidate_quality)):
        r, q = np.asarray(rates, dtype=float), np.asarray(quality, dtype=float)
        if r.shape != (4,) or q.shape != (4,) or not np.isfinite(r).all() or not np.isfinite(q).all() or (r <= 0).any():
            return {'percent':None, 'reason':'Four finite positive rates and finite qualities are required'}
        # Inputs stay in QP order. Reverse together only after validating both.
        if not (np.diff(r) < 0).all() or not (np.diff(q) < 0).all():
            return {'percent':None, 'reason':'Rate and quality must both strictly decrease in QP order; plateau/nonmonotone curve'}
        curves.append((q[::-1], np.log(r[::-1])))
    lo, hi = max(c[0][0] for c in curves), min(c[0][-1] for c in curves)
    if hi <= lo:
        return {'percent':None, 'reason':'No nonzero common quality interval'}
    integrals = [PchipInterpolator(q, log_r).integrate(lo, hi) for q, log_r in curves]
    result = float(100*np.expm1((integrals[1]-integrals[0])/(hi-lo)))
    return {'percent':result, 'reason':None, 'quality_overlap':[float(lo),float(hi)],
            'method':'PCHIP log-rate integration; common support only; four fixed QPs'}


def comparisons(summaries, qps=(30,35,40,45), codecs=('h264','h265')):
    rows = []
    for arm, summary in summaries.items():
        for codec in codecs:
            for task, rate_key, quality_key in (('ar','ar_video_bpp','ar_top1'),
                                                 ('od','coco_image_bpp','mAP')):
                curves = []
                for variant in ('identity','npp'):
                    group = [next(c for c in summary['curves'] if
                                  (c['codec'],c['qp'],c['arm']) == (codec,qp,variant)) for qp in qps]
                    curves.append(([c[rate_key] for c in group],
                                   [c['coco'][quality_key] if task=='od' else c[quality_key] for c in group]))
                rows.append({'arm':arm,'codec':codec,'task':task, **bd_rate(*curves[0],*curves[1])})
    return rows
