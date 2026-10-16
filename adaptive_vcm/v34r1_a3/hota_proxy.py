"""Official HOTA on observed source/candidate tracks. No GT, future, or detector."""
import numpy as np


def source_hota(reference,current,metric,*,lookback=None):
    if len(reference)!=len(current) or not reference: raise ValueError('aligned nonempty observed track histories required')
    if lookback is not None and (type(lookback) is not int or lookback<1): raise ValueError('positive causal lookback required')
    numbers=[r['frame_number'] for r in reference]
    if numbers!=[r['frame_number'] for r in current] or any(b-a!=1 for a,b in zip(numbers,numbers[1:])):
        raise ValueError('source/candidate histories must share contiguous observed frame numbers')
    if lookback is not None: reference=reference[-lookback:]; current=current[-lookback:]
    def normalize(frames):
        values=[]; ids=[]
        for frame in frames:
            raw=np.asarray(frame['track_ids']); boxes=np.asarray(frame['boxes'],float).reshape(-1,4)
            if (raw.shape!=(len(boxes),) or not np.isfinite(raw).all() or not np.equal(raw,np.floor(raw)).all()
                or len(set(raw.tolist()))!=len(raw) or not np.isfinite(boxes).all()
                or (len(boxes) and not (boxes[:,2:]>boxes[:,:2]).all())):
                raise ValueError('finite unique integer IDs and positive boxes required per frame')
            ids.extend(int(i) for i in raw); values.append((raw.astype(int),boxes))
        mapping={v:i for i,v in enumerate(sorted(set(ids)))}
        return [(np.array([mapping[int(i)] for i in raw],int),boxes) for raw,boxes in values],len(mapping)
    source,n_source=normalize(reference); decoded,n_decoded=normalize(current)
    n_gt=sum(len(i) for i,_ in source); n_dt=sum(len(i) for i,_ in decoded)
    if n_gt==0: return None
    similarity=[]
    for (_,a),(_,b) in zip(source,decoded):
        low=np.maximum(a[:,None,:2],b[None,:,:2]); high=np.minimum(a[:,None,2:],b[None,:,2:])
        intersection=np.maximum(high-low,0).prod(2)
        area_a=(a[:,2:]-a[:,:2]).prod(1); area_b=(b[:,2:]-b[:,:2]).prod(1)
        similarity.append(intersection/np.maximum(area_a[:,None]+area_b[None,:]-intersection,1e-12))
    data=dict(num_timesteps=len(source),num_gt_dets=n_gt,num_tracker_dets=n_dt,num_gt_ids=n_source,num_tracker_ids=n_decoded,
              gt_ids=[i for i,_ in source],tracker_ids=[i for i,_ in decoded],similarity_scores=similarity)
    result=metric.eval_sequence(data)
    return {name:float(np.mean(result[name])) for name in ('HOTA','DetA','AssA')}
