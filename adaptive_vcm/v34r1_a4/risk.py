"""Whitelisted sender features, portable tree critic, and paid-byte selection."""
import math
import numpy as np

CUTS=(.05,.1,.25,.5,.75,.9)


def _pred(value):
    boxes=np.asarray(value['boxes'],float).reshape(-1,4); scores=np.asarray(value['scores'],float); labels=np.asarray(value['labels'],int)
    if (scores.shape!=(len(boxes),) or labels.shape!=scores.shape or not np.isfinite(boxes).all() or not np.isfinite(scores).all()
        or np.any(scores<0) or np.any(scores>1) or np.any(labels<1) or np.any(labels>90)):
        raise ValueError('finite source/candidate detection arrays required')
    return boxes,scores,labels


def _stats(pred):
    boxes,scores,labels=pred; areas=np.maximum(boxes[:,2]-boxes[:,0],0)*np.maximum(boxes[:,3]-boxes[:,1],0)/(320*320)
    names=['count','mean_score','score_std','top_score','mean_area','total_area']
    values=[len(scores)/300,float(scores.mean()) if len(scores) else 0,float(scores.std()) if len(scores) else 0,
            float(scores.max()) if len(scores) else 0,float(areas.mean()) if len(areas) else 0,float(areas.sum())/300]
    for cut in CUTS:
        mask=scores>=cut; names.extend([f'count_ge_{cut}',f'score_sum_ge_{cut}'])
        values.extend([float(mask.sum())/300,float(scores[mask].sum())/300])
    return names,values


def _match(a,b):
    aa,sa,la=a; bb,sb,lb=b
    if not len(aa): return [0.]*10
    if not len(bb): best=np.zeros(len(aa)); retained=np.zeros(len(aa))
    else:
        lo=np.maximum(aa[:,None,:2],bb[None,:,:2]); hi=np.minimum(aa[:,None,2:],bb[None,:,2:])
        inter=np.maximum(hi-lo,0).prod(-1); area_a=np.maximum(aa[:,2:]-aa[:,:2],0).prod(-1); area_b=np.maximum(bb[:,2:]-bb[:,:2],0).prod(-1)
        overlap=inter/np.maximum(area_a[:,None]+area_b[None,:]-inter,1e-12)
        overlap=np.where(la[:,None]==lb[None,:],overlap,0); ix=overlap.argmax(1); best=overlap[np.arange(len(aa)),ix]
        retained=np.where(best>0,sb[ix],0)
    weights=sa/max(float(sa.sum()),1e-12); weak=(sa>=.05)&(sa<.25)
    return [float(best.mean()),float(np.dot(weights,best)),float(best.min()),
        *[float(np.dot(weights,best>=cut)) for cut in (.25,.5,.75)],
        float(np.dot(weights,np.maximum(sa-retained,0))),float(np.mean(best[weak])) if weak.any() else 0,
        float(np.mean(retained[weak])) if weak.any() else 0,float(np.mean(best[sa>=.5])) if np.any(sa>=.5) else 0]


def features(row,anchor,source):
    """No GT, COCO rows, IDs, hashes, or split membership are read."""
    profile=row['profile']; codec=row['codec']; qp=row['qp']
    if type(profile) is not int or not 0<=profile<21 or codec not in ('h264','h265') or qp not in (30,35,40,45): raise ValueError('frozen action/codec/four-QP required')
    if anchor['profile']!=0 or anchor['codec']!=codec or anchor['qp']!=qp: raise ValueError('same-condition identity required')
    if min(row['bytes'],anchor['bytes'],row['metadata_bytes'])<=0: raise ValueError('positive actual paid bytes required')
    names=['h265',*[f'qp_{q}' for q in (30,35,40,45)],*[f'profile_{i}' for i in range(21)],'log_paid_ratio','anchor_bpp','header_fraction']
    values=[float(codec=='h265'),*[float(qp==q) for q in (30,35,40,45)],*[float(profile==i) for i in range(21)],
            math.log(row['bytes']/anchor['bytes']),float(anchor['bpp']),row['metadata_bytes']/row['bytes']]
    for key in ('current_cost','feature_cost','pseudo_ap'):
        a,b=row[key],anchor[key]; missing=a is None or b is None
        names.extend([key,key+'_excess',key+'_missing']); values.extend([0 if a is None else float(a),0 if missing else float(a-b),float(missing)])
    clean,base,current=(_pred(v) for v in (source['source_predictions'],anchor['prediction'],row['prediction']))
    for tag,pred in (('clean',clean),('identity',base),('candidate',current)):
        keys,stats=_stats(pred); names.extend(tag+'_'+n for n in keys); values.extend(stats)
    for tag,a,b in (('clean_to_candidate',clean,current),('clean_to_identity',clean,base),('candidate_to_clean',current,clean)):
        names.extend(tag+'_'+str(i) for i in range(10)); values.extend(_match(a,b))
    # Class frequencies are legitimate encoder detections, never GT categories.
    for label in range(1,91):
        names.extend([f'clean_class_{label}',f'class_excess_{label}'])
        hist=lambda p:float(np.sum((p[2]==label)&(p[1]>=.25)))/300
        values.extend([hist(clean),hist(current)-hist(base)])
    if not all(math.isfinite(v) for v in values): raise ValueError('nonfinite source-only feature')
    return names,values


def choose_predicted(rows,predictions,threshold):
    if len(rows)!=len(predictions) or not math.isfinite(threshold): raise ValueError('complete predicted-risk grid and finite threshold required')
    table={r['profile']:r for r in rows}
    if len(table)!=len(rows) or 0 not in table: raise ValueError('unique profiles including identity required')
    anchor=table[0]; paid=anchor['bytes']
    if not math.isfinite(paid) or paid<=0: raise ValueError('positive paid identity required')
    accepted=[anchor]
    for row,value in zip(rows,predictions):
        if row['profile'] and value is not None and math.isfinite(value) and math.isfinite(row['bytes']) and 0<row['bytes']<=paid and value<=threshold:
            accepted.append(row)
    return min(accepted,key=lambda r:(r['bytes'],r['profile']))['profile']


def predict_forest(model,x):
    if model.get('schema')!='v34r1-risk-forest-1' or not model.get('features') or not 1<=len(model.get('trees',[]))<=256: raise ValueError('bounded portable forest required')
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=len(model['features']) or not np.isfinite(x).all(): raise ValueError('finite matching feature matrix required')
    result=np.zeros(len(x),float)
    for tree in model['trees']:
        left,right,feature=(np.asarray(tree[k],int) for k in ('left','right','feature'))
        threshold,value=(np.asarray(tree[k],float) for k in ('threshold','value')); n=len(left)
        if not 1<=n<=4095 or any(v.shape!=(n,) for v in (right,feature,threshold,value)) or not np.isfinite(threshold).all() or not np.isfinite(value).all(): raise ValueError('invalid tree arrays')
        for i in range(n):
            if left[i]==right[i]==-1: continue
            if not i<left[i]<n or not i<right[i]<n or not 0<=feature[i]<x.shape[1]: raise ValueError('unsafe/noncausal tree indices')
        nodes=np.zeros(len(x),int)
        for _ in range(n):
            active=left[nodes]!=-1
            if not active.any(): break
            ix=np.flatnonzero(active); at=nodes[ix]; nodes[ix]=np.where(x[ix,feature[at]]<=threshold[at],left[at],right[at])
        else: raise ValueError('nonterminating tree')
        result+=value[nodes]
    return (result/len(model['trees'])).tolist()
