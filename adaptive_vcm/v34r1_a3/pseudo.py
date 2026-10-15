"""Source detections are pseudo annotations; official AP matching, never GT boxes."""
import contextlib
import io
import numpy as np


def pseudo_damage(source,current,*,threshold=.25,topk=100):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if not 0<=threshold<=1 or topk<1: raise ValueError('invalid pseudo reference settings')
    indices=sorted((i for i,s in enumerate(source['scores']) if s>=threshold),
                   key=lambda i:(-source['scores'][i],i))[:topk]
    if not indices: return None
    def box(values):
        x,y,xx,yy=map(float,values)
        if xx<=x or yy<=y: raise ValueError('nonpositive detector box')
        return [x,y,xx-x,yy-y]
    annotations=[]
    for j,i in enumerate(indices,1):
        b=box(source['boxes'][i])
        annotations.append(dict(id=j,image_id=1,category_id=int(source['labels'][i]),bbox=b,area=b[2]*b[3],iscrowd=0))
    predictions=[dict(image_id=1,category_id=int(c),bbox=box(b),score=float(s))
                 for b,c,s in zip(current['boxes'],current['labels'],current['scores'])]
    if not predictions: return 1.
    categories=sorted({a['category_id'] for a in annotations}|{p['category_id'] for p in predictions})
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO(); gt.dataset=dict(info={},images=[dict(id=1)],annotations=annotations,
                                  categories=[dict(id=i) for i in categories]); gt.createIndex()
        dt=gt.loadRes(predictions); ev=COCOeval(gt,dt,'bbox'); ev.params.imgIds=[1]
        ev.evaluate(); ev.accumulate()
    # Official all-area, maxDets=100, IoU .50:.95 AP. Invalid classes stay -1.
    precision=ev.eval['precision'][:,:,:,0,2]; valid=precision[precision>-1]
    return 1.-float(np.mean(valid)) if valid.size else None
