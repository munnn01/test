"""GT training teacher only: exact one-image replacement in official cohort COCO AP.

COCO matching is image-local; precision accumulation is cohort-wide. Reuse the
unchanged images' official match tables, replace one image, then reaccumulate.
This teacher is never called by an operational sender selector.
"""
import contextlib
import io
import numpy as np


class MarginalAP:
    def __init__(self,meta,gt_by_id,image_ids,anchor_predictions):
        from pycocotools.coco import COCO
        self.ids=sorted(image_ids)
        if not self.ids or len(self.ids)!=len(set(self.ids)): raise ValueError('unique nonempty cohort IDs required')
        self.position={v:i for i,v in enumerate(self.ids)}
        self.categories=sorted(int(r['id']) for r in meta['categories'])
        with contextlib.redirect_stdout(io.StringIO()):
            self.gt=COCO(); self.gt.dataset=dict(info=meta.get('info',{}),images=[dict(id=i) for i in self.ids],
                categories=[dict(r) for r in meta['categories']],annotations=[dict(a) for i in self.ids for a in gt_by_id[i]])
            self.gt.createIndex()
        self.base=self._evaluate(anchor_predictions,self.ids)
        self.original=list(self.base.evalImgs)
        self.anchor_ap=self._score(self.base)
        if self.anchor_ap is None: raise ValueError('cohort needs positive nonignored GT')

    def _evaluate(self,predictions,ids):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        if any(r['image_id'] not in ids for r in predictions): raise ValueError('prediction image outside declared evaluation IDs')
        with contextlib.redirect_stdout(io.StringIO()):
            if predictions: dt=self.gt.loadRes([dict(r) for r in predictions])
            else:
                dt=COCO(); dt.dataset=dict(info={},images=[dict(id=i) for i in ids],
                    categories=[dict(id=i) for i in self.categories],annotations=[]); dt.createIndex()
            ev=COCOeval(self.gt,dt,'bbox'); ev.params.imgIds=list(ids); ev.params.catIds=self.categories
            # Equivalent official AP slice, omitting unused small/medium/large
            # and maxDets1/10 arrays to make marginal reaccumulation affordable.
            ev.params.areaRng=[[0,1e10]]; ev.params.areaRngLbl=['all']; ev.params.maxDets=[100]
            ev.evaluate()
        return ev

    @staticmethod
    def _score(ev):
        with contextlib.redirect_stdout(io.StringIO()): ev.accumulate()
        precision=ev.eval['precision'][:,:,:,0,0]; valid=precision[precision>-1]
        return float(np.mean(valid)) if valid.size else None

    def ap_with(self,image_id,predictions):
        if image_id not in self.position: raise ValueError('image is outside frozen cohort')
        candidate=self._evaluate(predictions,[image_id])
        if len(candidate.evalImgs)!=len(self.categories): raise ValueError('official evaluation layout drift')
        replaced=list(self.original); offset=self.position[image_id]; n=len(self.ids)
        for category,value in enumerate(candidate.evalImgs): replaced[category*n+offset]=value
        try:
            self.base.evalImgs=replaced
            value=self._score(self.base)
            if value is None: raise ValueError('unexpected missing cohort AP')
            return value
        finally: self.base.evalImgs=self.original
