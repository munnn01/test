"""Synthetic official-metric check: exact marginal cohort AP, no task models."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import unittest


def teacher():
    path=Path(__file__).resolve().parents[1]/'adaptive_vcm/v34r1_a4/marginal_ap.py'
    if not path.is_file(): raise AssertionError('exact marginal cohort-AP teacher missing')
    spec=importlib.util.spec_from_file_location('a4_marginal',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def pred(image,score,fp=False):
    return dict(image_id=image,category_id=1,bbox=[20,20,10,10] if fp else [0,0,10,10],score=score)


def direct(annotations,predictions,ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        categories=sorted({a['category_id'] for a in annotations}|{r['category_id'] for r in predictions})
        gt=COCO(); gt.dataset=dict(info={},images=[dict(id=i) for i in ids],categories=[dict(id=c) for c in categories],annotations=annotations); gt.createIndex()
        dt=gt.loadRes(copy.deepcopy(predictions)); ev=COCOeval(gt,dt,'bbox'); ev.params.imgIds=ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0])


class MarginalAPContracts(unittest.TestCase):
    def test_stitch_matches_full_official_evaluation_and_captures_global_score_order(self):
        m=teacher(); anns=[dict(id=i,image_id=i,category_id=1,bbox=[0,0,10,10],area=100,iscrowd=0) for i in (1,2)]
        anchors=[pred(1,.9),pred(1,.8,True),pred(2,.7)]
        oracle=m.MarginalAP(dict(info={},categories=[dict(id=1)]),{1:[anns[0]],2:[anns[1]]},[1,2],anchors)
        candidate=[pred(1,.6),pred(1,.5,True)]
        fresh=direct(anns,candidate+[pred(2,.7)],[1,2])
        self.assertAlmostEqual(oracle.ap_with(1,candidate),fresh,places=12)
        self.assertGreater(fresh,oracle.anchor_ap)
        self.assertAlmostEqual(direct(anns[:1],anchors[:2],[1]),direct(anns[:1],candidate,[1]),places=12)
        self.assertAlmostEqual(oracle.ap_with(1,anchors[:2]),oracle.anchor_ap,places=12)
        with self.assertRaises(ValueError): oracle.ap_with(3,candidate)

    def test_negative_images_have_a_defined_marginal_false_positive_cost(self):
        m=teacher(); ann=dict(id=1,image_id=2,category_id=1,bbox=[0,0,10,10],area=100,iscrowd=0)
        oracle=m.MarginalAP(dict(info={},categories=[dict(id=1)]),{1:[],2:[ann]},[1,2],[pred(1,.9,True),pred(2,.7)])
        self.assertGreater(oracle.ap_with(1,[]),oracle.anchor_ap)

    def test_teacher_does_not_mutate_cached_anchor_or_candidate_predictions(self):
        m=teacher(); ann=dict(id=1,image_id=1,category_id=1,bbox=[0,0,10,10],area=100,iscrowd=0)
        anchors=[pred(1,.9)]; candidate=[pred(1,.8)]; before=copy.deepcopy([anchors,candidate])
        value=m.MarginalAP(dict(categories=[dict(id=1)]),{1:[ann]},[1],anchors)
        value.ap_with(1,candidate)
        self.assertEqual([anchors,candidate],before)

    def test_multiple_categories_and_crowd_match_full_official_ap(self):
        m=teacher(); anns=[dict(id=1,image_id=1,category_id=1,bbox=[0,0,10,10],area=100,iscrowd=0),
            dict(id=2,image_id=2,category_id=3,bbox=[0,0,10,10],area=100,iscrowd=0),
            dict(id=3,image_id=2,category_id=1,bbox=[20,20,10,10],area=100,iscrowd=1)]
        anchors=[pred(1,.7),dict(pred(2,.8),category_id=3),pred(2,.9,True)]
        value=m.MarginalAP(dict(categories=[dict(id=1),dict(id=3)]),{1:anns[:1],2:anns[1:]},[2,1],anchors)
        candidate=[dict(pred(2,.6),category_id=3),pred(2,.9,True)]
        self.assertAlmostEqual(value.ap_with(2,candidate),direct(anns,[pred(1,.7)]+candidate,[1,2]),places=12)


if __name__=='__main__': unittest.main()
