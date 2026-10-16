"""Synthetic sender feature, portable critic, and actual-paid-byte contracts."""
import copy
import importlib.util
from pathlib import Path
import unittest


def module():
    path=Path(__file__).resolve().parents[1]/'adaptive_vcm/v34r1_a4/risk.py'
    if not path.exists(): raise AssertionError('source-only risk critic contracts missing')
    spec=importlib.util.spec_from_file_location('a4_risk',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def prediction(score=.9): return dict(boxes=[[0,0,10,10]],labels=[1],scores=[score])


def row(profile=0,paid=100):
    return dict(profile=profile,bytes=paid,metadata_bytes=40,bpp=paid*8/(320*320),codec='h264',qp=35,
                prediction=prediction(),current_cost=.1,feature_cost=.01,pseudo_ap=.2)


class RiskContracts(unittest.TestCase):
    def test_features_ignore_labels_identifiers_and_private_provenance(self):
        m=module(); a=row(); candidate=row(14,70); source=dict(source_predictions=prediction())
        before=copy.deepcopy([candidate,a,source]); names,x=m.features(candidate,a,source)
        changed=copy.deepcopy(candidate); changed.update(image_id=987654,split='test',coco=['forbidden'],ground_truth=['do not read'],packet_sha256='different')
        names2,y=m.features(changed,dict(a,image_id=42,ground_truth='secret'),dict(source,image_id=2,split='fit',ground_truth='secret'))
        self.assertEqual(names,names2); self.assertEqual(x,y); self.assertEqual([candidate,a,source],before)
        self.assertFalse(any(word in n for n in names for word in ('image_id','ground_truth','split','sha256')))
        weak=copy.deepcopy(source); weak['source_predictions']['scores']=[.15]
        self.assertNotEqual(x,m.features(candidate,a,weak)[1])

    def test_paid_cap_identity_fallback_and_ties_are_independent_of_gt(self):
        m=module(); rows=[row(0,100),row(14,50),row(13,50),row(20,110)]
        self.assertEqual(m.choose_predicted(rows,[0,.005,.005,-.5],.01),13)
        self.assertEqual(m.choose_predicted(rows,[0,.5,float('nan'),-.5],.01),0)
        self.assertEqual(m.choose_predicted(rows,[0,0,0,-.5],-.01),0)
        with self.assertRaises(ValueError): m.choose_predicted(rows,[0],.01)

    def test_portable_tree_uses_safe_indices_and_sklearn_float32_boundary(self):
        m=module(); model=dict(schema='v34r1-risk-forest-1',features=['x'],trees=[dict(
            left=[1,-1,-1],right=[2,-1,-1],feature=[0,-2,-2],threshold=[.5,-2,-2],value=[0,.01,.2])])
        values=m.predict_forest(model,[[.5],[.50000000001],[.6]])
        self.assertEqual(values,[.01,.01,.2])
        bad=copy.deepcopy(model); bad['trees'][0]['left'][0]=0
        with self.assertRaises(ValueError): m.predict_forest(bad,[[.2]])
        with self.assertRaises(ValueError): m.predict_forest(model,[[float('nan')]])


if __name__=='__main__': unittest.main()
