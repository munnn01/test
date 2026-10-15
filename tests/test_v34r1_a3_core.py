"""CPU fixtures for the prospective A3 split, source selector and complete grid."""
import importlib.util
import math
from pathlib import Path
import unittest


def core():
    path=Path(__file__).resolve().parents[1]/'adaptive_vcm/v34r1_a3/core.py'
    if not path.is_file():
        raise AssertionError('A3 source-only probe core is missing')
    spec=importlib.util.spec_from_file_location('a3_core',path)
    result=importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


class A3Contracts(unittest.TestCase):
    def test_fresh_split_excludes_every_old_id_and_groups_images(self):
        m=core(); images=[{'id':i,'file_name':f'{i:012d}.jpg'} for i in range(1000)]
        cohorts=m.fresh_cohorts(images,set(range(100)),dev=128,test=512)
        a={r['id'] for r in cohorts['fit']}; b={r['id'] for r in cohorts['validation']}
        c={r['id'] for r in cohorts['test']}
        self.assertEqual((len(a),len(b),len(c)),(64,64,512))
        self.assertFalse((a|b|c)&set(range(100)))
        self.assertFalse(a&b or a&c or b&c)
        self.assertEqual(cohorts,m.fresh_cohorts(list(reversed(images)),set(range(100)),dev=128,test=512))
        with self.assertRaises(ValueError): m.fresh_cohorts(images,set(range(800)),dev=128,test=512)

    def test_source_selector_rejects_guard_damage_and_paid_rate_overrun(self):
        m=core()
        rows=[{'profile':i,'bytes':v,'cost':c,'gt_ap':-999} for i,(v,c) in enumerate([(100,1),(40,1.2),(70,1.01),(101,0)])]
        self.assertEqual(m.choose(rows,'cost',.02),2)
        self.assertEqual(m.choose(rows,'cost',0),0)
        for r in rows: r['gt_ap']=float('nan')
        self.assertEqual(m.choose(rows,'cost',.02),2)
        rows[0]['cost']=None
        self.assertEqual(m.choose(rows,'cost',.02),0)

    def test_identity_required_and_nonfinite_thresholds_are_rejected(self):
        m=core()
        with self.assertRaises(ValueError): m.choose([{'profile':1,'bytes':10,'cost':0}],'cost',.1)
        for bad in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): m.choose([{'profile':0,'bytes':10,'cost':0}],'cost',bad)

    def test_complete_grid_cannot_drop_or_duplicate_a_profile_condition(self):
        m=core(); rows=[dict(image_id=i,codec=c,qp=q,profile=p) for i in (7,8) for c in ('h264','h265') for q in (30,35,40,45) for p in range(21)]
        self.assertEqual(m.audit_grid(rows,[7,8])['rows'],336)
        with self.assertRaises(ValueError): m.audit_grid(rows[:-1],[7,8])
        with self.assertRaises(ValueError): m.audit_grid(rows+[rows[0]],[7,8])
        rows[-1]={**rows[-1],'qp':50}
        with self.assertRaises(ValueError): m.audit_grid(rows,[7,8])

    def test_fit_threshold_uses_fit_metrics_only_with_predeclared_quality_limits(self):
        m=core(); base=[.3,.28,.25,.2]
        candidates=[dict(threshold=0,quality=base,log_rate_ratio=0),
                    dict(threshold=.01,quality=[x-.001 for x in base],log_rate_ratio=-.1),
                    dict(threshold=.1,quality=[x-.02 for x in base],log_rate_ratio=-.5)]
        self.assertEqual(m.fit_threshold(candidates,base)['threshold'],.01)
        candidates[1]['quality'][0]=.28
        self.assertEqual(m.fit_threshold(candidates,base)['threshold'],0)


if __name__=='__main__': unittest.main()
