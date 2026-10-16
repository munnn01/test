"""Synthetic rate/quality tradeoffs only; no real task scoring on this host."""
import importlib.util
import math
from pathlib import Path
import unittest


def module():
    path=Path(__file__).resolve().parents[1]/'adaptive_vcm/v34r1_a4/rd_fit.py'
    if not path.is_file(): raise AssertionError('A4 FIT rate-distortion objective missing')
    spec=importlib.util.spec_from_file_location('a4_rd_fit',path)
    value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


class RDFitContracts(unittest.TestCase):
    def anchor(self):
        return {c:dict(rate=[8.,4.,2.,1.],quality=[.8,.7,.6,.5]) for c in ('h264','h265')}

    def test_rate_quality_tradeoff_replaces_arbitrary_same_qp_quality_gate(self):
        m=module(); a=self.anchor(); base=a['h264']['quality']*2
        rows=[dict(threshold=.01,quality=[v-.012 for v in base],log_rate_ratio=math.log(.75)),
              dict(threshold=.02,quality=[v-.08 for v in base],log_rate_ratio=math.log(.70))]
        result=m.select_rd_fit(a,rows)
        self.assertEqual(result['selected']['threshold'],.01)
        self.assertLess(result['selected']['utility'],0)
        self.assertEqual(result['objective_is_bd_rate'],False)

    def test_identity_fallback_and_stable_ties(self):
        m=module(); a=self.anchor(); base=a['h264']['quality']*2
        row=dict(quality=base,log_rate_ratio=-.1)
        self.assertEqual(m.select_rd_fit(a,[dict(row,threshold=.04),dict(row,threshold=.02)])['selected']['threshold'],.02)
        self.assertIsNone(m.select_rd_fit(a,[dict(threshold=.01,quality=[v-.1 for v in base],log_rate_ratio=-.01)])['selected'])

    def test_nonmonotone_interior_does_not_become_a_repaired_bd_curve(self):
        m=module(); a=self.anchor(); a['h265']['quality'][1:3]=[.58,.62]
        row=dict(threshold=.02,quality=a['h264']['quality']+a['h265']['quality'],log_rate_ratio=-.1)
        result=m.select_rd_fit(a,[row])
        self.assertEqual(result['selected']['threshold'],.02)
        self.assertEqual(a['h265']['quality'],[.8,.58,.62,.5])
        self.assertEqual(result['objective_is_bd_rate'],False)

    def test_rejects_nonfinite_partial_or_invalid_endpoint_tables(self):
        m=module(); a=self.anchor(); base=a['h264']['quality']*2
        for row in [dict(threshold=.01,quality=base[:-1],log_rate_ratio=-.1),
                    dict(threshold=.01,quality=base,log_rate_ratio=float('nan')),
                    dict(threshold=-.01,quality=base,log_rate_ratio=-.1)]:
            with self.assertRaises(ValueError): m.select_rd_fit(a,[row])
        a['h264']['quality'][-1]=.8
        with self.assertRaises(ValueError): m.select_rd_fit(a,[])


if __name__=='__main__': unittest.main()
