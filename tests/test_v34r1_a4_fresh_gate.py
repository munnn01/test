"""Synthetic prospective resource gate; an approximate utility never becomes BD."""
import importlib.util
import math
from pathlib import Path
import unittest


def module():
    path=Path(__file__).resolve().parents[1]/'adaptive_vcm/v34r1_a4/fresh_gate.py'
    if not path.exists(): raise AssertionError('prospective fresh-data gate missing')
    spec=importlib.util.spec_from_file_location('a4_fresh_gate',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class FreshGateContracts(unittest.TestCase):
    def curves(self): return {c:dict(rate=[4,3,2,1],quality=[.8,.7,.6,.5]) for c in ('h264','h265')}

    def test_both_codecs_must_strictly_pass_frozen_limit(self):
        m=module(); a=self.curves(); scales=dict(h264=2,h265=3); limit=math.log(.85)
        self.assertFalse(m.fresh_rd_gate(a,a,dict(h264=limit,h265=-.3),scales,limit)['eligible_for_test'])
        self.assertFalse(m.fresh_rd_gate(a,a,dict(h264=-.3,h265=-.1),scales,limit)['eligible_for_test'])
        self.assertTrue(m.fresh_rd_gate(a,a,dict(h264=-.3,h265=-.3),scales,limit)['eligible_for_test'])

    def test_resource_decision_preserves_nonmonotonicity_and_makes_no_bd_claim(self):
        m=module(); a=self.curves(); a['h264']['quality']=[.8,.79,.81,.5]
        result=m.fresh_rd_gate(a,a,dict(h264=-.3,h265=-.3),dict(h264=2,h265=3),math.log(.85))
        self.assertTrue(result['eligible_for_test']); self.assertFalse(result['objective_is_bd_rate'])
        self.assertEqual(a['h264']['quality'],[.8,.79,.81,.5]); self.assertNotIn('point_passes',result)

    def test_incomplete_nonfinite_or_unfrozen_scales_cannot_open_test(self):
        m=module(); a=self.curves()
        with self.assertRaises(ValueError): m.fresh_rd_gate(a,a,dict(h264=-.3,h265=-.3),dict(h264=float('nan'),h265=3),math.log(.85))
        a['h264']['quality']=[.8,.7,.6]
        with self.assertRaises(ValueError): m.fresh_rd_gate(a,a,dict(h264=-.3,h265=-.3),dict(h264=2,h265=3),math.log(.85))


if __name__=='__main__': unittest.main()
