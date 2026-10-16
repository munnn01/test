"""Handmade grouping and point-based scale decisions only."""
import importlib.util
from pathlib import Path
import unittest


def module():
    path=Path(__file__).resolve().parents[1]/'adaptive_vcm/v34r1_a6/contracts.py'
    if not path.is_file(): raise AssertionError('A6 cohort/gate contracts missing')
    spec=importlib.util.spec_from_file_location('a6_contracts',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def bd(v):
    return dict(valid=True,pchip_bd_rate_pct=v,qps=[30,35,40,45],primary_interpolator='PCHIP',threshold_pct=-15.,invalid_reason=None)


def complete_blocks():
    return dict(grid=True,policy_lock=True,source_provenance=True,paid_header=True,
        independent_decode=True,source_only=True,no_substitution=True,measurement_reproducibility=True)


class A6Contracts(unittest.TestCase):
    def test_withhold_whole_cohort_without_any_image_overlap(self):
        folds=module().cohort_folds({'a':[1,2],'b':[3,4],'c':[5,6]},expected_size=2,forbidden=[7,8])
        self.assertEqual(len(folds),3)
        for fold in folds:
            self.assertEqual(len(fold['train_ids']),4);self.assertEqual(len(fold['heldout_ids']),2)
            self.assertFalse(set(fold['train_ids'])&set(fold['heldout_ids']))
        self.assertEqual(folds[0]['heldout_ids'],[1,2])

    def test_overlaps_forbidden_ids_and_incomplete_cohorts_fail(self):
        f=module().cohort_folds
        for groups,forbidden in [({'a':[1,2],'b':[2,4],'c':[5,6]},[]),({'a':[1,2],'b':[3,4],'c':[5,6]},[6]),
            ({'a':[1,2],'b':[3],'c':[5,6]},[]),({'a':[1,2],'b':[3,4]},[])]:
            with self.assertRaises(ValueError):f(groups,expected_size=2,forbidden=forbidden)

    def test_scale_gate_requires_both_strict_bds_and_all_blocks(self):
        f=module().scale4_gate;r={'h264':bd(-15.01),'h265':bd(-16.)}
        self.assertTrue(f(r,complete_blocks())['eligible_for_scale4'])
        r['h264']=bd(-15.);self.assertFalse(f(r,{'grid':True})['eligible_for_scale4'])
        r['h264']=bd(-20.);self.assertFalse(f(r,{'grid':False})['eligible_for_scale4'])
        self.assertFalse(f(r,{})['eligible_for_scale4'])

    def test_every_required_block_is_mandatory(self):
        f=module().scale4_gate;r={'h264':bd(-20.),'h265':bd(-20.)}
        for omitted in complete_blocks():
            blocks=complete_blocks();del blocks[omitted]
            self.assertFalse(f(r,blocks)['eligible_for_scale4'],omitted)
        self.assertFalse(f(r,{'grid':True})['eligible_for_scale4'])

    def test_invalid_or_repaired_tables_never_open_expansion(self):
        f=module().scale4_gate
        for bad in [dict(bd(-20.),valid=False),dict(bd(-20.),pchip_bd_rate_pct=None),dict(bd(-20.),pchip_bd_rate_pct=float('nan')),
            dict(bd(-20.),qps=[30,35,40]),dict(bd(-20.),primary_interpolator='repaired')]:
            self.assertFalse(f({'h264':bad,'h265':bd(-20.)},{'grid':True})['eligible_for_scale4'])
        self.assertFalse(f({'h264':bd(-20.)},{'grid':True})['eligible_for_scale4'])


if __name__=='__main__':unittest.main()
