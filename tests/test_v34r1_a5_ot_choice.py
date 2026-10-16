"""Handmade paid-rate/source-HOTA tables only; no real tracker or GT."""
import importlib.util
import math
from pathlib import Path
import unittest


def module():
    path=Path(__file__).resolve().parents[1]/'adaptive_vcm/v34r1_a5/ot_choice.py'
    if not path.is_file(): raise AssertionError('A5 paid-rate/HOTA tradeoff selector missing')
    spec=importlib.util.spec_from_file_location('a5_ot_choice',path)
    value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


class OTChoiceContracts(unittest.TestCase):
    def test_soft_tradeoff_can_accept_five_point_proxy_loss_then_reject_at_larger_lambda(self):
        choose=module().choose_lagrangian
        rates=[1.,.7,.6]; costs=[.2,.25,.5]; actions=[0,13,20]
        index,audit=choose(rates,costs,4.,actions)
        self.assertEqual(index,1)
        self.assertAlmostEqual(audit['objective'][1],math.log(.7)+4*.05)
        self.assertEqual(choose(rates,costs,8.,actions)[0],0)
        self.assertEqual(choose(rates,costs,0.,actions)[0],2)
        self.assertFalse(audit['objective_is_bd_rate'])

    def test_paid_cap_empty_source_and_action_id_ties(self):
        choose=module().choose_lagrangian
        self.assertEqual(choose([1.,.8,.8,1.01],[.2,.2,.2,0.],1.,[0,9,3,20])[0],2)
        self.assertEqual(choose([1.,.5],[None,None],0.,[0,20])[0],0)
        self.assertEqual(choose([1.,1.],[.2,.2],1.,[0,20])[0],0)
        self.assertEqual(choose([1.,1.],[.2,.1],1.,[0,20])[0],1)

    def test_invalid_or_unaligned_tables_fail_closed(self):
        choose=module().choose_lagrangian
        for rate,cost,weight,actions in [([1.],[.2],-1,[0]),([1.],[.2],float('nan'),[0]),
            ([.9],[.2],1,[0]),([1.,0.],[.2,.2],1,[0,1]),([1.,.8],[.2,None],1,[0,1]),
            ([1.,.8],[.2,1.1],1,[0,1]),([1.,.8],[.2,.1],1,[0,0]),([1.],[.2],1,[2])]:
            with self.assertRaises(ValueError): choose(rate,cost,weight,actions)


if __name__=='__main__': unittest.main()
