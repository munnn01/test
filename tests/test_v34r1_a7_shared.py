"""Handmade cross-QP tables: no real task computations."""
import unittest
from adaptive_vcm.v34r1_a7.shared import choose_shared,select_valid_fit


class Shared(unittest.TestCase):
    def test_same_action_for_all_qps_and_paid_cap_on_every_qp(self):
        rates=[[1,.5,.8],[1,.5,.8],[1,.5,.8],[1,1.01,.8]];costs=[[.1,.1,.1]]*4
        action,audit=choose_shared(rates,costs,32,[0,1,2])
        self.assertEqual(action,2);self.assertEqual(audit['eligible_actions'],[0,2])
        self.assertEqual(audit['shared_across_qps'],[30,35,40,45])

    def test_identity_for_empty_reference_and_risk_sensitive_choice(self):
        rates=[[1,.5]]*4
        self.assertEqual(choose_shared(rates,[[None,None]]*4,32,[0,1])[0],0)
        self.assertEqual(choose_shared(rates,[[.1,.5]]*4,32,[0,1])[0],0)
        self.assertEqual(choose_shared(rates,[[.1,.5]]*4,0,[0,1])[0],1)
        with self.assertRaises(ValueError):choose_shared(rates[:3],[[.1,.5]]*3,1,[0,1])

    def test_fit_choice_rejects_invalid_curve_despite_better_utility(self):
        anchor=dict(rate=[4,3,2,1],quality=[.8,.7,.6,.5])
        choices=[dict(weight=0,quality=[.9,.8,.7,.6],log_rate_ratio=-.5,bd_valid=False),
            dict(weight=1,quality=[.8,.7,.6,.5],log_rate_ratio=-.1,bd_valid=True)]
        self.assertEqual(select_valid_fit(anchor,choices)['selected']['weight'],1)
        self.assertIsNone(select_valid_fit(anchor,choices[:1])['selected'])


if __name__=='__main__':unittest.main()
