"""Prospective strict target and four-QP contracts; runs without torch or datasets."""
import importlib
import importlib.util
import unittest
import numpy as np


def implementation():
    if importlib.util.find_spec('adaptive_vcm.v34r1') is None:
        raise AssertionError('v34r1 strict target implementation is missing')
    return importlib.import_module('adaptive_vcm.v34r1.metrics')


class TargetContracts(unittest.TestCase):
    def curves(self, ratio=.84):
        return {'anchor': {'rate': [1., 2., 3., 4.], 'quality': [.5, .6, .7, .8]},
                'adaptive': {'rate': (np.array([1., 2., 3., 4.])*ratio).tolist(), 'quality': [.5, .6, .7, .8]}}

    def test_strict_minus_fifteen_boundary(self):
        m = implementation()
        for ratio, expected in [(1., False), (.851, False), (.85, False), (.849, True)]:
            c = self.curves(ratio)
            value = m.assess_curve(c['anchor'], c['adaptive'], qps=[30,35,40,45])
            self.assertEqual(value['point_passes'], expected, (ratio, value))

    def test_missing_task_or_codec_cannot_pass(self):
        m = implementation()
        endpoints = {key: {'valid': True, 'point_passes': True, 'threshold_confirmed': True} for key in m.ENDPOINTS}
        self.assertTrue(m.assess_all(endpoints)['all_point_targets'])
        del endpoints['ot/h265']
        report = m.assess_all(endpoints)
        self.assertFalse(report['all_point_targets'])
        self.assertIn('ot/h265', report['missing_endpoints'])

    def test_point_estimate_does_not_prove_confidence_threshold(self):
        m = implementation(); c = self.curves()
        result = m.assess_curve(c['anchor'], c['adaptive'], qps=[30,35,40,45],
                                ci={'hi': -1., 'finite_fraction': 1.})
        self.assertTrue(result['point_passes'])
        self.assertFalse(result['threshold_confirmed'])

    def test_nonmonotone_curve_and_wrong_qp_grid_fail_closed(self):
        m = implementation(); c = self.curves()
        c['adaptive']['quality'] = [.5,.7,.6,.8]
        self.assertFalse(m.assess_curve(c['anchor'],c['adaptive'],qps=[30,35,40,45])['valid'])
        with self.assertRaises(ValueError):
            m.assess_curve(c['anchor'],c['adaptive'],qps=[30,35,40,50])

    def test_no_common_quality_interval_is_invalid(self):
        m = implementation(); c = self.curves()
        c['adaptive']['quality'] = [.9,.91,.92,.93]
        result = m.assess_curve(c['anchor'],c['adaptive'],qps=[30,35,40,45])
        self.assertFalse(result['valid'])
        self.assertIsNone(result['pchip_bd_rate_pct'])

    def test_conditional_finite_draw_interval_cannot_certify_threshold(self):
        m=implementation(); c=self.curves()
        ci=dict(hi=-20.,draws=5000,finite_fraction=.95,familywise_endpoints=6,
                independent_unit='video',independent_count=128,one_sided_confidence=1-.05/6)
        result=m.assess_curve(c['anchor'],c['adaptive'],qps=[30,35,40,45],ci=ci)
        self.assertFalse(result['threshold_confirmed'])


if __name__ == '__main__':
    unittest.main()
