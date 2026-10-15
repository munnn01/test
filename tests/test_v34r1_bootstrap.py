import unittest
from adaptive_vcm.v34r1.metrics import paired_ar_ci


class BootstrapTests(unittest.TestCase):
    def test_matched_curves_retain_known_rate_saving_in_every_finite_draw(self):
        rows=[]
        for i in range(20):
            for qp,quality_limit,rate in zip((30,35,40,45),(20,15,10,5),(4.,3.,2.,1.)):
                for arm,ratio in [('anchor',1.),('adaptive',.8)]:
                    rows.append(dict(id=str(i),codec='h264',qp=qp,arm=arm,bpp=rate*ratio,correct=int(i<quality_limit)))
        ci=paired_ar_ci(rows,'h264',draws=200,seed=7)
        self.assertAlmostEqual(ci['lo'],-20.,places=8); self.assertAlmostEqual(ci['hi'],-20.,places=8)
        self.assertEqual(ci['independent_unit'],'video'); self.assertEqual(ci['independent_count'],20)
        self.assertGreater(ci['finite_fraction'],.9)
        with self.assertRaises(ValueError): paired_ar_ci(rows[:-1],'h264',draws=10)

    def test_invalid_draws_cannot_disappear_from_upper_tail(self):
        rows=[dict(id=str(i),codec='h264',qp=q,arm=a,bpp=(51-q)*ratio,correct=1)
              for i in range(4) for q in (30,35,40,45) for a,ratio in [('anchor',1),('adaptive',.8)]]
        ci=paired_ar_ci(rows,'h264',draws=20)
        self.assertIsNone(ci['hi']); self.assertIsNone(ci['lo'])
        self.assertEqual(ci['finite_fraction'],0)
        self.assertEqual(ci['invalid_draw_policy'],'worst-case-unbounded')

    def test_mixed_invalid_mass_above_familywise_alpha_keeps_upper_bound_unbounded(self):
        rows=[]
        for i in range(10):
            for qp,limit,rate in zip((30,35,40,45),(10,6,4,3),(4.,3.,2.,1.)):
                for arm,ratio in [('anchor',1.),('adaptive',.8)]:
                    rows.append(dict(id=str(i),codec='h264',qp=qp,arm=arm,bpp=rate*ratio,correct=int(i<limit)))
        ci=paired_ar_ci(rows,'h264',draws=2000,seed=7)
        self.assertGreater(ci['finite_fraction'],.95)
        self.assertLess(ci['finite_fraction'],1-.05/6)
        self.assertIsNone(ci['hi'])
