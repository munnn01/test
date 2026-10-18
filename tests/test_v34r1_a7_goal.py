"""Artificial comparison records for the user-amended one-codec resource gate."""
import unittest
from adaptive_vcm.v34r1_a6.contracts import one_codec_gate,REQUIRED_SCALE_BLOCKS


def bd(value):
    return dict(qps=[30,35,40,45],primary_interpolator='PCHIP',threshold_pct=-15.,valid=True,
                invalid_reason=None,pchip_bd_rate_pct=value)


class Goal(unittest.TestCase):
    def test_one_valid_codec_suffices_and_invalid_record_is_preserved(self):
        bad=dict(bd(None),valid=False,invalid_reason='quality is not monotone with rate')
        rows=dict(h264=bad,h265=bd(-16.49));blocks=dict.fromkeys(REQUIRED_SCALE_BLOCKS,True)
        result=one_codec_gate(rows,blocks)
        self.assertTrue(result['eligible_for_scale4']);self.assertEqual(result['passed_codecs'],['h265'])
        self.assertIsNone(rows['h264']['pchip_bd_rate_pct'])

    def test_strict_negative_threshold_complete_reporting_and_blocks(self):
        rows=dict(h264=bd(-15.),h265=bd(-14.));blocks=dict.fromkeys(REQUIRED_SCALE_BLOCKS,True)
        self.assertFalse(one_codec_gate(rows,blocks)['eligible_for_scale4'])
        rows['h264']=bd(-20.)
        for missing in REQUIRED_SCALE_BLOCKS:
            incomplete=dict(blocks);del incomplete[missing]
            self.assertFalse(one_codec_gate(rows,incomplete)['eligible_for_scale4'])
        self.assertFalse(one_codec_gate({'h264':bd(-20.)},blocks)['eligible_for_scale4'])
        rows['h264']=bd(float('nan'));self.assertFalse(one_codec_gate(rows,blocks)['eligible_for_scale4'])


if __name__=='__main__':unittest.main()
