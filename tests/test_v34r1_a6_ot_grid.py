"""Small artificial OT rows: no actual images, models, codecs or tracking."""
import unittest
from adaptive_vcm.v34r1_a6.contracts import audit_ot21


def rows():
    return [dict(sequence='s',split='fit',start=t,codec=c,qp=q,action=a,bytes=140,
                 elementary_bytes=100,metadata_bytes=40)
            for t in (1,17,33,49) for c in ('h264','h265') for q in (30,35,40,45) for a in range(21)]


class TestOTGrid(unittest.TestCase):
    def test_full_bank_and_codec_shards(self):
        self.assertEqual(audit_ot21(rows(),{'s':'fit'}),672)
        for c in ('h264','h265'):
            self.assertEqual(audit_ot21([r for r in rows() if r['codec']==c],{'s':'fit'},codec=c),336)

    def test_reject_partial_duplicate_and_wrong_condition(self):
        values=rows()
        for bad in (values[:-1],values+[values[0]],values+[dict(values[0],qp=50)],
                    [dict(r,split='validation') for r in values],
                    [dict(r,bytes=100) for r in values]):
            with self.assertRaises(ValueError):audit_ot21(bad,{'s':'fit'})


if __name__=='__main__':unittest.main()
