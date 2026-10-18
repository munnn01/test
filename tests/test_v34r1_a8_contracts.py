"""Synthetic source allocation and immutable four-QP expansion contracts."""
import unittest


class Expansion(unittest.TestCase):
    def module(self):
        from adaptive_vcm.v34r1_a8 import contracts
        return contracts

    def test_stratification_excludes_seen_and_shards_cover_exactly_once(self):
        m=self.module();items=[dict(id=str(i),label=i//6) for i in range(24)]
        rows=m.stratified(items,10,{str(0)},seed='fixture')
        self.assertEqual(len(rows),10);self.assertNotIn('0',[r['id'] for r in rows])
        counts=[sum(r['label']==c for r in rows) for c in range(4)]
        self.assertEqual(sorted(counts),[2,2,3,3])
        pieces=[m.shard(rows,i,4) for i in range(4)]
        self.assertEqual(sorted(r['id'] for p in pieces for r in p),sorted(r['id'] for r in rows))
        with self.assertRaises(ValueError):m.stratified(items,25,set(),seed='fixture')
        with self.assertRaises(ValueError):m.shard(rows,4,4)

    def test_exact_grid_detects_missing_duplicate_and_unpaid_conditions(self):
        m=self.module();rows=[dict(id=n,codec=c,qp=q,arm=a,bytes=50,elementary_bytes=10,metadata_bytes=40)
            for n in ['a','b'] for c in ('h264','h265') for q in (30,35,40,45) for a in ('anchor','adaptive')]
        self.assertEqual(m.audit_selected(rows,['a','b']),32)
        for broken in (rows[:-1],rows+[rows[0]],[dict(x,bytes=49) for x in rows]):
            with self.assertRaises(ValueError):m.audit_selected(broken,['a','b'])

    def test_ot_plan_is_native_and_uses_new_physical_videos(self):
        m=self.module();s=dict(name='dancetrack0081',frameRate='20',seqLength='600',imWidth='1920',imHeight='1080',imDir='img1',imExt='.jpg')
        self.assertEqual(m.native_plan(s,256)['fps'],20)
        with self.assertRaises(ValueError):m.native_plan(dict(s,seqLength='100'),256)
        with self.assertRaises(ValueError):m.native_plan(dict(s,frameRate='0'),256)


if __name__=='__main__':unittest.main()
