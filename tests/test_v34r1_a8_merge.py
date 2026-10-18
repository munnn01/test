"""Synthetic incomplete-shard rejection and official nonlinear HOTA pooling."""
import unittest
import math
from tests.test_v34r1_a3_hota import metric


class Merge(unittest.TestCase):
    def test_partial_receipts_wrong_counts_and_other_commits_are_rejected(self):
        from adaptive_vcm.v34r1_a8.merge import validate_receipt
        tasks={n:dict(complete=True) for n in ('ar','od','ot')}
        status=dict(complete=True,commit='a'*40,shard=0,source_manifest_sha256='b'*64)
        complete=dict(complete=True,shard=0,ar=250,od=250,ot=3,tasks=tasks)
        self.assertEqual(validate_receipt(status,tasks,complete,'a'*40,'b'*64),0)
        for s,c in [(dict(status,complete=False),complete),(status,dict(complete,od=249)),(dict(status,commit='c'*40),complete)]:
            with self.assertRaises(ValueError):validate_receipt(s,tasks,c,'a'*40,'b'*64)

    def test_shards_must_match_prespecified_identity_partition(self):
        from adaptive_vcm.v34r1_a8.merge import check_partition
        expected=list('abcdefgh');parts={i:[r for j,r in enumerate(expected) if j%4==i] for i in range(4)}
        self.assertEqual(check_partition(parts,expected),8)
        for broken in ({i:v for i,v in parts.items() if i!=3},dict(parts,**{})):
            if len(broken)==4:broken[0]=parts[1]
            with self.assertRaises(ValueError):check_partition(broken,expected)

    def test_official_pooled_hota_weights_raw_stats_not_sequence_means(self):
        from adaptive_vcm.v34r1_a8.merge import pool_hota
        def row(tp,fn,ass):return dict(HOTA_TP=[tp]*19,HOTA_FN=[fn]*19,HOTA_FP=[0]*19,AssA=[ass]*19,AssRe=[ass]*19,AssPr=[ass]*19,LocA=[1.]*19)
        actual=pool_hota(metric(),{'a':row(4,0,1.),'b':row(1,9,.25)})
        expected=math.sqrt((5/14)*.85)
        self.assertAlmostEqual(actual['HOTA'],expected)
        self.assertNotAlmostEqual(actual['HOTA'],(1+math.sqrt(.1*.25))/2)


if __name__=='__main__':unittest.main()
