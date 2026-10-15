"""Synthetic official-COCO fixtures only; no task model or media inference."""
import importlib.util
from pathlib import Path
import unittest


def proxy():
    path=Path(__file__).resolve().parents[1]/'adaptive_vcm/v34r1_a3/pseudo.py'
    if not path.is_file(): raise AssertionError('A3 pseudo-AP block is missing')
    spec=importlib.util.spec_from_file_location('a3_pseudo',path)
    result=importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


class PseudoAPContracts(unittest.TestCase):
    def test_class_aware_localization_and_score_ordered_false_positives(self):
        m=proxy(); source=dict(boxes=[[0,0,10,10]],labels=[1],scores=[.9])
        self.assertAlmostEqual(m.pseudo_damage(source,source),0,places=12)
        wrong=dict(boxes=[[0,0,10,10]],labels=[2],scores=[.9])
        self.assertAlmostEqual(m.pseudo_damage(source,wrong),1,places=12)
        fp=dict(boxes=[[20,20,30,30],[0,0,10,10]],labels=[1,1],scores=[.99,.9])
        self.assertGreater(m.pseudo_damage(source,fp),.49)
        self.assertIsNone(m.pseudo_damage(dict(boxes=[],labels=[],scores=[]),source))

    def test_source_threshold_and_top_target_cap_are_explicit(self):
        m=proxy(); source=dict(boxes=[[0,0,10,10],[20,20,30,30]],labels=[1,1],scores=[.9,.2])
        decoded=dict(boxes=[[0,0,10,10]],labels=[1],scores=[.9])
        self.assertAlmostEqual(m.pseudo_damage(source,decoded,threshold=.25,topk=100),0,places=12)
        self.assertGreater(m.pseudo_damage(source,decoded,threshold=.1,topk=100),.49)


if __name__=='__main__': unittest.main()
