"""Synthetic source-track HOTA fixtures, without a detector or codec workload."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
import numpy as np


def block():
    path=Path(__file__).resolve().parents[1]/'adaptive_vcm/v34r1_a3/hota_proxy.py'
    if not path.is_file(): raise AssertionError('source-track HOTA adapter is missing')
    spec=importlib.util.spec_from_file_location('a3_hota',path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def metric():
    source=Path(os.environ.get('V34_TRACKEVAL','D:/TEST/data/TrackEval'))
    sys.path.insert(0,str(source))
    if 'float' not in np.__dict__: np.float=float
    from trackeval.metrics.hota import HOTA
    return HOTA()


def frame(number,ids):
    return dict(frame_number=number,track_ids=np.array(ids,int),boxes=np.array([[0,0,10,10] for _ in ids],float).reshape(-1,4))


class HOTASourceContracts(unittest.TestCase):
    def test_id_relabeling_is_invariant_but_a_switch_lowers_association(self):
        m=block(); h=metric(); source=[frame(1,[5]),frame(2,[5])]
        same=[frame(1,[99]),frame(2,[99])]
        score=m.source_hota(source,same,h)
        self.assertAlmostEqual(score['HOTA'],1); self.assertAlmostEqual(score['AssA'],1)
        switched=[frame(1,[99]),frame(2,[100])]
        score=m.source_hota(source,switched,h)
        self.assertLess(score['HOTA'],1); self.assertLess(score['AssA'],1)
        self.assertAlmostEqual(score['DetA'],1)
        self.assertAlmostEqual(m.source_hota(source,switched,h,lookback=1)['HOTA'],1)

    def test_empty_source_is_undefined_and_temporal_or_duplicate_ids_are_rejected(self):
        m=block(); h=metric()
        self.assertIsNone(m.source_hota([frame(1,[])],[frame(1,[2])],h))
        with self.assertRaises(ValueError): m.source_hota([frame(1,[1])],[frame(2,[1])],h)
        with self.assertRaises(ValueError): m.source_hota([frame(1,[1,1])],[frame(1,[1])],h)
        with self.assertRaises(ValueError): m.source_hota([frame(1,[1])],[],h)

    def test_long_prefix_dilutes_recent_misses_but_trailing_window_retains_them(self):
        m=block(); h=metric(); source=[frame(i,[5]) for i in range(1,513)]
        candidate=[frame(i,[99] if i<=496 else []) for i in range(1,513)]
        prefix=m.source_hota(source,candidate,h)
        trailing=m.source_hota(source,candidate,h,lookback=32)
        self.assertLess(1-prefix['HOTA'],.04)
        self.assertGreater(1-trailing['HOTA'],.4)


if __name__=='__main__': unittest.main()
