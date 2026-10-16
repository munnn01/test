"""Handmade event rows: actual decodes differ from deduplicated task forwards."""
import importlib.util
from pathlib import Path
import unittest


def module():
    path=Path(__file__).resolve().parents[1]/'adaptive_vcm/v34r1_a4/sender_costs.py'
    if not path.exists(): raise AssertionError('actual sender cost counters missing')
    spec=importlib.util.spec_from_file_location('a4_sender_costs',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class SenderCosts(unittest.TestCase):
    def test_independent_stream_decodes_are_not_detector_cache_hits(self):
        m=module(); rows=[dict(packet_reused=False,decoded_reused=False),dict(packet_reused=False,decoded_reused=True),dict(packet_reused=True,decoded_reused=True)]
        v=m.measurement_counts(rows,source_calls=1,controller_calls=8,repeat_roundtrips=1,repeat_detector_calls=1)
        self.assertEqual(v['actual_encodes'],3); self.assertEqual(v['actual_decodes'],3)
        self.assertEqual(v['decoded_task_model_forwards'],2); self.assertEqual(v['source_task_model_forwards'],1)
        self.assertEqual(v['a2_controller_forwards'],8)


if __name__=='__main__': unittest.main()
