"""Synthetic bytes/calls: preserve wires and account repeated model calls."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import os


class FakeModel:
    def register_forward_hook(self,fn):self.fn=fn;return self
    def remove(self):pass
    def run(self,x):self.fn(self,(x,),None)


class Evidence(unittest.TestCase):
    def test_packet_bytes_are_verified_and_reused_by_content_hash(self):
        from adaptive_vcm.v34r1_a8.evidence import packet_file
        with tempfile.TemporaryDirectory(dir=os.environ.get('V34_AUDIT_TMP_DIR')) as tmp:
            raw=b'handmade wire';digest=hashlib.sha256(raw).hexdigest();base=Path(tmp)
            name=packet_file(base,raw,digest,len(raw));self.assertEqual((base/name).read_bytes(),raw)
            self.assertEqual(packet_file(base,raw,digest,len(raw)),name)
            with self.assertRaises(ValueError):packet_file(base,raw,digest,len(raw)+1)
            with self.assertRaises(ValueError):packet_file(base,raw,'0'*64,len(raw))

    def test_meter_counts_actual_calls_units_and_repeats_separately(self):
        from adaptive_vcm.v34r1_a8.evidence import ForwardMeter
        model=FakeModel();meter=ForwardMeter();meter.attach('detector',model)
        meter.stage('source');model.run([1,2,3]);meter.stage('repeat/receiver');model.run([1,2])
        self.assertEqual(meter.counts['source/detector_calls'],1);self.assertEqual(meter.counts['source/detector_units'],3)
        self.assertEqual(meter.counts['repeat/receiver/detector_units'],2)


if __name__=='__main__':unittest.main()
