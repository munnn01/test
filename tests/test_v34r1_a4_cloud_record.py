"""CPU fixtures: failed cloud setup leaves evidence without overwriting old runs."""
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import os


def module():
    p=Path(__file__).resolve().parents[1]/'adaptive_vcm/v34r1_a4/cloud_record.py'
    if not p.exists(): raise AssertionError('failure-recording cloud wrapper missing')
    spec=importlib.util.spec_from_file_location('a4_cloud_record',p); value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


class CloudRecordContracts(unittest.TestCase):
    def temp(self):
        base=os.environ.get('V34_AUDIT_TMP_DIR')
        if base: Path(base).mkdir(parents=True,exist_ok=True)
        return tempfile.TemporaryDirectory(dir=base)

    def test_failed_setup_records_error_and_available_logs_without_masking_original(self):
        m=module()
        with self.temp() as temp:
            root=Path(temp); log=root/'blocks.log'; log.write_text('synthetic contracts failed')
            with self.assertRaisesRegex(RuntimeError,'parent hash mismatch'):
                with m.record_cloud_run(root/'result',root/'result.tgz',dict(commit='a'*40,test_scored=False),logs=[log,root/'missing.log']):
                    raise RuntimeError('parent hash mismatch')
            status=json.loads((root/'result/cloud-status.json').read_text())
            self.assertFalse(status['complete']); self.assertEqual(status['error_type'],'RuntimeError')
            self.assertTrue((root/'result/blocks.log').exists())
            with tarfile.open(root/'result.tgz') as tar: self.assertIn('result/cloud-status.json',tar.getnames())

    def test_existing_output_is_preserved_and_success_omits_gt_from_bundle(self):
        m=module()
        with self.temp() as temp:
            root=Path(temp); base=root/'result'
            with m.record_cloud_run(base,root/'result.tgz',dict(test_scored=False)):
                (base/'ground_truth').mkdir(parents=True); (base/'ground_truth/gt.txt').write_text('not for export')
            before=(base/'cloud-status.json').read_bytes()
            with self.assertRaises(ValueError):
                with m.record_cloud_run(base,root/'result.tgz',dict(test_scored=False)): pass
            self.assertEqual(before,(base/'cloud-status.json').read_bytes())
            with tarfile.open(root/'result.tgz') as tar: self.assertFalse(any('ground_truth' in v for v in tar.getnames()))
            self.assertTrue(json.loads(before)['complete'])

    def test_tracker_parity_rejects_changed_association_or_missing_output(self):
        m=module()
        with self.temp() as temp:
            root=Path(temp); a=root/'a'; b=root/'b'; a.mkdir(); b.mkdir()
            (a/'seq.txt').write_text('1,7,0,0,10,10,.9,-1,-1,-1\n')
            (b/'seq.txt').write_text('1,8,0,0,10,10,.9,-1,-1,-1\n')
            self.assertTrue(hasattr(m,'assert_same_files'),'exact tracker-byte parity check missing')
            with self.assertRaises(ValueError): m.assert_same_files(a,b)
            (b/'seq.txt').write_bytes((a/'seq.txt').read_bytes()); m.assert_same_files(a,b)
            (b/'extra.txt').write_text('unexpected')
            with self.assertRaises(ValueError): m.assert_same_files(a,b)

    def test_changed_fresh_reservation_cannot_be_silently_rebound(self):
        import hashlib
        m=module(); raw=b'{"fresh_ids":[1,2]}'
        self.assertTrue(hasattr(m,'require_sha256'),'frozen input binding missing')
        expected=hashlib.sha256(raw).hexdigest()
        self.assertEqual(m.require_sha256(raw,expected),expected)
        with self.assertRaises(ValueError): m.require_sha256(b'{"fresh_ids":[3,4]}',expected)


if __name__=='__main__': unittest.main()
