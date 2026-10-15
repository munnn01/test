import unittest
from itertools import product
from adaptive_vcm.v34r1.audit import audit_grid
from adaptive_vcm.v34r1.audit import verify_source
import hashlib
from pathlib import Path
import tempfile


class GridTests(unittest.TestCase):
    def rows(self):
        return [dict(id=i,codec=c,qp=q,arm=a,bytes=12,elementary_bytes=10,metadata_bytes=2,
                     packet_sha256='a'*64,decoded_sha256='b'*64)
                for i,c,q,a in product(('one','two'),('h264','h265'),(30,35,40,45),('anchor','adaptive'))]

    def test_missing_duplicate_and_unexpected_measurements_fail(self):
        rows=self.rows()
        self.assertTrue(audit_grid(rows,source_ids=['one','two'],arms=['anchor','adaptive'])['complete'])
        for corrupt in [rows[:-1],rows+[rows[0]],rows+[dict(rows[0],qp=50)]]:
            with self.assertRaises(ValueError): audit_grid(corrupt,source_ids=['one','two'],arms=['anchor','adaptive'])

    def test_bad_accounting_and_duplicate_sources_fail(self):
        rows=self.rows(); rows[0]['bytes']=13
        with self.assertRaises(ValueError): audit_grid(rows,source_ids=['one','two'],arms=['anchor','adaptive'])
        with self.assertRaises(ValueError): audit_grid(self.rows(),source_ids=['one','one'],arms=['anchor','adaptive'])

    def test_frozen_source_bytes_must_match_and_paths_stay_inside_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'frame.jpg'; source.write_bytes(b'original')
            sha=hashlib.sha256(source.read_bytes()).hexdigest()
            verify_source({'path':str(root),'file_sha256':{'frame.jpg':sha}})
            source.write_bytes(b'modified')
            with self.assertRaises(ValueError): verify_source({'path':str(source),'file_sha256':sha})
            with self.assertRaises(ValueError): verify_source({'path':str(root),'file_sha256':{'../outside.jpg':sha}})
            with self.assertRaises(ValueError): verify_source({'path':str(source)})

    def test_unlisted_ar_jpeg_cannot_shift_the_selected_frame_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'img_00001.jpg').write_bytes(b'frame')
            sha=hashlib.sha256(b'frame').hexdigest()
            item=dict(path=str(root),kind='frames',provider_window_files=['img_00001.jpg'],file_sha256={'img_00001.jpg':sha})
            verify_source(item)
            (root/'img_00000.jpg').write_bytes(b'extra')
            with self.assertRaises(ValueError): verify_source(item)
            ot=dict(path=str(root),sequence='MOT20-01',frames=16,file_sha256={'img_00001.jpg':sha})
            with self.assertRaises(ValueError): verify_source(ot)
