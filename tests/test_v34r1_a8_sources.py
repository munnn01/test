"""Synthetic bytes only: restore the exact declared AR window inventory."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os


class SourceRestore(unittest.TestCase):
    def test_full_provider_directory_is_reduced_to_declared_window(self):
        root=Path(__file__).resolve().parents[1]
        import sys
        sys.path.insert(0,str(root/'scripts'))
        import prepare_v34r1_a8_sources as m
        with tempfile.TemporaryDirectory(dir=os.environ.get('V34_AUDIT_TMP_DIR')) as tmp:
            base=Path(tmp);folder=base/'provider/video';folder.mkdir(parents=True)
            for i in range(1,5):(folder/f'img_{i:05d}.jpg').write_bytes(bytes([i]))
            hashes={f'img_{i:05d}.jpg':hashlib.sha256(bytes([i])).hexdigest() for i in (2,3)}
            row=dict(id='video',file_sha256=hashes,provider_window_files=list(hashes))
            with patch.object(m,'ROOT',base/'repo'),patch.object(m,'mounted',return_value=folder/'img_00001.jpg'):
                restored=m.restore_ar(row)
            actual=Path(restored['path'])
            self.assertNotEqual(actual,folder);self.assertEqual({p.name for p in actual.iterdir()},set(hashes))


if __name__=='__main__':unittest.main()
