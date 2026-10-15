"""Cloud restoration must survive transient transport errors without replacing data."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import requests

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_v34r1_cloud_data as restore


class CloudDataTests(unittest.TestCase):
    def test_transient_http_and_timeout_retry_same_frozen_file(self):
        raw=b'exact-frozen-source'
        failed=Mock(status_code=404)
        failed.raise_for_status.side_effect=requests.HTTPError('404 from provider')
        valid=Mock(status_code=200,content=raw)
        with tempfile.TemporaryDirectory() as directory:
            destination=Path(directory)/'img_00001.jpg'
            with patch.object(restore,'mounted',return_value=None), patch.object(restore.time,'sleep'), \
                 patch.object(restore.requests,'post',side_effect=[failed,requests.Timeout('transient'),valid]) as post:
                restore.ar_frame(('frozen-id','img_00001.jpg',destination,hashlib.sha256(raw).hexdigest()))
            self.assertEqual(destination.read_bytes(),raw)
            self.assertEqual(post.call_count,3)
            self.assertEqual({c.kwargs['json']['fileName'] for c in post.call_args_list},{'frozen-id/img_00001.jpg'})

    def test_persistent_http_failure_names_source_and_writes_no_replacement(self):
        failed=Mock(status_code=404)
        failed.raise_for_status.side_effect=requests.HTTPError('404 from provider')
        with tempfile.TemporaryDirectory() as directory:
            destination=Path(directory)/'img_00001.jpg'
            with patch.object(restore,'mounted',return_value=None), patch.object(restore.time,'sleep'), \
                 patch.object(restore.requests,'post',return_value=failed):
                with self.assertRaisesRegex(RuntimeError,'frozen-id/img_00001.jpg'):
                    restore.ar_frame(('frozen-id','img_00001.jpg',destination,'0'*64))
            self.assertFalse(destination.exists())
