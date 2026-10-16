"""Publication integrity rejects changed, escaping and undeclared evidence bytes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import contextmanager


@contextmanager
def temporary_evidence():
    base=Path(os.environ.get('V34_AUDIT_TMP_DIR',tempfile.gettempdir())).resolve()
    with tempfile.TemporaryDirectory(dir=base) as folder:
        root=Path(folder).resolve()
        if not root.is_relative_to(base): raise ValueError('temporary evidence path escapes its base')
        yield root


def verifier():
    path=Path(__file__).resolve().parents[1]/'scripts/verify_v34r1_a3.py'
    if not path.is_file(): raise AssertionError('A3 evidence verifier missing')
    spec=importlib.util.spec_from_file_location('a3_evidence',path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class EvidenceIntegrity(unittest.TestCase):
    def test_changed_and_undeclared_files_are_rejected(self):
        m=verifier()
        with temporary_evidence() as root:
            raw=b'exact evidence\n'; (root/'data.json').write_bytes(raw)
            obj=dict(files=[dict(path='data.json',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())])
            (root/'publication-manifest.json').write_text(json.dumps(obj))
            self.assertEqual(m.verify_files(root),obj)
            (root/'extra.json').write_text('{}')
            with self.assertRaises(ValueError): m.verify_files(root)
            (root/'extra.json').unlink(); (root/'data.json').write_bytes(raw+b'\n')
            with self.assertRaises(ValueError): m.verify_files(root)

    def test_path_escape_is_rejected_before_reading(self):
        m=verifier()
        with temporary_evidence() as root:
            obj=dict(files=[dict(path='../outside.json',bytes=1,sha256='a'*64)])
            (root/'publication-manifest.json').write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError,'path'): m.verify_files(root)


if __name__=='__main__': unittest.main()
