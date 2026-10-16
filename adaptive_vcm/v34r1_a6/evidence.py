"""Byte authentication shared by cloud A6 executors; no learned operations."""
import json
from pathlib import Path
from ..v34r1_a4.cloud_record import require_sha256


def publication(folder,expected):
    folder=Path(folder).resolve();raw=(folder/'publication-manifest.json').read_bytes();require_sha256(raw,expected)
    manifest=json.loads(raw)
    for entry in manifest['files']:
        path=(folder/entry['path']).resolve()
        if not path.is_relative_to(folder) or path.stat().st_size!=entry['bytes']:raise ValueError('parent evidence path/size drift')
        require_sha256(path.read_bytes(),entry['sha256'])
    status=json.loads((folder/'cloud-status.json').read_text())
    if not status['complete'] or status.get('test_scored') is not False:raise ValueError('complete DEV-only parent evidence required')
    return manifest
