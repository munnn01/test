from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
from adaptive_vcm.v34.holdout import read_frames
from adaptive_vcm.v34r1.frames import read_selected


def test_explicit_provider_indices_match_original_32_frame_center_stride_pixels():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory); rng=np.random.default_rng(13)
        for i in range(32):
            Image.fromarray(rng.integers(0,256,(40,50,3),dtype=np.uint8)).save(root/f'img_{i+1:05d}.jpg')
        expected,_=read_frames(root,frames=16,stride=2,size=32)
        item=dict(path=str(root),provider_original_frames=100,
                  provider_selected_paths=[f'videoid/img_{i+1:05d}.jpg' for i in range(0,32,2)])
        actual,recipe=read_selected(item,frames=16,size=32)
        assert np.array_equal(expected,actual)
        assert recipe['repeated_frames']==0 and recipe['decoded_unique_frames']==16
