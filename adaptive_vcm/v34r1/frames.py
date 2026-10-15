"""Read precisely the predeclared provider JPEG indices, without interpolation/padding."""
import hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def read_selected(item,*,frames,size):
    root=Path(item['path']).resolve(); names=[Path(p).name for p in item['provider_selected_paths']]
    if len(names)!=frames or len(set(names))!=frames: raise ValueError('distinct predeclared source frames required')
    picked=[]; files=[]
    for name in names:
        path=(root/name).resolve()
        if not path.is_relative_to(root): raise ValueError('provider frame escapes source root')
        with Image.open(path) as im: image=np.asarray(im.convert('RGB'))
        picked.append(cv2.resize(image,(size,size),interpolation=cv2.INTER_AREA))
        files.append(dict(name=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return np.stack(picked),dict(kind='explicit-provider-center-stride-indices',
        provider_original_frames=item['provider_original_frames'],provider_selected_paths=item['provider_selected_paths'],
        decoded_unique_frames=frames,repeated_frames=0,selected_frame_files=files,
        original_video_cadence='not verified; provider F10 title is not ground-truth FPS')
