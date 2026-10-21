"""Actual standard-codec round trips; count all elementary-stream bytes."""
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from .models import validate_video


def find_ffmpeg():
    path = shutil.which("ffmpeg")
    candidate = Path(sys.executable).parent / "Library/bin/ffmpeg.exe"
    if path:
        return path
    if candidate.is_file():
        return str(candidate)
    raise RuntimeError("FFmpeg with libx264 and libx265 is required")


@dataclass
class CodecOutput:
    reconstruction: torch.Tensor
    bpp: torch.Tensor
    streams: list[bytes]
    seconds: float


class RealCodec:
    def __init__(self, preset="medium", gop=16, fps=25, threads=2, timeout=120):
        if type(gop) is not int or gop < 1 or fps <= 0 or threads < 1 or timeout <= 0:
            raise ValueError("invalid codec runtime settings")
        self.ffmpeg = find_ffmpeg()
        self.preset, self.gop, self.fps = preset, gop, fps
        self.threads, self.timeout = threads, timeout

    def _call(self, args, data=None):
        proc = subprocess.run([self.ffmpeg, "-nostdin", "-v", "error", *args],
                              input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=self.timeout)
        if proc.returncode:
            raise RuntimeError(f"FFmpeg exit {proc.returncode}: {proc.stderr.decode(errors='replace')[-3000:]}")
        return proc.stdout

    @torch.no_grad()
    def roundtrip(self, x, codec, qp):
        validate_video(x)
        if codec not in ("h264", "h265") or type(qp) is not int or not 0 <= qp <= 51:
            raise ValueError("invalid codec or QP")
        b, t, _, h, w = x.shape
        if h % 2 or w % 2:
            raise ValueError("YUV420 requires even dimensions")
        if x.min() < 0 or x.max() > 1:
            raise ValueError("RGB must be in [0,1]")
        arrays = (x.detach().cpu().permute(0, 1, 3, 4, 2).numpy() * 255).round().astype(np.uint8)
        outputs, streams = [], []
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="arod-codec-") as tmp:
            for i, rgb in enumerate(arrays):
                path = Path(tmp) / f"{i}.{'264' if codec == 'h264' else '265'}"
                params = f"keyint={self.gop}:min-keyint={self.gop}:scenecut=0:bframes=0:open-gop=0"
                if codec == "h265":
                    params += f":pools={self.threads}:frame-threads=1:log-level=error"
                self._call(["-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                            "-r", str(self.fps), "-i", "pipe:0", "-an", "-c:v",
                            "libx264" if codec == "h264" else "libx265", "-preset", self.preset,
                            "-qp", str(qp), "-threads", str(self.threads), "-pix_fmt", "yuv420p",
                            "-x264-params" if codec == "h264" else "-x265-params", params,
                            "-f", "h264" if codec == "h264" else "hevc", str(path)], rgb.tobytes())
                stream = path.read_bytes()
                if not stream:
                    raise RuntimeError("codec produced zero bytes")
                raw = self._call(["-xerror", "-err_detect", "explode", "-threads", str(self.threads),
                                  "-i", str(path), "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
                if len(raw) != t * h * w * 3:
                    raise RuntimeError("codec decode frame/geometry mismatch")
                outputs.append(np.frombuffer(raw, np.uint8).reshape(t, h, w, 3).copy())
                streams.append(stream)
        decoded = torch.from_numpy(np.stack(outputs)).permute(0, 1, 4, 2, 3).to(x) / 255
        bpp = x.new_tensor([8 * len(s) / (t * h * w) for s in streams])
        return CodecOutput(decoded, bpp, streams, time.perf_counter() - start)
