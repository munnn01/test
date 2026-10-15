"""Install a local FFmpeg7.1 binary after imageio-ffmpeg==0.6.0 is installed."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import imageio_ffmpeg

root=Path(__file__).resolve().parents[1]
source=Path(imageio_ffmpeg.get_ffmpeg_exe())
version=subprocess.check_output([str(source),'-version'],text=True).splitlines()[0]
if '7.1' not in version: raise ValueError('requires imageio-ffmpeg0.6.0 bundled FFmpeg7.1')
target=root/'.tools/ffmpeg'/('ffmpeg.exe' if source.suffix=='.exe' else 'ffmpeg')
target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
print(version); print('SHA256',hashlib.sha256(target.read_bytes()).hexdigest()); print(target)
