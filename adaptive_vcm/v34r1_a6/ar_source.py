"""Pure provider frame-index contract, identical to the original A2 sample."""
import re
from pathlib import PurePosixPath


def historical_parity(hashes,selected,old):
    """Authenticate what was measured originally, without inventing older hashes."""
    prior=old['file_sha256'];declared=old['provider_window_files']
    if (set(prior)!=set(declared) or not set(prior)<=set(hashes)
        or selected!=old['provider_selected_paths']
        or any(PurePosixPath(p).name not in prior for p in selected)
        or any(hashes[n]!=sha for n,sha in prior.items())):
        raise ValueError('exact historical declared bytes/selected indices drift')
    return len(prior)


def provider_window(names,count):
    names=sorted(names)
    if (type(count) is not int or count<32 or len(names)!=count or len(set(names))!=count
        or any(not re.fullmatch(r'img_\d{5}\.jpg',n) for n in names)
        or names!=[f'img_{i:05d}.jpg' for i in range(1,count+1)]):
        raise ValueError('complete contiguous provider directory, no replacement required')
    start=(count-32)//2;window=names[start:start+32];selected=window[::2]
    return window,selected
