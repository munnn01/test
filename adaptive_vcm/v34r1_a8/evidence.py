"""Byte-checked selected packets and actual model-forward counters."""
from collections import Counter
import hashlib
from pathlib import Path


def packet_file(base,raw,digest,size):
    if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('selected wire byte/SHA mismatch')
    path=Path(base)/'packets'/(digest+'.bin');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('existing packet content drift')
    else:path.write_bytes(raw)
    return path.relative_to(base).as_posix()


class ForwardMeter:
    def __init__(self):self.counts=Counter();self.current='unclassified';self.handles=[]
    def stage(self,value):self.current=value
    def attach(self,name,model):
        def count(module,args,result):
            self.counts[self.current+'/'+name+'_calls']+=1
            self.counts[self.current+'/'+name+'_units']+=len(args[0])
        self.handles.append(model.register_forward_hook(count))
    def close(self):
        for handle in self.handles:handle.remove()
