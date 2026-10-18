"""Pure identity, sample and complete paid-grid contracts."""
from collections import defaultdict
from fractions import Fraction
import hashlib

QPS=(30,35,40,45)
CODECS=('h264','h265')


def rank(seed,value):return hashlib.sha256((seed+'/'+str(value)).encode()).hexdigest()


def stratified(items,count,forbidden,*,seed):
    groups=defaultdict(list);ids=set()
    for row in items:
        if row['id'] in ids:raise ValueError('duplicate source identity')
        ids.add(row['id'])
        if row['id'] not in forbidden:groups[row['label']].append(row)
    if not groups or type(count) is not int or count<1:raise ValueError('positive count and nonempty classes required')
    labels=sorted(groups,key=lambda c:rank(seed+'/class',c));base,extra=divmod(count,len(labels));out=[]
    for i,label in enumerate(labels):
        need=base+(i<extra);ordered=sorted(groups[label],key=lambda r:rank(seed,r['id']))
        if len(ordered)<need:raise ValueError('insufficient sources in a predeclared class; no substitution')
        out.extend(ordered[:need])
    return sorted(out,key=lambda r:rank(seed+'/order',r['id']))


def shard(rows,index,count):
    if type(index) is not int or type(count) is not int or not 0<=index<count:raise ValueError('valid shard index required')
    return [r for i,r in enumerate(rows) if i%count==index]


def audit_selected(rows,ids):
    if len(ids)!=len(set(ids)) or not ids:raise ValueError('unique source IDs required')
    expected={(n,c,q,a) for n in ids for c in CODECS for q in QPS for a in ('anchor','adaptive')};seen=set()
    for r in rows:
        key=(r['id'],r['codec'],r['qp'],r['arm'])
        if (key not in expected or key in seen or r['metadata_bytes']!=40 or r['elementary_bytes']<=0
            or r['bytes']!=r['elementary_bytes']+40):raise ValueError('missing, duplicate, undeclared or unpaid selected stream')
        seen.add(key)
    if seen!=expected:raise ValueError('complete paired four-QP selected grid required')
    return len(seen)


def native_plan(seq,frames):
    fps=Fraction(seq['frameRate']);total=int(seq['seqLength']);width=int(seq['imWidth']);height=int(seq['imHeight'])
    if (fps.denominator!=1 or not 1<=fps.numerator<=120 or min(total,width,height)<1 or type(frames) is not int
        or frames<16 or frames%16 or frames>total or seq['imDir']!='img1' or seq['imExt']!='.jpg'):
        raise ValueError('complete native-cadence tracking prefix required')
    return dict(name=seq['name'],fps=int(fps),full_frames=total,width=width,height=height,frames=frames)
