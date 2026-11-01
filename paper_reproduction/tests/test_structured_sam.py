import pytest
import torch
from reproduce.sam_structured import structured_sam_maps,StructuredMaskCache,SAMGaussianNPP

def settings():
 return dict(min_area_fraction=.005,max_area_fraction=.6,pred_iou_thresh=.8,stability_score_thresh=.9,max_masks=8,smoothing=5)
def test_instance_seams_and_two_channel_geometry():
 a=torch.zeros(12,12);b=a.clone();a[2:10,1:6]=1;b[2:10,6:11]=1
 rows=[dict(segmentation=m,predicted_iou=1.,stability_score=1.) for m in (a,b)]
 maps,info=structured_sam_maps(rows,(12,12),settings())
 assert maps.shape==(1,1,2,12,12) and maps[0,0,1,5,5]>0 and maps[0,0,1,5,6]>0
 assert info['accepted']==2 and info['edge_coverage']>0
 empty,_=structured_sam_maps([], (12,12),settings())
 assert empty.sum()==0

def test_structured_cache_rejects_tampered_edge(tmp_path):
 p=tmp_path/'x.bin';p.write_bytes(b'image')
 row=dict(id='image/1',path=str(p));x=torch.rand(1,1,3,12,12)
 cache=StructuredMaskCache(tmp_path/'cache',settings());maps=torch.rand(1,1,2,12,12)
 cache.store(row,x,maps)
 assert cache.load(row,x).shape==maps.shape
 saved=torch.load(cache.path(row),weights_only=False);saved['mask'][0,0,1,0,0]=.123
 torch.save(saved,cache.path(row))
 with pytest.raises(ValueError):cache.load(row,x)

def test_gaussian_preserves_constant_and_changes_texture():
 g=SAMGaussianNPP(.8);constant=torch.full((1,1,3,16,18),.4)
 assert torch.allclose(g(constant,35),constant,atol=1e-6)
 torch.manual_seed(19);x=torch.rand_like(constant)
 assert g(x,35).std()<x.std() and g(x,35).shape==x.shape
