import math
from types import SimpleNamespace
import pytest
import torch
from reproduce.codec_content_rate import nal_accounting,ContentProxy,fit_scales,gradient_balance,screen_gate

def test_nal_bytes_accounted_exactly_for_both_codecs():
    avc=b'\x00\x00\x00\x01\x67abcd'+b'\x00\x00\x01\x68ef'+b'\x00\x00\x01\x65payload'
    r=nal_accounting(avc,'h264')
    assert r['content_bytes']==11 and r['overhead_bytes']==15 and r['total_bytes']==26
    hevc=b'\x00\x00\x01'+bytes([32<<1,1])+b'ab'+b'\x00\x00\x00\x01'+bytes([19<<1,1])+b'xyz'
    r=nal_accounting(hevc,'h265')
    assert r['content_bytes']==9 and r['overhead_bytes']==7 and r['total_bytes']==16
    with pytest.raises(ValueError):nal_accounting(b'not-annexb','h264')
    with pytest.raises(ValueError):nal_accounting(b'\x00\x00\x01\x67abcd','h264')

class Bank(torch.nn.Module):
    def forward(self,x,q):return SimpleNamespace(reconstruction=x,bpp=x.mean()+.1)
class Codec:
    def overhead(self,x,q):return 100
def test_overhead_scales_by_paid_pixels_without_image_gradient():
    p=ContentProxy(Bank(),Codec());p.content_scale.fill_(2)
    x=torch.full((1,1,3,10,20),.2,requires_grad=True)
    r=p(x,35)
    assert float(r.overhead_bpp)==4
    assert torch.allclose(r.bpp,2*(x.mean()+.1)+4)
    r.bpp.backward()
    assert torch.allclose(x.grad,torch.full_like(x,2/x.numel()))
    bigger=p(torch.full((1,1,3,20,40),.2),35)
    assert float(bigger.overhead_bpp)==1
    saved=p.state_dict();assert all(isinstance(v,torch.Tensor) for v in saved.values())
    clone=ContentProxy(Bank(),Codec());clone.load_state_dict(saved)
    assert torch.equal(clone.content_scale,p.content_scale)

def test_scale_fit_does_not_learn_non_vcl_bytes():
    rows=[dict(qp=q,raw_bpp=r,content_bpp=3*r,source='train/a') for q in (30,35,40,45) for r in (.1,.2,.4)]
    assert all(abs(s-3)<1e-6 for s in fit_scales(rows))
    rows[0]['content_bpp']=0
    with pytest.raises(ValueError):fit_scales(rows)

def test_gradient_norms_match_weighted_terms_and_cosine():
    w=torch.nn.Parameter(torch.tensor([1.,2.]))
    info=gradient_balance(w.sum(),2*w.sum(),(w*w).sum(),[w])
    assert info['gradient_rate_norm']==pytest.approx(math.sqrt(2))
    assert info['gradient_task_norm']==pytest.approx(math.sqrt(8))
    assert info['gradient_rate_task_ratio']==pytest.approx(.5)
    assert info['gradient_rate_task_cosine']==pytest.approx(1)

def test_screen_gate_rejects_large_errors_and_direction_failure():
    cfg={'rate_median_error_max':.35,'rate_p90_error_max':.8,'rate_direction_min':.75}
    good=[dict(real_bpp=1.,relaxed_bpp=1.1)]*8
    assert screen_gate(good,dict(aligned=8,untied=8),cfg)['passed']
    assert not screen_gate([dict(real_bpp=1.,relaxed_bpp=8.)]*8,dict(aligned=8,untied=8),cfg)['passed']
    assert not screen_gate(good,dict(aligned=3,untied=8),cfg)['passed']
