import importlib
import importlib.util
import unittest
import numpy as np


def module(name):
    path='adaptive_vcm.v34r1.'+name
    if importlib.util.find_spec(path) is None:
        raise AssertionError(name+' block is missing')
    return importlib.import_module(path)


class FilterContracts(unittest.TestCase):
    def setUp(self):
        self.rgb=np.random.default_rng(7).integers(0,256,(2,32,32,3),dtype=np.uint8)
        self.maps=np.zeros_like(self.rgb,dtype=np.float32)
        self.maps[:,:,:,2]=1
        self.maps[:,8:24,8:24,2]=0
        self.maps[:,8:24,8:24,0]=1

    def test_identity_is_exact_and_stronger_background_preserves_object_core(self):
        p=module('preprocessing')
        self.assertTrue(np.array_equal(p.filter_clip(self.rgb,self.maps,p.PROFILES[0]),self.rgb))
        y=p.filter_clip(self.rgb,self.maps,p.PROFILES[13])
        self.assertEqual(y.shape,self.rgb.shape)
        self.assertEqual(y.dtype,np.uint8)
        self.assertTrue(np.array_equal(y[:,8:24,8:24],self.rgb[:,8:24,8:24]))
        self.assertLess(y[:,:8].astype(float).var(),self.rgb[:,:8].astype(float).var())

    def test_invalid_maps_and_profile_are_rejected(self):
        p=module('preprocessing')
        with self.assertRaises(ValueError): p.filter_clip(self.rgb,self.maps*0,p.PROFILES[0])
        with self.assertRaises(ValueError): p.Profile('bad',(-2,)*6)
        with self.assertRaises(ValueError): p.Profile('bad',(0,)*6,sigma_mid=float('nan'))

    def test_first_thirteen_profiles_keep_legacy_pixels(self):
        p=module('preprocessing')
        from adaptive_vcm.v34.preprocessing import filter_clip as old_filter
        from adaptive_vcm.v34.protocol import ACTIONS
        for action,profile in zip(ACTIONS,p.PROFILES[:13]):
            np.testing.assert_array_equal(p.filter_clip(self.rgb,self.maps,profile),old_filter(self.rgb,self.maps,action))


class SelectionContracts(unittest.TestCase):
    def test_measured_guard_uses_actual_rate_and_falls_back_to_identity(self):
        p=module('policy')
        choice,report=p.select_measured([1,.5,.7],[[0,None,None],[.2,None,None],[.01,None,None]],
                                        [True,False,False],[.025,0,0])
        self.assertEqual(choice,2)
        self.assertIn(1,report['rejected'])
        choice,_=p.select_measured([1,.5],[[0,None,None],[.2,None,None]],[1,0,0],[.025,0,0])
        self.assertEqual(choice,0)

    def test_invalid_anchor_and_budget_cannot_silently_choose_an_edit(self):
        p=module('policy')
        with self.assertRaises(ValueError): p.select_measured([1,.5],[[float('nan'),None,None],[0,None,None]],[1,0,0],[.1,0,0])
        with self.assertRaises(ValueError): p.select_measured([1,.5],[[0,None,None],[0,None,None]],[1,0,0],[.1,0,0],budget=.9)

    def test_shortlist_keeps_identity_and_explicit_unlearned_candidates(self):
        p=module('policy')
        prediction={'logits':np.array([0,10,1.]),'risk':np.array([[0,0,0],[100,0,0],[.001,0,0]]), 'rate':np.array([1,.1,.7])}
        actions=p.shortlist(prediction,[1,0,0],[.05,0,0],max_learned=1,extras=[3,4])
        self.assertEqual(actions,[0,2,3,4])


if __name__=='__main__': unittest.main()
