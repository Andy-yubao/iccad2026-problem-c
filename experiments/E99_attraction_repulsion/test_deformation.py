"""Mechanism and budget tests of V3, including balanced nonzero stress."""
from dataclasses import replace
import unittest
import numpy as np
from .deformation import Config,relax,preprocess


class DeformationChecks(unittest.TestCase):
    def test_balanced_compression_and_tension(self):
        for gap,sign in ((-.2,-1),(.2,1)):
            p=np.array([[10.,10.,2.,2.],[9.-gap,10.5,1.,1.],[12.+gap,10.5,1.,1.]])
            out,d=relax(p,[False,True,True],fixed_shape=[False,True,True],config=Config(iterations=1))
            self.assertEqual(np.sign(out[0,2]-p[0,2]),sign)
            self.assertEqual(np.sign(out[0,3]-p[0,3]),-sign)
            np.testing.assert_allclose(out[0,:2]+out[0,2:]/2,p[0,:2]+p[0,2:]/2,atol=2e-14)
            np.testing.assert_array_equal(out[1:],p[1:])
            vertical=p[:,[1,0,3,2]]
            v,_=relax(vertical,[False,True,True],fixed_shape=[False,True,True],config=Config(iterations=1))
            np.testing.assert_allclose(v,out[:,[1,0,3,2]],atol=2e-14)

    def test_center_and_shape_paths_and_area(self):
        p=np.array([[10.,10.,3.,2.],[11.,10.,2.,4.],[12.,12.,4.,1.],[9.,11.,2.,3.]])
        out,d=relax(p,fixed_shape=np.zeros(4,bool),return_history=True)
        h=d['history'];center=h[:,:,:2]+h[:,:,2:]/2;short=p[:,2:].min(1)
        steps=np.linalg.norm(np.diff(center,axis=0),axis=-1)
        self.assertTrue((steps<=.03*short+1e-12).all())
        self.assertTrue((steps.sum(0)<.2*short).all())
        q=np.log(h[:,:,2]/p[:,2]);self.assertTrue((np.abs(np.diff(q,axis=0))<=np.log1p(.015)+1e-12).all())
        self.assertTrue((np.abs(np.diff(q,axis=0)).sum(0)<=np.log1p(.15)+1e-12).all())
        np.testing.assert_allclose(h[:,:,2]*h[:,:,3],np.broadcast_to(p[:,2]*p[:,3],h.shape[:2]),rtol=1e-13)
        scaled,_=relax(p*7,fixed_shape=np.zeros(4,bool))
        np.testing.assert_allclose(scaled,out*7,atol=3e-12)
        self.assertEqual(d['config']['iterations'],10)

    def test_fixed_shape_may_move_preplaced_may_not(self):
        p=np.array([[10.,10.,2.,2.],[11.2,10.,2.,2.]])
        out,_=relax(p,[True,False],fixed_shape=[True,True])
        np.testing.assert_array_equal(out[:,2:],p[:,2:]);np.testing.assert_array_equal(out[0],p[0])
        self.assertGreater(np.linalg.norm(out[1,:2]-p[1,:2]),0)

    def test_inverse_mass_and_translation_only(self):
        p=np.array([[10.,10.,2.,2.],[11.8,10.5,1.,1.]])
        out,_=relax(p,config=Config(iterations=1,shape_enabled=False))
        np.testing.assert_allclose((out[0,:2]-p[0,:2])*4,-(out[1,:2]-p[1,:2]),atol=2e-14)
        np.testing.assert_array_equal(out[:,2:],p[:,2:])

    def test_domain_identity_and_public_adapter(self):
        p=np.array([[0.,0.,1.,1.],[.8,0.,1.,1.]])
        c=np.zeros((2,5));c[0,1]=1;s=np.full((2,4),-1.);s[0]=p[0]
        inputs=(2,np.ones(2),None,None,None,c,s)
        out,d=preprocess(p,inputs,return_history=True)
        self.assertTrue((d['history'][:,:,:2]>=0).all());np.testing.assert_array_equal(out[0],p[0])
        identity,_=relax(p,fixed_shape=[False,False],config=Config(iterations=0))
        np.testing.assert_array_equal(identity,p)
        with self.assertRaises(ValueError):relax(p)
        with self.assertRaises(ValueError):Config(iterations=11)
        with self.assertRaises(ValueError):Config(total_fraction=.2)
        with self.assertRaises(ValueError):Config(shape_step_fraction=.02)
        with self.assertRaises(ValueError):relax(p,fixed_shape=[False])

    def test_file_entry_and_output_protection(self):
        from pathlib import Path
        import tempfile,subprocess,sys
        workspace=Path(__file__).resolve().parents[2]
        cache=(workspace/'.cache/E99_deform_cli').resolve()
        assert cache.is_relative_to(workspace);cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            folder=Path(tmp).resolve();assert folder.is_relative_to(cache)
            source=folder/'input.npz';target=folder/'output.npz'
            p=np.array([[10.,10.,2.,2.],[11.8,10.5,1.,1.]])
            np.savez(source,positions=p,preplaced=[False,True],fixed_shape=[False,True])
            command=[sys.executable,'-B','-m','experiments.E99_attraction_repulsion.deform_file','--input',str(source),'--output',str(target)]
            subprocess.run(command,check=True,capture_output=True)
            expected,_=relax(p,[False,True],fixed_shape=[False,True])
            with np.load(target,allow_pickle=False) as saved:np.testing.assert_array_equal(saved['positions'],expected)
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)


if __name__=='__main__':unittest.main()
