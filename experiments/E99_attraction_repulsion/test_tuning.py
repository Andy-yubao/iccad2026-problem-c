"""Configurable budget, exact V3 reuse, process lock and run-cache checks."""
from dataclasses import asdict
from pathlib import Path
import json,tempfile,unittest
from unittest.mock import patch
import numpy as np
from .deformation import Config as V3Config,relax
from .tuning_config import Parameters,load
from . import tune


class TuningChecks(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parents[2]
        self.parent=(self.root/'.cache/E99_tuning_tests').resolve()
        assert self.parent.is_relative_to(self.root);self.parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=self.parent)
        self.folder=Path(self.temp.name).resolve();assert self.folder.is_relative_to(self.parent)
    def tearDown(self):self.temp.cleanup()

    def test_same_v3_values_exact_same_geometry(self):
        p=np.array([[10.,10.,3.,2.],[11.,10.,2.,4.],[12.,12.,4.,1.]])
        v3,_=relax(p,fixed_shape=[False,False,False],config=V3Config())
        configurable,_=relax(p,fixed_shape=[False,False,False],config=Parameters(**asdict(V3Config())))
        np.testing.assert_array_equal(v3,configurable)

    def test_larger_budgets_area_and_locks(self):
        p=np.array([[10.,10.,3.,2.],[11.,10.,2.,4.],[12.,12.,4.,1.],[9.,11.,2.,3.]])
        out,d=relax(p,[True,False,False,False],fixed_shape=[True,True,False,False],config=Parameters(),return_history=True)
        h=d['history'];centers=h[:,:,:2]+h[:,:,2:]/2;short=p[:,2:].min(1)
        steps=np.linalg.norm(np.diff(centers,axis=0),axis=-1)
        self.assertTrue((steps<=.05*short+1e-12).all());self.assertTrue((steps.sum(0)<=.4*short+1e-12).all())
        q=np.log(h[:,:,2]/p[:,2]);self.assertTrue((np.abs(np.diff(q,axis=0)).sum(0)<=np.log1p(.5)+1e-12).all())
        np.testing.assert_array_equal(out[0],p[0]);np.testing.assert_array_equal(out[1,2:],p[1,2:])
        np.testing.assert_allclose(out[:,2:].prod(1),p[:,2:].prod(1),rtol=1e-13)
        self.assertEqual(d['config']['iterations'],20)

    def test_schema_and_fail_fast(self):
        value=json.loads((Path(__file__).parent/'tuning.json').read_text(encoding='utf-8'))
        path=self.folder/'config.json';path.write_text(json.dumps(value),encoding='utf-8')
        settings,p=load(path);self.assertEqual(p.iterations,20)
        tune.effective_planner(settings['planner'])
        for kwargs in ({'iterations':-1},{'iterations':1.5},{'step_fraction':0},{'shape_total_fraction':float('nan')},{'mass_weighted':False}):
            with self.assertRaises(ValueError):Parameters(**kwargs)
        value['relaxation']['typo_parameter']=1;path.write_text(json.dumps(value),encoding='utf-8')
        with self.assertRaises(TypeError):load(path)
        with self.assertRaises(ValueError):tune.effective_planner({'unknown':True})
        with self.assertRaises(ValueError):tune.effective_planner({'adaptive':'yes'})

    def test_cache_resume_binding_and_no_overwrite(self):
        runs=self.folder/'runs';settings={'label':'first'};identity={'numerics':1}
        with patch.object(tune,'RUNS',runs):
            out,cached,resume=tune.acquire_run(settings,identity);self.assertFalse(cached)
            with self.assertRaises(RuntimeError):tune.acquire_run(settings,identity)
            tune.save(out/'results.json',{'finished':True})
            reused,cached,_=tune.acquire_run({'label':'renamed'},identity)
            self.assertEqual(reused,out);self.assertTrue(cached)
            with self.assertRaises(ValueError):tune.acquire_run(settings,{'numerics':2},resume=out)
            different,_,_=tune.acquire_run(settings,{'numerics':2});self.assertNotEqual(different,out)
            data=out/'coordinates/0000.npz';np.savez(data,positions=np.ones((1,4)))
            row=dict(source_sha256='source',coordinates_sha256=tune.sha(data))
            tune.save(out/'cases/0000.json',row)
            self.assertEqual(tune.verify_case(out,0,'source'),row)
            with self.assertRaises(ValueError):tune.verify_case(out,0,'changed')

    def test_process_lock_released(self):
        path=self.folder/'one.lock'
        with tune.process_lock(path):
            with self.assertRaises((RuntimeError,OSError)):
                with tune.process_lock(path):pass
        with tune.process_lock(path):pass

    def test_display_changes_preserve_numerical_cache(self):
        import copy
        identity=dict(revision='v4',relaxation={'iterations':20},source_sha256={
            'experiments/E99_attraction_repulsion/deformation.py':'kernel',
            'experiments/E99_attraction_repulsion/tune.py':'old_renderer',
            'visualization/layouts.py':'old_style'})
        changed=copy.deepcopy(identity)
        changed['source_sha256']['experiments/E99_attraction_repulsion/tune.py']='new_renderer'
        changed['source_sha256']['visualization/layouts.py']='new_style'
        self.assertEqual(tune.numerical_key(identity),tune.numerical_key(changed))
        changed['source_sha256']['experiments/E99_attraction_repulsion/deformation.py']='different_kernel'
        self.assertNotEqual(tune.numerical_key(identity),tune.numerical_key(changed))


if __name__=='__main__':unittest.main()
