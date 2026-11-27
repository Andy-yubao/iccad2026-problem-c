"""Offline checks for readiness/permission decisions; no Kaggle requests or GPU use."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('e161_deploy_api',Path(__file__).with_name('deploy_api.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class DeploymentChecks(unittest.TestCase):
    def setUp(self):
        self.meta=dict(is_private=True,current_version_number=1)
        self.files={'e161-transfer.tar.gz.bin':100,'transfer-manifest.json':20}

    def test_positive_private_reads_allow_unavailable_status_endpoint(self):
        self.assertTrue(module.assess_dataset(None,self.meta,self.files,self.files))

    def test_real_http_403_status_path_uses_positive_private_reads(self):
        from requests import HTTPError,Response
        response=Response();response.status_code=403
        files=[SimpleNamespace(name=name,total_bytes=size) for name,size in self.files.items()]
        class ReadablePrivateDataset:
            def dataset_status(self,dataset): raise HTTPError('Forbidden',response=response)
            def dataset_list_files(self,dataset,page_size):
                return SimpleNamespace(error_message=None,files=files)
        with patch.object(module,'dataset_metadata',return_value=self.meta):
            result=module.wait_dataset(ReadablePrivateDataset(),'owner/data',self.files,wait_seconds=0)
        self.assertEqual(result['status_endpoint_error']['http_status'],403)
        self.assertEqual(result['files'],self.files)

    def test_status_ready_requires_actual_read_access(self):
        self.assertFalse(module.assess_dataset('ready',None,self.files,self.files))
        self.assertFalse(module.assess_dataset('ready',self.meta,None,self.files))

    def test_public_dataset_is_rejected(self):
        with self.assertRaises(RuntimeError):
            module.assess_dataset('ready',dict(self.meta,is_private=False),self.files,self.files)

    def test_unpublished_or_processing_dataset_waits(self):
        self.assertFalse(module.assess_dataset(None,dict(self.meta,current_version_number=0),self.files,self.files))
        self.assertFalse(module.assess_dataset('processing',self.meta,self.files,self.files))

    def test_missing_files_wait_but_wrong_sizes_fail(self):
        self.assertFalse(module.assess_dataset('ready',self.meta,{},self.files))
        with self.assertRaises(RuntimeError):
            module.assess_dataset('ready',self.meta,dict(self.files,**{'transfer-manifest.json':21}),self.files)

    def test_existing_upload_never_calls_create(self):
        class NoUpload:
            def dataset_create_new(self,*args,**kwargs): raise AssertionError('Unexpected reupload')
        temp_root=Path(__file__).resolve().parents[2]/'tmp'
        with tempfile.TemporaryDirectory(prefix='e161-deploy-',dir=temp_root) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(temp_root.resolve()))
            module.ensure_created(NoUpload(),Path(folder),'owner/data','sha',Path(folder)/'receipt.json',True)

    def test_accepted_upload_receipt_prevents_reupload(self):
        class UploadOnce:
            count=0
            def dataset_create_new(self,*args,**kwargs):
                self.count+=1
                return SimpleNamespace(error=None,status='ok',url='https://www.kaggle.com/datasets/owner/data')
        api=UploadOnce();temp_root=Path(__file__).resolve().parents[2]/'tmp'
        with tempfile.TemporaryDirectory(prefix='e161-deploy-',dir=temp_root) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(temp_root.resolve()))
            path=Path(folder);receipt=path/'receipt.json'
            module.ensure_created(api,path,'owner/data','sha',receipt)
            module.ensure_created(api,path,'owner/data','sha',receipt)
        self.assertEqual(api.count,1)

    def test_kernel_error_cannot_be_recorded_as_success(self):
        with self.assertRaises(RuntimeError):
            module.check_push_response(SimpleNamespace(error='GPU quota unavailable'))
        with self.assertRaises(RuntimeError):
            module.check_push_response(SimpleNamespace(error=None,invalid_dataset_sources=['owner/data']))
        with self.assertRaises(RuntimeError):
            module.check_push_response(SimpleNamespace(error=None,invalid_dataset_sources=[],version_number=0))
        module.check_push_response(SimpleNamespace(error=None,invalid_dataset_sources=[],version_number=1))

if __name__=='__main__': unittest.main()
