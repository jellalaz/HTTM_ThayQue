"""Kiểm tra đóng gói độc lập với training và các trường hợp input lỗi."""
from pathlib import Path
import gzip
import importlib.util
import json
import sys
import unittest
import math
import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from inference03 import predict, sequence_predict
from experiments03 import SequenceModel


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(ROOT/'vercel_app/models/tl03.json.gz','rt') as f:cls.bundle=json.load(f)
    def test_tabular_parity(self):
        errors=[]
        for s in json.loads((ROOT/'artifacts_tl03/data/tabular_parity.json').read_text()):
            actual=predict(self.bundle,s)
            if s['task']=='housing':self.assertAlmostEqual(actual['prediction'],s['expected'],places=9);errors.append(abs(actual['prediction']-s['expected']))
            else:
                for k,v in s['expected'].items():self.assertAlmostEqual(actual['scores'][k],v,places=10);errors.append(abs(actual['scores'][k]-v))
        print('Tabular max absolute error',max(errors))
    def test_cnn_logits(self):
        errors=[]
        for s in json.loads((ROOT/'artifacts_tl03/data/cnn_parity.json').read_text()):
            r=predict(self.bundle,{'task':'cnn','pixels':s['pixels']})
            self.assertEqual(r['digit'],max(range(10),key=lambda i:s['logits'][i]))
            error=max(abs(a-b) for a,b in zip(r['logits'],s['logits']));errors.append(error);self.assertLess(error,3e-5)
        print('CNN max absolute logit error',max(errors))
    def test_all_recurrent_checkpoints(self):
        errors=[]
        for domain in ['fpt','gold']:
            for kind in ['RNN','LSTM','GRU']:
                cp=torch.load(ROOT/f'artifacts_tl03/models/{domain}_{kind.lower()}.pt',weights_only=False)
                m=dict(self.bundle['sequences'][domain]);m.update({k:cp[k] for k in ['mu','sigma','ym','yscale','window']});m['kind']=kind;m['weights']={k:v.tolist() for k,v in cp['state_dict'].items()}
                model=SequenceModel(kind);model.load_state_dict(cp['state_dict']);model.eval()
                x=(np.array(m['last_returns'])-m['mu'])/m['sigma']
                with torch.no_grad():ret=model(torch.tensor(x[None,:,None],dtype=torch.float32)).item()*m['yscale']+m['ym']
                expected=m['last_close']*math.exp(ret);got=sequence_predict(m,m['last_returns'],m['last_close'])['prediction']
                errors.append(abs(expected-got));self.assertLess(abs(expected-got),max(1e-4,expected*1e-7))
        print('Recurrent max absolute price error',max(errors))
    def test_invalid_inputs(self):
        for p in [{'task':'cnn','pixels':[0]*783},{'task':'cnn','pixels':[float('nan')]*784},{'task':'sequence','returns':[0]*29},{'task':'sequence','domain':'unknown'},{'task':'housing','features':{'Area':'bad'}},{'task':'housing','features':{'Area':float('inf')}},{'task':'unknown'}]:
            with self.assertRaises((ValueError,TypeError)):predict(self.bundle,p)
    def test_blank_and_imputation(self):
        result=predict(self.bundle,{'task':'housing','features':{}})
        self.assertTrue(math.isfinite(result['prediction']));self.assertTrue(result['warnings'])


if __name__=='__main__':unittest.main(verbosity=2)
