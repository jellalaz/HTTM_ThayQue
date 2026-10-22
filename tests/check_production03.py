"""Smoke test read-only của inference production bằng dữ liệu công khai/tổng hợp."""
from pathlib import Path
import json
import time
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
base='https://rag-httm-chatbot.vercel.app'
results=[]
page=requests.get(base+'/',timeout=30);page.raise_for_status()
assert 'THỰC NGHIỆM TIỂU LUẬN 03' in page.text and 'HealthChat' in page.text and 'panel-gold' in page.text
r=requests.get(base+'/api/predict',timeout=30);r.raise_for_status();assert r.json()['version']=='3.0'
sample=json.loads((ROOT/'artifacts_tl03/data/cnn_parity.json').read_text())[0]
parity=json.loads((ROOT/'artifacts_tl03/data/tabular_parity.json').read_text())
payloads=[{'task':'cnn','pixels':sample['pixels']},next(s for s in parity if s['task']=='housing'),next(s for s in parity if s['task']=='diabetes_classifier'),{'task':'sequence','domain':'fpt'},{'task':'sequence','domain':'gold'}]
for payload in payloads:
    payload={k:v for k,v in payload.items() if k!='expected'}
    started=time.perf_counter();r=requests.post(base+'/api/predict',json=payload,timeout=40)
    assert r.status_code==200,(payload['task'],r.status_code,r.text[:300])
    result=r.json()
    if payload['task']=='cnn':assert result['digit']==7
    elif payload['task']=='housing':assert result['prediction']>0
    elif payload['task']=='diabetes_classifier':assert result['label'] in ['High','Low','Moderate']
    else:assert result['prediction']>0
    row={'task':payload['task'],'domain':payload.get('domain'),'HTTP':r.status_code,'seconds':time.perf_counter()-started,'result':result}
    results.append(row);print(payload['task'],payload.get('domain',''),r.status_code,round(row['seconds'],3))
r=requests.post(base+'/api/predict',json={'task':'cnn','pixels':[0]},timeout=30)
assert r.status_code==400
out={'url':base,'checked_at':datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat(),'frontend_HTTP':page.status_code,'health_endpoint':True,'invalid_input_HTTP':r.status_code,'cases':results}
(ROOT/'artifacts_tl03/data/production_verified.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('Production checks passed')
