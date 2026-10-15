"""Build giao diện từ mô hình đã xuất, dùng khi bảo trì; người dùng vẫn chỉ chạy notebook."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
source=lambda name:(ROOT/'src'/name).read_text()
config=json.loads((ROOT/'artifacts_giangkt/data/web_model_config.json').read_text())
html=source('web.html').replace('__STYLES__',source('styles.css'))
html=html.replace('__ENGINE__',source('clinical.js')+'\n'+source('engine.js'))
html=html.replace('__APP__',source('app.js')).replace('__MODEL_CONFIG__',json.dumps(config,ensure_ascii=False,allow_nan=False))
(ROOT/'vercel_app/index.html').write_text(html,encoding='utf-8')
print('Built web from saved trained model:',config['version'])
