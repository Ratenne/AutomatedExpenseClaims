from io import BytesIO
from PIL import Image
import app
import os

# Diagnostics
print('TESSDATA_DIR=', getattr(app, 'TESSDATA_DIR', None))
print('TESS_LANG=', getattr(app, 'TESS_LANG', None))
print('TESS_CONFIG=', getattr(app, 'TESS_CONFIG', None))
print('exists kor_best:', os.path.exists(os.path.join(getattr(app, 'TESSDATA_DIR', ''), 'kor_best.traineddata')))
print('exists kor:', os.path.exists(os.path.join(getattr(app, 'TESSDATA_DIR', ''), 'kor.traineddata')))

# Create a small white PNG
img = Image.new('RGB', (200,100), color=(255,255,255))
buf = BytesIO()
img.save(buf, format='PNG')
buf.seek(0)

client = app.app.test_client()
resp = client.post('/ocr', data={'image': (buf, 'test.png')}, content_type='multipart/form-data')
print('STATUS', resp.status_code)
print(resp.get_data(as_text=True))
