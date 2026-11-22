"""Pruebas de estabilidad sobre fotos reales con etiquetas conocidas.
Las variantes derivan de las mismas fotos: no son fotos independientes.
"""
from pathlib import Path
import json
from collections import Counter
import argparse
import cv2
import numpy as np
import vision

out=Path('precision_stress'); out.mkdir(exist_ok=True)
cfg=vision.load_config()
rows=[]
for name in ['caja/chat_fila.jpg','caja/20261118_142533.jpg','borde2/n10.jpg']:
    p=Path('fotos_reales')/name
    im=cv2.imread(str(p)); gt=json.loads(p.with_suffix('.json').read_text())['labels']
    _,buf=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,50])
    variants={
        'original': im,
        'luz_65': np.clip(im.astype(np.float32)*.65,0,255).astype(np.uint8),
        'luz_125': np.clip(im.astype(np.float32)*1.25,0,255).astype(np.uint8),
        'giro_90': cv2.rotate(im,cv2.ROTATE_90_CLOCKWISE),
        'giro_180': cv2.rotate(im,cv2.ROTATE_180),
        'mitad_resolucion':cv2.resize(im,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA),
        'jpeg_50':cv2.imdecode(buf,cv2.IMREAD_COLOR),
        'desenfoque_5':cv2.GaussianBlur(im,(5,5),0),
    }
    for variant,img in variants.items():
        res=vision.analyze(img,cfg); pred=[d['label'] for d in res['dice']]
        ok=Counter(gt)==Counter(pred)
        row=dict(foto=name,variante=variant,gt=gt,pred=pred,ok=ok)
        rows.append(row)
        print(name,variant,'OK' if ok else pred,flush=True)
        if not ok:
            key=p.stem+'_'+variant
            cv2.imwrite(str(out/(key+'_caras.jpg')),vision.face_mosaic(res))
            annotated=vision.draw(img,res)
            cv2.imwrite(str(out/(key+'_resultado.jpg')),cv2.resize(annotated,(1000,750)))
(out/'resultados.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print('TOTAL',sum(r['ok'] for r in rows),'/',len(rows))
if not all(r['ok'] for r in rows):
    raise SystemExit(1)
