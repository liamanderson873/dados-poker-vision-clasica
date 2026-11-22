"""Analiza TODAS las imagenes de entrada, con y sin anotaciones.

py probar_todas.py [carpeta_fotos] [--out pruebas_todas]
No analiza resultados generados fuera de la carpeta de entrada.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time
import cv2
import numpy as np
import vision

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fotos', nargs='?', default='fotos_reales')
    parser.add_argument('--out', default='pruebas_todas')
    args = parser.parse_args()
    root, out = Path(args.fotos).resolve(), Path(args.out).resolve()
    if out == root or root in out.parents:
        parser.error('La salida debe estar fuera de la carpeta de fotos de entrada.')
    out.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in root.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'})
    cfg = vision.load_config()
    rows, thumbs = [], []
    for i, path in enumerate(images, 1):
        relative = path.relative_to(root)
        target = out / relative.with_suffix('')
        target.parent.mkdir(parents=True, exist_ok=True)
        row = {'foto': str(relative), 'estado': '', 'detectados': 0, 'caras': '',
               'esperadas': '', 'coincide_json': '', 'avisos': 0}
        try:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError('Formato no legible')
            dbg = {}
            result = vision.analyze(img, cfg, dbg)
            labels = [d['label'] for d in result['dice']]
            row.update(detectados=len(labels), caras=' '.join(labels),
                       avisos=len([d for d in result['dice'] if d['warnings']]), estado='sin anotacion')
            js = path.with_suffix('.json')
            if js.exists():
                truth = json.loads(js.read_text(encoding='utf-8'))['labels']
                match = Counter(labels) == Counter(truth)
                row.update(esperadas=' '.join(truth), coincide_json=match,
                           estado='coincide con JSON' if match else 'REVISAR discrepancia')
            if relative.parts[0] == 'montaje':
                row['estado'] = 'foto del montaje, fuera del tipo de escena cenital'
            if relative.parts[0] == 'capturas_programa':
                row['estado'] = 'captura de pantalla, no foto original'
            annotated = vision.draw(img, result)
            scale = min(1, 1400 / max(annotated.shape[:2]))
            annotated = cv2.resize(annotated, None, fx=scale, fy=scale)
            cv2.imwrite(str(target) + '_resultado.jpg', annotated)
            cv2.imwrite(str(target) + '_caras.jpg', vision.face_mosaic(result))
            cv2.imwrite(str(target) + '_mascara.png', dbg['object_mask'])
            tile = np.full((270, 360, 3), 245, np.uint8)
            scale = min(360/annotated.shape[1], 235/annotated.shape[0])
            thumb = cv2.resize(annotated, None, fx=scale, fy=scale)
            tile[:thumb.shape[0], :thumb.shape[1]] = thumb
            cv2.putText(tile, f'{i:02d} {relative.as_posix()}'[:45], (4,252), cv2.FONT_HERSHEY_SIMPLEX, .38, (0,0,0), 1)
            thumbs.append(tile)
        except Exception as error:
            row['estado'] = f'ERROR: {error}'
        rows.append(row)
        print(f"[{i}/{len(images)}] {row['foto']}: {row['detectados']} -> {row['caras']} | {row['estado']}", flush=True)
    for start in range(0, len(thumbs), 12):
        group = thumbs[start:start+12]
        group += [np.full((270,360,3), 245, np.uint8)] * (12-len(group))
        sheet = np.vstack([np.hstack(group[j:j+3]) for j in range(0,12,3)])
        cv2.imwrite(str(out / f'resumen_{start//12+1:02d}.jpg'), sheet)
    (out/'resultados.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    if rows:
        with (out/'resultados.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    checked = [r for r in rows if r['coincide_json'] != '']
    lines = ['# Prueba de todas las imagenes', '',
             f"Imagenes procesadas: {len(rows)}. Errores de ejecucion: {sum(r['estado'].startswith('ERROR') for r in rows)}.",
             f"Con JSON: {len(checked)}. Coinciden las etiquetas y su cantidad: {sum(r['coincide_json'] for r in checked)}.", '',
             'Las anotaciones originales tienen discrepancias en dados de borde. Coincidir con el JSON no demuestra exactitud por posicion. Las fotos sin anotacion no tienen una tasa de acierto medida. Las capturas de pantalla y fotos de montaje se procesan por exhaustividad, pero no son escenas cenitales originales.', '',
             '| Foto | Dados | Caras | Estado |', '|---|---:|---|---|']
    lines += [f"| {r['foto']} | {r['detectados']} | {r['caras']} | {r['estado']} |" for r in rows]
    (out/'RESULTADOS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

if __name__ == '__main__':
    main()
