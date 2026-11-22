"""
run.py - Analiza una o varias imagenes y guarda las figuras intermedias.

Uso:
    python run.py foto.jpg [mas.jpg ...] [--out salida] [--verbose]

Genera en --out, por imagen:
    <nombre>_resultado.png   contornos + clase + jugada
    <nombre>_mascara.png     mascara de objetos (segmentacion)
    <nombre>_caras.png       mosaico cara rectificada | simbolo | clase
"""
import argparse
import os

import cv2

import vision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imagenes", nargs="+")
    ap.add_argument("--out", default="salida")
    ap.add_argument("--config", default=None)
    ap.add_argument("--verbose", action="store_true", help="muestra caracteristicas y traza del arbol")
    a = ap.parse_args()
    cfg = vision.load_config(a.config)
    os.makedirs(a.out, exist_ok=True)

    for path in a.imagenes:
        img = cv2.imread(path)
        if img is None:
            print("No se puede leer", path)
            continue
        dbg = {}
        res = vision.analyze(img, cfg, dbg)
        base = os.path.join(a.out, os.path.splitext(os.path.basename(path))[0])
        cv2.imwrite(base + "_resultado.png", vision.draw(img, res))
        cv2.imwrite(base + "_mascara.png", dbg["object_mask"])
        cv2.imwrite(base + "_caras.png", vision.face_mosaic(res))

        h = res["hand"]
        print(f"\n{path}: {len(res['dice'])} dados -> {h['name']} {h['detail']}"
              f"{'' if h['complete'] else '  (escena incompleta: no hay 5 dados)'}")
        if res["mm_per_px"]:
            print(f"  escala: {res['mm_per_px']:.3f} mm/px")
        for warning in res.get("warnings", []):
            print("  REVISAR:", warning)
        for i, d in enumerate(res["dice"], 1):
            print(f"  {i}: {d['label']:>3}  en ({d['center'][0]:.0f},{d['center'][1]:.0f})")
            if a.verbose:
                print("      " + " | ".join(d["trace"]))
                print("      " + ", ".join(f"{k}={v:.3f}" for k, v in d["features"].items()))


if __name__ == "__main__":
    main()
