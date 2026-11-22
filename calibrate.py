"""
calibrate.py - Justificacion EXPERIMENTAL de parametros y umbrales.

Subcomandos:
  felt IMG                  Mide L y a* (Lab) del tapete y de los dados; justifica felt_lab.
  resolution IMG            Escala mm/px, campo de vision y grosor de trazo minimo.
  features DIR [--csv f]    DIR/<clase>/*.jpg (fotos con TODOS los dados mostrando esa cara).
                            Extrae caracteristicas de cada dado y muestra estadisticas por clase.
  thresholds CSV [--write]  Para cada nodo del arbol de config.json: min/max por lado,
                            margen y umbral propuesto (punto medio del hueco).
  suggest CSV               Propone un arbol (busqueda voraz de la caracteristica con mayor
                            margen normalizado). Es una AYUDA: el alumno decide y justifica.
  plots CSV --out dir       Histogramas por clase de cada caracteristica (figuras del informe).
"""
import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np

import vision
from tree import to_text, to_mermaid

IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def _images(d):
    out = []
    for e in IMG_EXT:
        out += glob.glob(os.path.join(d, e))
    return sorted(out)


# ---------------------------------------------------------------------------
def cmd_felt(a, cfg):
    """Mide en Lab el tapete y los dados detectados y comprueba que los umbrales
    de felt_lab caen en el hueco entre ambos. Justifica felt_lab."""
    img = cv2.imread(a.img)
    _, blur, _ = vision.preprocess(img, cfg)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(int)
    L, A = lab[..., 0], lab[..., 1] - 128
    obj = vision.object_mask(blur, cfg) > 0
    # Tapete = todo lo que no es objeto, quitando un margen alrededor de los dados
    near = cv2.dilate(np.uint8(obj) * 255, np.ones((25, 25), np.uint8)) > 0
    k = cfg["segmentation"]["morph_kernel"]
    roi = vision.felt_roi(vision.felt_mask(blur, cfg),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    felt = ~near & roi  # solo dentro del tapete (fuera hay mesa, sabana...)
    # Cara de dado = interior de los objetos detectados
    die = cv2.erode(np.uint8(obj) * 255, np.ones((9, 9), np.uint8)) > 0
    f = cfg["felt_lab"]
    for name, m in (("tapete", felt), ("dados", die)):
        if not m.any():
            continue
        pl = np.percentile(L[m], [1, 5, 50, 95, 99, 99.9])
        pa = np.percentile(A[m], [1, 5, 50, 95, 99, 99.9])
        print(f"{name:7s} ({m.mean() * 100:.1f} % de la imagen):")
        print(f"   L : p1={pl[0]:.0f} p5={pl[1]:.0f} mediana={pl[2]:.0f} p95={pl[3]:.0f} p99={pl[4]:.0f} p99.9={pl[5]:.0f}")
        print(f"   a*: p1={pa[0]:.0f} p5={pa[1]:.0f} mediana={pa[2]:.0f} p95={pa[3]:.0f} p99={pa[4]:.0f} p99.9={pa[5]:.0f}")
    if felt.any():
        pasa = ((A[felt] > f["a_felt_max"]) & (L[felt] >= f["l_obj_low"])) | (L[felt] >= f["l_obj_min"])
        print(f"\nUmbrales actuales: a_felt_max={f['a_felt_max']}  l_obj_low={f['l_obj_low']}  l_obj_min={f['l_obj_min']}")
        print(f"  pixeles de tapete que pasarian por objeto: {100 * pasa.mean():.3f} %"
              " (sueltos: los elimina la apertura morfologica)")
        print(f"  referencia: a_felt_max ~ p99.9(a* tapete) + 2 = {np.percentile(A[felt], 99.9) + 2:.0f};"
              f"  l_obj_min ~ p99.9(L tapete) + 5 = {np.percentile(L[felt], 99.9) + 5:.0f}")


def cmd_resolution(a, cfg):
    """Justifica que la camara resuelve el detalle mas pequeno: grosor del trazo
    del simbolo medido con la transformada de distancia (2*max = anchura)."""
    img = cv2.imread(a.img)
    res = vision.analyze(img, cfg)
    if not res["dice"]:
        print("No se detectan dados")
        return
    H, W = res["image_shape"]
    mmpx = res["mm_per_px"]
    side = np.median([d["side_px"] for d in res["dice"]])
    strokes = []
    for d in res["dice"]:
        m = d["symbol_mask"]
        if cv2.countNonZero(m):
            dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
            # anchura de trazo en px de la cara rectificada -> px de imagen
            k = side / vision.load_config(a.config)["face"]["size_px"]
            strokes.append(2 * np.median(dist[dist > 0.5 * dist.max()]) * k)
    stroke_px = float(np.median(strokes)) if strokes else 0
    print(f"Imagen: {W}x{H} px")
    print(f"Lado del dado: {side:.1f} px  ({cfg['die']['side_mm']} mm)  -> {mmpx:.3f} mm/px")
    print(f"Campo de vision: {W * mmpx:.0f} x {H * mmpx:.0f} mm  (requisito >= 500 x 500 mm: "
          f"{'OK' if W * mmpx >= 500 and H * mmpx >= 500 else 'NO CUMPLE'})")
    print(f"Grosor de trazo del simbolo: {stroke_px:.1f} px = {stroke_px * mmpx:.2f} mm")
    print(f"Criterio (Nyquist practico): trazo >= 3 px -> {'OK' if stroke_px >= 3 else 'INSUFICIENTE'}")


# ---------------------------------------------------------------------------
def cmd_features(a, cfg):
    rows = []
    for cls in cfg["classes"]:
        d = os.path.join(a.dir, cls)
        if not os.path.isdir(d):
            print(f"(aviso) falta carpeta {d}")
            continue
        for p in _images(d):
            res = vision.analyze(cv2.imread(p), cfg)
            for i, r in enumerate(res["dice"]):
                row = {"class": cls, "image": os.path.basename(p), "die": i}
                row.update(r["features"])
                rows.append(row)
    if not rows:
        print("Sin datos")
        return
    with open(a.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} dados medidos -> {a.csv}\n")
    _print_stats(rows, cfg["classes"])


def _load_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    feats = [k for k in rows[0] if k not in ("class", "image", "die", "ink_thr")]
    for r in rows:
        for k in feats + ["ink_thr"]:
            r[k] = float(r[k])
    return rows, feats


def _print_stats(rows, classes):
    feats = [k for k in rows[0] if k not in ("class", "image", "die")]
    print("Media +- desviacion [min, max] por clase (tabla para el informe)\n")
    for k in feats:
        print(f"{k}")
        for c in classes:
            v = np.array([r[k] for r in rows if r["class"] == c], float)
            if len(v):
                print(f"   {c:>3} (n={len(v):3d}): {v.mean():8.3f} +- {v.std():6.3f}  [{v.min():.3f}, {v.max():.3f}]")


# ---------------------------------------------------------------------------
def _leaves(node):
    return {node["label"]} if "label" in node else _leaves(node["left"]) | _leaves(node["right"])


def _split_stats(rows, feat, left_cls, right_cls):
    L = np.array([r[feat] for r in rows if r["class"] in left_cls])
    R = np.array([r[feat] for r in rows if r["class"] in right_cls])
    return L, R


def cmd_thresholds(a, cfg):
    """Verifica cada nodo: las clases del subarbol izquierdo deben quedar <= umbral
    y las del derecho > umbral. Propone el punto medio del hueco (maximo margen)."""
    rows, _ = _load_csv(a.csv)

    def walk(node, path="raiz"):
        if "label" in node:
            return
        lc, rc = _leaves(node["left"]), _leaves(node["right"])
        L, R = _split_stats(rows, node["feature"], lc, rc)
        print(f"\nNodo {path}: {node['feature']} <= {node['threshold']}")
        print(f"   izquierda {sorted(lc)}: n={len(L)} max={L.max() if len(L) else float('nan'):.3f}")
        print(f"   derecha   {sorted(rc)}: n={len(R)} min={R.min() if len(R) else float('nan'):.3f}")
        if len(L) and len(R):
            gap = R.min() - L.max()
            err = int(np.sum(L > node["threshold"]) + np.sum(R <= node["threshold"]))
            prop = (R.min() + L.max()) / 2
            spread = np.std(np.concatenate([L, R])) + 1e-9
            print(f"   hueco={gap:.3f} (margen normalizado {gap / spread:.2f}), errores con umbral actual={err}")
            if gap > 0:
                print(f"   umbral propuesto (punto medio) = {prop:.4f}")
                if a.write:
                    node["threshold"] = round(float(prop), 4)
            else:
                print("   !! Las clases se SOLAPAN en esta caracteristica: revisar nodo")
        walk(node["left"], path + ".si")
        walk(node["right"], path + ".no")

    walk(cfg["tree"])
    if a.write:
        path = a.config or os.path.join(vision.HERE, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print("\nUmbrales escritos en", path)


# ---------------------------------------------------------------------------
def _best_split(rows, classes, feats):
    """Para cada caracteristica ordena las clases por su media y prueba los cortes
    entre clases consecutivas. Puntuacion: (errores, -hueco/desviacion)."""
    best = None
    for k in feats:
        means = sorted(classes, key=lambda c: np.mean([r[k] for r in rows if r["class"] == c]))
        for i in range(1, len(means)):
            lc, rc = set(means[:i]), set(means[i:])
            L, R = _split_stats(rows, k, lc, rc)
            if not len(L) or not len(R):
                continue
            gap = R.min() - L.max()
            thr = (R.min() + L.max()) / 2
            if gap <= 0:  # solapan: umbral que minimiza errores
                cands = np.unique(np.concatenate([L, R]))
                errs = [np.sum(L > t) + np.sum(R <= t) for t in cands]
                thr = float(cands[int(np.argmin(errs))])
            err = int(np.sum(L > thr) + np.sum(R <= thr))
            spread = np.std(np.concatenate([L, R])) + 1e-9
            # preferir cortes equilibrados ante empate (arbol mas bajo)
            balance = min(len(lc), len(rc)) / len(classes)
            score = (err, -(gap / spread) - 0.1 * balance)
            if best is None or score < best[0]:
                best = (score, k, round(float(thr), 4), lc, rc, gap)
    return best


def _build(rows, classes, feats, depth=0):
    if len(classes) == 1:
        return {"label": next(iter(classes))}
    best = _best_split(rows, classes, feats)
    _, k, thr, lc, rc, gap = best
    sub = [r for r in rows if r["class"] in classes]
    return {"feature": k, "threshold": thr,
            "why": f"separa {sorted(lc)} de {sorted(rc)}; hueco medido={gap:.3f}",
            "left": _build(sub, lc, feats, depth + 1),
            "right": _build(sub, rc, feats, depth + 1)}


def cmd_suggest(a, cfg):
    rows, feats = _load_csv(a.csv)
    if a.features:
        feats = a.features.split(",")
    classes = {r["class"] for r in rows}
    t = _build(rows, classes, feats)
    print(to_text(t))
    print(to_mermaid(t))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(t, f, indent=2, ensure_ascii=False)
    print("\nArbol sugerido guardado en", a.out, "(copiar a config.json -> 'tree' tras revisarlo)")


def cmd_plots(a, cfg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib")
        return
    rows, feats = _load_csv(a.csv)
    os.makedirs(a.out, exist_ok=True)
    classes = [c for c in cfg["classes"] if any(r["class"] == c for r in rows)]
    for k in feats:
        fig, ax = plt.subplots(figsize=(6, 3))
        data = [[r[k] for r in rows if r["class"] == c] for c in classes]
        ax.boxplot(data, labels=classes)
        ax.set_title(k)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"box_{k}.png"), dpi=120)
        plt.close(fig)
    print("Figuras en", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("felt"); p.add_argument("img")
    p = sp.add_parser("resolution"); p.add_argument("img")
    p = sp.add_parser("features"); p.add_argument("dir"); p.add_argument("--csv", default="features.csv")
    p = sp.add_parser("thresholds"); p.add_argument("csv"); p.add_argument("--write", action="store_true")
    p = sp.add_parser("suggest"); p.add_argument("csv"); p.add_argument("--out", default="tree_sugerido.json")
    p.add_argument("--features", default=None, help="lista separada por comas de caracteristicas permitidas")
    p = sp.add_parser("plots"); p.add_argument("csv"); p.add_argument("--out", default="figuras")
    a = ap.parse_args()
    cfg = vision.load_config(a.config)
    {"felt": cmd_felt, "resolution": cmd_resolution, "features": cmd_features,
     "thresholds": cmd_thresholds, "suggest": cmd_suggest, "plots": cmd_plots}[a.cmd](a, cfg)


if __name__ == "__main__":
    main()
