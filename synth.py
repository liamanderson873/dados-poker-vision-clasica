"""
synth.py - Generador de escenas SINTETICAS para probar el software antes de
tener el montaje fisico. NO sustituye a las imagenes reales: los umbrales del
informe deben calibrarse con fotos del montaje real.

Uso:
    python synth.py escenas  out/test  --n 40          # escenas aleatorias + ground truth
    python synth.py calib    out/calib --n 4           # carpeta por clase (todas las caras iguales)
"""
import argparse
import json
import os
import random

import cv2
import numpy as np

CLASSES = ["7", "8", "J", "Q", "K", "A"]
# Imitacion de los dados reales: 7 = siete puntos negros, 8 = ocho puntos rojos,
# J y Q negras, K roja, As = un punto rojo. Color BGR.
BLACK, RED = (25, 25, 25), (40, 30, 210)
INK = {"7": BLACK, "8": RED, "J": BLACK, "Q": BLACK, "K": RED, "A": RED}
# Posiciones de los puntos en una rejilla 3x3 (fila, columna)
DOTS = {"7": [(0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2)],
        "8": [(0, 0), (0, 1), (0, 2), (1, 0.5), (1, 1.5), (2, 0), (2, 1), (2, 2)],
        "A": [(1, 1)]}


def die_face(label, size=200):
    """Cara de dado de alta resolucion: fondo marfil, esquinas redondeadas, simbolo centrado."""
    face = np.full((size, size, 3), (225, 238, 242), np.uint8)
    if label in DOTS:
        step = size * 0.24
        r = int(size * (0.07 if label != "A" else 0.09))
        for fy, fx in DOTS[label]:
            c = (int(size / 2 + (fx - 1) * step), int(size / 2 + (fy - 1) * step))
            cv2.circle(face, c, r, INK[label], -1, cv2.LINE_AA)
    else:
        txt_scale, th = 4.2, 12
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, txt_scale, th)
        org = ((size - tw) // 2, (size + tht) // 2)
        cv2.putText(face, label, org, cv2.FONT_HERSHEY_SIMPLEX, txt_scale, INK[label], th, cv2.LINE_AA)
    mask = np.zeros((size, size), np.uint8)
    r = size // 8
    cv2.rectangle(mask, (r, 0), (size - r, size), 255, -1)
    cv2.rectangle(mask, (0, r), (size, size - r), 255, -1)
    for cx, cy in [(r, r), (size - r, r), (r, size - r), (size - r, size - r)]:
        cv2.circle(mask, (cx, cy), r, 255, -1)
    return face, mask


def felt(w, h, rng):
    """Tapete verde con textura y gradiente de iluminacion."""
    base = np.zeros((h, w, 3), np.float32)
    base[:] = (45, 120, 40)
    noise = rng.normal(0, 8, (h, w, 1)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
    grad = 1.15 - 0.35 * np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    return np.clip(base * grad[..., None] + noise, 0, 255)


def paste(canvas, face, mask, center, side, angle):
    """Escala, rota y pega la cara con sombra suave."""
    f = cv2.resize(face, (side, side), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (side, side), interpolation=cv2.INTER_AREA)
    pad = side
    big_f = cv2.copyMakeBorder(f, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    big_m = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    c = (big_f.shape[1] / 2, big_f.shape[0] / 2)
    R = cv2.getRotationMatrix2D(c, angle, 1.0)
    big_f = cv2.warpAffine(big_f, R, big_f.shape[1::-1], flags=cv2.INTER_LINEAR)
    big_m = cv2.warpAffine(big_m, R, big_m.shape[1::-1], flags=cv2.INTER_LINEAR)
    H, W = big_m.shape
    x0, y0 = int(center[0] - W / 2), int(center[1] - H / 2)
    # Sombra desplazada
    sh = cv2.GaussianBlur(big_m, (0, 0), side * 0.08).astype(np.float32) / 255 * 0.45
    sx, sy = x0 + side // 12, y0 + side // 12
    region = canvas[sy:sy + H, sx:sx + W]
    if region.shape[:2] == sh.shape:
        region *= (1 - sh[..., None])
    region = canvas[y0:y0 + H, x0:x0 + W]
    a = (big_m.astype(np.float32) / 255)[..., None]
    region[:] = region * (1 - a) + big_f.astype(np.float32) * a


def make_scene(labels, rng, w=1800, h=1350, side=56):
    canvas = felt(w, h, rng)
    placed = []
    gt = []
    margin = side * 1.8  # el parche rotado ocupa 3*lado; dejar sitio para pegarlo entero
    for lab in labels:
        for _ in range(500):
            x, y = rng.uniform(margin, w - margin), rng.uniform(margin, h - margin)
            if all((x - px) ** 2 + (y - py) ** 2 > (side * 1.6) ** 2 for px, py in placed):
                break
        placed.append((x, y))
        s = int(side * rng.uniform(0.95, 1.05))
        face, mask = die_face(lab)
        paste(canvas, face, mask, (x, y), s, rng.uniform(0, 360))
        gt.append({"label": lab, "x": round(float(x), 1), "y": round(float(y), 1)})
    canvas += rng.normal(0, 3, canvas.shape)
    img = np.clip(canvas, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["escenas", "calib"])
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    if a.modo == "escenas":
        for i in range(a.n):
            k = random.choice([5, 5, 5, 4, 3, 6])  # numero de dados variable
            labels = [random.choice(CLASSES) for _ in range(k)]
            img, gt = make_scene(labels, rng)
            name = os.path.join(a.out, f"escena_{i:03d}")
            cv2.imwrite(name + ".png", img)
            with open(name + ".json", "w", encoding="utf-8") as f:
                json.dump({"labels": labels, "dice": gt}, f, indent=1)
    else:
        for c in CLASSES:
            d = os.path.join(a.out, c)
            os.makedirs(d, exist_ok=True)
            for i in range(a.n):
                img, _ = make_scene([c] * 5, rng)
                cv2.imwrite(os.path.join(d, f"{c}_{i:02d}.png"), img)
    print("ok ->", a.out)


if __name__ == "__main__":
    main()
