"""
evaluate.py - Medida cuantitativa de aciertos y errores en los 3 niveles.

Cada imagen de prueba X.jpg tiene su ground truth X.json:
    {"labels": ["K", "K", "9", "A", "10"],                 <- obligatorio
     "dice": [{"label": "K", "x": 812, "y": 400}, ...]}     <- opcional (posiciones)
     "hand": "Pareja"                                       <- opcional (si no, se calcula de labels)

Con posiciones: emparejamiento por distancia (< 0.75 lado del dado).
Sin posiciones: comparacion de multiconjuntos de etiquetas (no permite matriz de confusion exacta).

Uso:
    python evaluate.py DIR [--out resultados]
    python evaluate.py DIR --bootstrap     # crea X.json con lo que predice el sistema
                                           # (CORREGIRLOS A MANO antes de evaluar)
"""
import argparse
import csv
import glob
import json
import os
from collections import Counter

import cv2
import numpy as np

import vision
from hand import evaluate_hand


def _match(pred, gt, max_d):
    """Emparejamiento voraz por distancia minima."""
    pairs, used = [], set()
    cand = sorted(((np.hypot(p["center"][0] - g["x"], p["center"][1] - g["y"]), i, j)
                   for i, p in enumerate(pred) for j, g in enumerate(gt)))
    used_p = set()
    for d, i, j in cand:
        if d > max_d or i in used_p or j in used:
            continue
        used_p.add(i)
        used.add(j)
        pairs.append((i, j))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default="resultados")
    ap.add_argument("--config", default=None)
    ap.add_argument("--bootstrap", action="store_true")
    a = ap.parse_args()
    cfg = vision.load_config(a.config)
    classes = cfg["classes"]
    imgs = sorted(p for e in ("*.jpg", "*.jpeg", "*.png") for p in glob.glob(os.path.join(a.dir, e)))
    os.makedirs(a.out, exist_ok=True)

    conf = {t: Counter() for t in classes + ["(fondo)"]}  # conf[real][pred]
    tp = fp = fn = 0
    id_ok = id_tot = 0
    hand_ok = scene_ok = 0
    n_scenes = 0
    rows = []

    for p in imgs:
        img = cv2.imread(p)
        res = vision.analyze(img, cfg)
        js = os.path.splitext(p)[0] + ".json"
        pred = res["dice"]
        if a.bootstrap:
            with open(js, "w", encoding="utf-8") as f:
                json.dump({"labels": [d["label"] for d in pred],
                           "dice": [{"label": d["label"], "x": round(d["center"][0]),
                                     "y": round(d["center"][1])} for d in pred]}, f, indent=1)
            cv2.imwrite(os.path.join(a.out, os.path.basename(p) + "_check.png"), vision.draw(img, res))
            continue
        if not os.path.exists(js):
            print("(sin ground truth)", p)
            continue
        gt = json.load(open(js, encoding="utf-8"))
        n_scenes += 1
        gl = gt["labels"]
        gh = gt.get("hand") or evaluate_hand(gl, classes)["name"]
        pl = [d["label"] for d in pred]

        if gt.get("dice"):
            side = np.median([d["side_px"] for d in pred]) if pred else 50
            pairs = _match(pred, gt["dice"], 0.75 * side)
            s_tp = len(pairs)
            s_fp = len(pred) - s_tp
            s_fn = len(gt["dice"]) - s_tp
            s_ok = 0
            for i, j in pairs:
                conf[gt["dice"][j]["label"]][pred[i]["label"]] += 1
                s_ok += pred[i]["label"] == gt["dice"][j]["label"]
            mi = {i for i, _ in pairs}
            mj = {j for _, j in pairs}
            for i, d in enumerate(pred):
                if i not in mi:
                    conf["(fondo)"][d["label"]] += 1
            for j, g in enumerate(gt["dice"]):
                if j not in mj:
                    conf[g["label"]]["(no detectado)"] += 1
        else:
            s_tp = min(len(pl), len(gl))
            s_fp, s_fn = max(0, len(pl) - len(gl)), max(0, len(gl) - len(pl))
            s_ok = sum((Counter(pl) & Counter(gl)).values())

        tp, fp, fn = tp + s_tp, fp + s_fp, fn + s_fn
        id_ok += s_ok
        id_tot += s_tp
        h_ok = res["hand"]["name"] == gh
        hand_ok += h_ok
        sc_ok = Counter(pl) == Counter(gl)
        scene_ok += sc_ok
        rows.append({"imagen": os.path.basename(p), "dados_real": len(gl), "dados_pred": len(pl),
                     "real": " ".join(gl), "pred": " ".join(pl), "id_ok": s_ok,
                     "jugada_real": gh, "jugada_pred": res["hand"]["name"], "jugada_ok": int(h_ok)})
        if not (h_ok and sc_ok):
            cv2.imwrite(os.path.join(a.out, "ERROR_" + os.path.basename(p)), vision.draw(img, res))

    if a.bootstrap:
        print("Ground truth provisional creado. Revisa y CORRIGE cada .json antes de evaluar.")
        return
    if not n_scenes:
        print("No hay escenas con ground truth")
        return

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    lines = [
        f"# Resultados ({n_scenes} escenas, {tp + fn} dados reales)\n",
        "| Nivel | Metrica | Valor |", "|---|---|---|",
        f"| 1 Deteccion | Precision | {prec:.3f} ({tp}/{tp + fp}) |",
        f"| 1 Deteccion | Exhaustividad (recall) | {rec:.3f} ({tp}/{tp + fn}) |",
        f"| 1 Deteccion | F1 | {f1:.3f} |",
        f"| 2 Identificacion | Exactitud sobre dados detectados | {id_ok / max(1, id_tot):.3f} ({id_ok}/{id_tot}) |",
        f"| 2 Identificacion | Escenas con todas las caras correctas | {scene_ok / n_scenes:.3f} ({scene_ok}/{n_scenes}) |",
        f"| 3 Interpretacion | Jugada correcta | {hand_ok / n_scenes:.3f} ({hand_ok}/{n_scenes}) |",
        "",
    ]
    cols = classes + ["(no detectado)"]
    if any(sum(conf[c].values()) for c in conf):
        lines += ["## Matriz de confusion (filas = real, columnas = prediccion)\n",
                  "| real \\ pred | " + " | ".join(cols) + " |",
                  "|---" * (len(cols) + 1) + "|"]
        for r in classes + ["(fondo)"]:
            lines.append(f"| {r} | " + " | ".join(str(conf[r][c]) for c in cols) + " |")
        lines.append("")
        lines.append("| Clase | Precision | Recall |\n|---|---|---|")
        for c in classes:
            tpc = conf[c][c]
            pc = sum(conf[r][c] for r in conf)
            rc = sum(conf[c].values())
            lines.append(f"| {c} | {tpc / max(1, pc):.3f} | {tpc / max(1, rc):.3f} |")
    md = "\n".join(lines)
    print(md)
    with open(os.path.join(a.out, "resultados.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    with open(os.path.join(a.out, "por_escena.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nGuardado en {a.out}/resultados.md y por_escena.csv (escenas fallidas: ERROR_*.png)")


if __name__ == "__main__":
    main()
