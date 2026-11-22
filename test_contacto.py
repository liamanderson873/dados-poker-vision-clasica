import sys, cv2, numpy as np
sys.path.insert(0, ".")
import vision, synth
from pathlib import Path
Path("salida").mkdir(exist_ok=True)
cfg = vision.load_config("config.json")
rng = np.random.default_rng(3)
ok = 0; N = 20
for t in range(N):
    canvas = synth.felt(1800, 1350, rng)
    labs = [str(x) for x in rng.choice(synth.CLASSES, 5)]
    x0, y0 = rng.uniform(400, 1300), rng.uniform(400, 900)
    for i, l in enumerate(labs):
        f, m = synth.die_face(l)
        synth.paste(canvas, f, m, (x0 + i * 60, y0 + rng.uniform(-8, 8)), 56, rng.uniform(-10, 10))
    img = np.clip(canvas, 0, 255).astype(np.uint8)
    if t == 0: cv2.imwrite("salida/contacto_ejemplo.png", vision.draw(img, vision.analyze(img, cfg)))
    got = sorted(d["label"] for d in vision.analyze(img, cfg)["dice"])
    ok += got == sorted(labs)
    if got != sorted(labs): print(t, sorted(labs), got)
print(f"contacto: {ok}/{N} escenas correctas")

if ok != N:
    raise SystemExit(1)
