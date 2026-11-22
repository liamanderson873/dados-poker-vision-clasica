"""Sin margen reservado: las seis caras en centro, lados y esquinas.

Las caras de los extremos llegan al borde de la imagen; su simbolo sigue
completo. Son pruebas sinteticas de cobertura, no de perspectiva real.
"""
import sys
import unittest
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import synth
import vision


def scene(label):
    img = np.full((480, 640, 3), (45, 120, 40), np.uint8)
    face, mask = synth.die_face(label)
    face = cv2.resize(face, (60, 60))
    mask = cv2.resize(mask, (60, 60))
    for y in (0, 210, 420):
        for x in (0, 290, 580):
            patch = img[y:y+60, x:x+60]
            patch[mask > 127] = face[mask > 127]
    return img

class Cobertura(unittest.TestCase):
    def test_seis_caras_en_nueve_posiciones(self):
        cfg = vision.load_config()
        out = Path('audit_tapete'); out.mkdir(exist_ok=True)
        for label in synth.CLASSES:
            with self.subTest(cara=label):
                img = scene(label)
                result = vision.analyze(img, cfg)
                cv2.imwrite(str(out / f'{label}.jpg'), vision.draw(img, result))
                self.assertEqual(Counter(d['label'] for d in result['dice']), Counter({label: 9}))

if __name__ == '__main__':
    unittest.main()
