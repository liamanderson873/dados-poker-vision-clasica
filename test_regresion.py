"""Regresion real: recuperacion de seis dados y ausencia de falsas detecciones."""
from collections import Counter
from pathlib import Path
import sys
import unittest
import cv2
import vision

CFG = vision.load_config()

class FotosReales(unittest.TestCase):
    def test_fila_sin_confundir_carton_con_dados(self):
        r = vision.analyze(cv2.imread('fotos_reales/caja/chat_fila.jpg'), CFG)
        self.assertEqual(Counter(d['label'] for d in r['dice']), Counter(['A','K','Q','J','8','7']))

    def test_seis_dados_en_n10(self):
        r = vision.analyze(cv2.imread('fotos_reales/borde2/n10.jpg'), CFG)
        self.assertEqual(Counter(d['label'] for d in r['dice']), Counter(['Q','J','J','A','8','J']))

    def test_caja_vacia(self):
        r = vision.analyze(cv2.imread('fotos_reales/caja_vacia.jpg'), CFG)
        self.assertEqual(r['dice'], [])

if __name__ == '__main__':
    unittest.main()
