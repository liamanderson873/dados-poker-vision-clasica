"""Regresion de las seis fotos reales recibidas para la narracion."""
from pathlib import Path
from collections import Counter
import unittest
import cv2
import vision
class Narracion(unittest.TestCase):
 def test_seis_escenas(self):
  cfg=vision.load_config();base=Path(__file__).parent/'fotos_reales/narracion_20261122'
  cases={'poker':['K','K','K','K','A'],'escalera':['7','8','J','Q','K'],'full':['Q','Q','Q','A','A'],'pegados':['7','8','J','Q','K'],'tres':['J','J','A'],'vacio':[]}
  for name,expected in cases.items():
   with self.subTest(foto=name):
    image=cv2.imread(str(base/(name+'.jpeg')));self.assertIsNotNone(image)
    r=vision.analyze(image,cfg);self.assertEqual(Counter(d['label'] for d in r['dice']),Counter(expected))
if __name__=='__main__':unittest.main()
