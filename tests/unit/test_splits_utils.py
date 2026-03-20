"""
Pruebas para funciones de manejo de splits
"""

import sys
import os

# Añadir la carpeta sagemaker al path para poder importar
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sagemaker')))

# Importar las funciones que vamos a testear
from train_interpolacion import extract_split_distance

class TestExtractSplitDistance:
    """Pruebas para extract_split_distance"""

    def test_splits_kilometricos_normales(self):
        """km_5, km_10, etc."""
        assert extract_split_distance("km_5") == 5.0
        assert extract_split_distance("km_10") == 10.0
        assert extract_split_distance("km_42") == 42.0

    def test_splits_con_decimales(self):
        """km_18_2, km_21_0975"""
        assert extract_split_distance("km_18_2") == 18.2
        assert extract_split_distance("km_21_0975") == 21.0975
        assert extract_split_distance("km_42_195") == 42.195

    def test_splits_especiales(self):
        """half, finish, start"""
        assert extract_split_distance("half") == 21.0975
        assert extract_split_distance("finish") == 42.195
        assert extract_split_distance("start") == 0.0

    def test_splits_con_punto(self):
        """km_18.2, km_21.0975"""
        assert extract_split_distance("km_18.2") == 18.2
        assert extract_split_distance("km_21.0975") == 21.0975

    def test_casos_borde(self):
        """None, vacío, inválido"""
        assert extract_split_distance(None) is None
        assert extract_split_distance("") is None
        assert extract_split_distance("no_es_split") is None
        assert extract_split_distance("km_") is None