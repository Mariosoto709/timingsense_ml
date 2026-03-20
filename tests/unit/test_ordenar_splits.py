"""
Pruebas para ordenar_splits_personalizado
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sagemaker')))

from train_interpolacion import ordenar_splits_personalizado


class TestOrdenarSplits:
    """Pruebas para ordenar_splits_personalizado"""

    def test_ordenar_splits_simples(self):
        splits = ["km_10", "km_5", "km_20", "km_15"]
        ordenado = ordenar_splits_personalizado(splits)
        assert ordenado == ["km_5", "km_10", "km_15", "km_20"]

    def test_ordenar_con_especiales(self):
        splits = ["km_42_195", "km_5", "half", "km_10"]
        ordenado = ordenar_splits_personalizado(splits)
        assert ordenado == ["km_5", "km_10", "half", "km_42_195"]

    def test_ordenar_con_decimales(self):
        splits = ["km_18_2", "km_5", "km_21_0975", "km_10"]
        ordenado = ordenar_splits_personalizado(splits)
        assert ordenado == ["km_5", "km_10", "km_18_2", "km_21_0975"]

    def test_ordenar_con_start(self):
        splits = ["finish", "km_10", "start", "km_5"]
        ordenado = ordenar_splits_personalizado(splits)
        assert ordenado == ["start", "km_5", "km_10", "finish"]