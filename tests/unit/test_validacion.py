"""
Pruebas para funciones de validación de 3 niveles
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sagemaker')))

from train_interpolacion import (
    nivel1_mejor_que_naive,
    nivel2_consistencia_error,
    nivel3_sin_outliers_catastroficos,
    validar_modelo_completo
)


class TestNivel1MejorQueNaive:
    def test_modelo_mejor(self):
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([105, 195, 305, 395, 505])
        resultado = nivel1_mejor_que_naive(y_true, y_pred)
        assert resultado['aprueba'] == True
        assert resultado['mejora'] > 0

    def test_modelo_peor(self):
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([500, 400, 300, 200, 100])
        resultado = nivel1_mejor_que_naive(y_true, y_pred)
        assert resultado['aprueba'] == False
        assert resultado['mejora'] < 0


class TestNivel2ConsistenciaError:
    def test_error_consistente(self):
        errores = np.array([25, 28, 30, 32, 35, 38, 40, 42, 45, 48])
        resultado = nivel2_consistencia_error(errores)
        assert resultado['aprueba'] == True
        assert resultado['cv'] <= 0.5

    def test_error_inconsistente(self):
        errores = np.array([5, 10, 15, 20, 25, 30, 35, 200, 250, 300])
        resultado = nivel2_consistencia_error(errores)
        assert resultado['aprueba'] == False
        assert resultado['cv'] > 0.5


class TestNivel3SinOutliers:
    def test_sin_outliers(self):
        errores = np.array([25, 28, 30, 32, 35, 38, 40, 42, 45, 48])
        resultado = nivel3_sin_outliers_catastroficos(errores)
        assert resultado['aprueba'] == True
        assert resultado['relacion'] <= 3.0

    def test_con_outliers(self):
        errores = np.array([25, 28, 30, 32, 35, 38, 40, 42, 200, 250])
        resultado = nivel3_sin_outliers_catastroficos(errores)
        assert resultado['aprueba'] == False
        assert resultado['relacion'] > 3.0


class TestValidacionCompleta:
    def test_modelo_bueno(self):
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([105, 195, 305, 395, 505])
        resultado = validar_modelo_completo(y_true, y_pred, "km_10", "km_5")
        assert resultado['aprobado'] == True
        assert resultado['puntuacion_calidad'] >= 60

    def test_modelo_malo(self):
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([500, 400, 300, 200, 100])
        resultado = validar_modelo_completo(y_true, y_pred, "km_10", "km_5")
        assert resultado['aprobado'] == False
        assert resultado['puntuacion_calidad'] < 60