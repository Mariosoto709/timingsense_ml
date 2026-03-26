"""
tests/unit/test_sagemaker_utils.py
Pruebas unitarias para funciones auxiliares de SageMaker
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# Añadir la carpeta sagemaker al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sagemaker')))

from train_interpolacion import (
    ordenar_splits_personalizado,
    identificar_splits,
    procesar_genero,
    calcular_metricas_detalladas,
    nivel1_mejor_que_naive,
    nivel2_consistencia_error,
    nivel3_sin_outliers_catastroficos,
    validar_modelo_completo
)


# =============================================================
# TESTS PARA identificar_splits
# =============================================================

class TestIdentificarSplits:
    """Pruebas para identificar_splits"""
    
    def test_identificar_splits_km(self):
        """Identifica splits km_X correctamente"""
        df = pd.DataFrame({
            'km_5': [1800, 1900],
            'km_10': [3600, 3800],
            'km_15': [5400, 5700],
            'athlete_id': ['a1', 'a2'],
            'gender': ['M', 'F'],
            'age': [30, 25],
            'event_id': ['e1', 'e2'],
            'event_std': ['std1', 'std2']
        })
        splits = identificar_splits(df)
        
        assert "km_5" in splits
        assert "km_10" in splits
        assert "km_15" in splits
        assert "athlete_id" not in splits
        assert "gender" not in splits
        assert "age" not in splits
        assert "event_id" not in splits
        assert "event_std" not in splits
    
    def test_identificar_splits_especiales(self):
        """Identifica splits especiales (half, finish, start)"""
        df = pd.DataFrame({
            'half': [6300, 6500],
            'finish': [12600, 13000],
            'start': [0, 0],
            'athlete_id': ['a1', 'a2']
        })
        splits = identificar_splits(df)
        
        assert "half" in splits
        assert "finish" in splits
        assert "start" in splits
    
    def test_identificar_splits_ignora_rawtime(self):
        """Ignora columnas que empiezan con rawtime_"""
        df = pd.DataFrame({
            'km_5': [1800],
            'rawtime_km_5': [1800],
            'athlete_id': ['a1']
        })
        splits = identificar_splits(df)
        
        assert "km_5" in splits
        assert "rawtime_km_5" not in splits
    
    def test_identificar_splits_vacio(self):
        """DataFrame sin splits → lista vacía"""
        df = pd.DataFrame({
            'athlete_id': ['a1'],
            'gender': ['M'],
            'age': [30]
        })
        splits = identificar_splits(df)
        
        assert splits == []


# =============================================================
# TESTS PARA procesar_genero
# =============================================================

class TestProcesarGenero:
    """Pruebas para procesar_genero"""
    
    def test_procesar_genero_string_m_f(self):
        """Convierte 'M' y 'F' a 0 y 1"""
        df = pd.DataFrame({
            'gender': ['M', 'F', 'M', 'F']
        })
        df = procesar_genero(df)
        
        assert df['gender'].tolist() == [0, 1, 0, 1]
    
    def test_procesar_genero_string_male_female(self):
        """Convierte 'male' y 'female' a 0 y 1"""
        df = pd.DataFrame({
            'gender': ['male', 'female', 'Male', 'Female']
        })
        df = procesar_genero(df)
        
        assert df['gender'].tolist() == [0, 1, 0, 1]
    
    def test_procesar_genero_numeric(self):
        """Mantiene valores numéricos"""
        df = pd.DataFrame({
            'gender': [0, 1, 0, 1]
        })
        df = procesar_genero(df)
        
        assert df['gender'].tolist() == [0, 1, 0, 1]
    
    def test_procesar_genero_sin_columna(self):
        """Si no hay columna gender, devuelve df igual"""
        df = pd.DataFrame({'km_5': [1800, 1900]})
        original = df.copy()
        df = procesar_genero(df)
        
        assert df.equals(original)
    
    def test_procesar_genero_valores_desconocidos(self):
        """Valores desconocidos se convierten a 0"""
        df = pd.DataFrame({
            'gender': ['M', 'X', 'F', 'unknown']
        })
        df = procesar_genero(df)
        
        assert df['gender'].tolist() == [0, 0, 1, 0]  # X y unknown → 0


# =============================================================
# TESTS PARA calcular_metricas_detalladas
# =============================================================

class TestCalcularMetricasDetalladas:
    """Pruebas para calcular_metricas_detalladas"""
    
    def test_calcular_metricas_basicas(self):
        """Cálculo básico de métricas"""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([105, 195, 305, 395, 505])
        
        metricas = calcular_metricas_detalladas(
            y_true, y_pred, "km_10", "km_5"
        )
        
        assert metricas["split_objetivo"] == "km_10"
        assert metricas["posicion_atleta"] == "km_5"
        assert metricas["n_samples"] == 5
        assert metricas["mae"] == 5.0  # (5+5+5+5+5)/5 = 5
        assert metricas["mape"] == pytest.approx(2.28, rel=0.1)  # ~1.9%
        assert metricas["r2"] > 0.99
    
    def test_calcular_metricas_con_outliers(self):
        """Métricas con outliers"""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([100, 200, 300, 400, 1000])  # último outlier
        
        metricas = calcular_metricas_detalladas(
            y_true, y_pred, "km_10", "km_5"
        )
        
        assert metricas["max_error"] == 500
        assert metricas["p95_error"] == pytest.approx(400, abs=0.1)
        assert metricas["p99_error"] == pytest.approx(480, abs=10)

    
    def test_calcular_metricas_perfectas(self):
        """Predicción perfecta"""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([100, 200, 300, 400, 500])
        
        metricas = calcular_metricas_detalladas(
            y_true, y_pred, "km_10", "km_5"
        )
        
        assert metricas["mae"] == 0
        assert metricas["rmse"] == 0
        assert metricas["r2"] == 1.0
        assert metricas["mape"] == 0


# =============================================================
# TESTS PARA nivel1_mejor_que_naive
# =============================================================

class TestNivel1MejorQueNaive:
    """Pruebas para nivel1_mejor_que_naive"""
    
    def test_modelo_mejor_que_naive(self):
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([105, 195, 305, 395, 505])
        
        resultado = nivel1_mejor_que_naive(y_true, y_pred)
        
        assert resultado["aprueba"] == True
        assert resultado["mejora"] > 0
        assert resultado["mae_modelo"] < resultado["mae_naive"]
    
    def test_modelo_peor_que_naive(self):
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([500, 400, 300, 200, 100])
        
        resultado = nivel1_mejor_que_naive(y_true, y_pred)
        
        assert resultado["aprueba"] == False
        assert resultado["mejora"] < 0


# =============================================================
# TESTS PARA nivel2_consistencia_error
# =============================================================

class TestNivel2ConsistenciaError:
    """Pruebas para nivel2_consistencia_error"""
    
    def test_error_consistente(self):
        errores = np.array([25, 28, 30, 32, 35, 38, 40, 42, 45, 48])
        
        resultado = nivel2_consistencia_error(errores)
        
        assert resultado["aprueba"] == True
        assert resultado["cv"] <= 0.5
    
    def test_error_inconsistente(self):
        errores = np.array([5, 10, 15, 20, 25, 30, 35, 200, 250, 300])
        
        resultado = nivel2_consistencia_error(errores)
        
        assert resultado["aprueba"] == False
        assert resultado["cv"] > 0.5


# =============================================================
# TESTS PARA nivel3_sin_outliers_catastroficos
# =============================================================

class TestNivel3SinOutliersCatastroficos:
    """Pruebas para nivel3_sin_outliers_catastroficos"""
    
    def test_sin_outliers(self):
        errores = np.array([25, 28, 30, 32, 35, 38, 40, 42, 45, 48])
        
        resultado = nivel3_sin_outliers_catastroficos(errores)
        
        assert resultado["aprueba"] == True
        assert resultado["relacion"] <= 3.0
    
    def test_con_outliers(self):
        errores = np.array([25, 28, 30, 32, 35, 38, 40, 42, 200, 250])
        
        resultado = nivel3_sin_outliers_catastroficos(errores)
        
        assert resultado["aprueba"] == False
        assert resultado["relacion"] > 3.0


# =============================================================
# TESTS PARA validar_modelo_completo
# =============================================================

class TestValidarModeloCompleto:
    """Pruebas para validar_modelo_completo"""
    
    def test_modelo_bueno(self):
        """Modelo que cumple los 3 criterios"""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([105, 195, 305, 395, 505])
        
        resultado = validar_modelo_completo(y_true, y_pred, "km_10", "km_5")
        
        assert resultado["aprobado"] == True
        assert resultado["puntuacion_calidad"] >= 60
        assert resultado["split_objetivo"] == "km_10"
        assert resultado["posicion_atleta"] == "km_5"
    
    def test_modelo_malo_por_mae(self):
        """Modelo rechazado por MAE alto"""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([0, 0, 600, 600, 600])  # errores grandes
        
        resultado = validar_modelo_completo(y_true, y_pred, "km_10", "km_5")
        
        assert resultado["aprobado"] == False
        assert resultado["niveles"]["mejor_que_naive"]["aprueba"] == False
    
    def test_modelo_malo_por_inconsistente(self):
        """Modelo rechazado por error inconsistente"""
        y_true = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        y_pred = np.array([100, 200, 300, 400, 500, 600, 700, 800, 500, 1500])  # outliers
        
        resultado = validar_modelo_completo(y_true, y_pred, "km_10", "km_5")
        
        # Puede fallar por inconsistencia o outliers
        assert resultado["aprobado"] == False
    
    def test_modelo_malo_por_outliers(self):
        """Modelo rechazado por outliers catastróficos"""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([100, 200, 300, 400, 1000])  # último outlier
        
        resultado = validar_modelo_completo(y_true, y_pred, "km_10", "km_5")
        
        assert resultado["aprobado"] == False
        # Debe fallar nivel 3 (outliers)
        assert resultado["niveles"]["sin_outliers"]["aprueba"] == False


# =============================================================
# TESTS PARA ordenar_splits_personalizado (ya existen, añadimos más)
# =============================================================

class TestOrdenarSplitsPersonalizado:
    """Pruebas adicionales para ordenar_splits_personalizado"""
    
    def test_ordenar_splits_con_media(self):
        """Ordena incluyendo 'media' como sinónimo de half"""
        splits = ["finish", "km_5", "media", "km_10"]
        ordenado = ordenar_splits_personalizado(splits)
        # media (21.0975) debe ir después de km_10 (10) y antes de finish (42.195)
        assert ordenado.index("km_5") < ordenado.index("km_10")
        assert ordenado.index("km_10") < ordenado.index("media")
        assert ordenado.index("media") < ordenado.index("finish")
    
    def test_ordenar_splits_con_meta(self):
        """Ordena incluyendo 'meta' como sinónimo de finish"""
        splits = ["meta", "km_5", "half", "km_10"]
        ordenado = ordenar_splits_personalizado(splits)
        assert ordenado[-1] == "meta"
    
    def test_ordenar_splits_con_salida(self):
        """Ordena incluyendo 'salida' como sinónimo de start"""
        splits = ["finish", "km_5", "salida", "km_10"]
        ordenado = ordenar_splits_personalizado(splits)
        assert ordenado[0] == "salida"
    
    def test_ordenar_splits_ya_ordenado(self):
        """Si ya está ordenado, se mantiene"""
        splits = ["start", "km_5", "km_10", "half", "finish"]
        ordenado = ordenar_splits_personalizado(splits)
        assert ordenado == splits

if __name__ == "__main__":
    pytest.main([__file__, "-v"])