"""
tests/unit/test_glue_utils.py
Pruebas unitarias para funciones auxiliares de Glue
"""

import sys
import os
import pytest

# Importar desde glue_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../glue')))

from glue_utils import (
    extract_split_distance,
    splits_son_equivalentes,
    get_split_type,
    find_closest_split,
    analyze_split_requirements
)


# =============================================================
# TESTS PARA extract_split_distance (ya existen, los mantenemos)
# =============================================================

class TestExtractSplitDistance:
    """Pruebas para extract_split_distance"""
    
    def test_splits_kilometricos_normales(self):
        assert extract_split_distance("km_5") == 5.0
        assert extract_split_distance("km_10") == 10.0
        assert extract_split_distance("km_42") == 42.0
    
    def test_splits_con_decimales(self):
        assert extract_split_distance("km_18_2") == 18.2
        assert extract_split_distance("km_21_0975") == 21.0975
    
    def test_splits_especiales(self):
        assert extract_split_distance("half") == 21.0975
        assert extract_split_distance("finish") == 42.195
        assert extract_split_distance("start") == 0.0
    
    def test_splits_con_punto(self):
        assert extract_split_distance("km_18.2") == 18.2
    
    def test_casos_borde(self):
        assert extract_split_distance(None) is None
        assert extract_split_distance("") is None
        assert extract_split_distance("km_") is None


# =============================================================
# TESTS PARA splits_son_equivalentes
# =============================================================

class TestSplitsSonEquivalentes:
    """Pruebas para splits_son_equivalentes"""
    
    def test_mismo_nombre(self):
        assert splits_son_equivalentes("km_5", "km_5") == True
    
    def test_punto_vs_guion(self):
        assert splits_son_equivalentes("km_18_2", "km_18.2") == True
    
    def test_half_equivalencia(self):
        assert splits_son_equivalentes("half", "km_21.0975") == True
    
    def test_finish_equivalencia(self):
        assert splits_son_equivalentes("finish", "km_42.195") == True
    
    def test_no_equivalentes(self):
        assert splits_son_equivalentes("km_5", "km_10") == False


# =============================================================
# TESTS PARA get_split_type
# =============================================================

class TestGetSplitType:
    """Pruebas para get_split_type"""
    
    def test_get_split_type_distance_km(self):
        assert get_split_type("km_5") == "distance"
        assert get_split_type("km_10") == "distance"
    
    def test_get_split_type_distance_half(self):
        assert get_split_type("half") == "distance"
    
    def test_get_split_type_distance_finish(self):
        assert get_split_type("finish") == "distance"
    
    def test_get_split_type_other(self):
        assert get_split_type("nombre_split") == "other"
        assert get_split_type(None) == "other"


# =============================================================
# TESTS PARA find_closest_split
# =============================================================

class TestFindClosestSplit:
    """Pruebas para find_closest_split"""
    
    def test_find_closest_split_exact(self):
        historical_splits = ["km_5", "km_10", "km_15"]
        closest, dist = find_closest_split(historical_splits, 10.0)
        assert closest == "km_10"
        assert dist == 10.0
    
    def test_find_closest_split_approximate(self):
        historical_splits = ["km_5", "km_10", "km_15"]
        closest, dist = find_closest_split(historical_splits, 12.0)
        assert closest == "km_10"
        assert dist == 10.0
    
    def test_find_closest_split_empty_list(self):
        result = find_closest_split([], 10.0)
        assert result == (None, None)


# =============================================================
# TESTS PARA analyze_split_requirements
# =============================================================

class TestAnalyzeSplitRequirements:
    """Pruebas para analyze_split_requirements"""
    
    @pytest.fixture
    def carreras_historicas_sample(self):
        return [
            {"splits": ["km_5", "km_10", "half", "finish"]},
            {"splits": ["km_5", "km_10", "km_15", "half", "finish"]}
        ]
    
    def test_analyze_split_requirements_direct(self, carreras_historicas_sample):
        splits_objetivo = ["km_5", "km_10"]
        resultado = analyze_split_requirements(splits_objetivo, carreras_historicas_sample)
        
        assert "km_5" in resultado["splits_directos"]
        assert "km_10" in resultado["splits_directos"]
        assert len(resultado["splits_interpolables"]) == 0
    
    def test_analyze_split_requirements_interpolate(self, carreras_historicas_sample):
        splits_objetivo = ["km_18_2"]
        resultado = analyze_split_requirements(splits_objetivo, carreras_historicas_sample)
        
        assert len(resultado["splits_directos"]) == 0
        assert len(resultado["splits_interpolables"]) == 1
        assert resultado["mapping"]["km_18_2"][0] == "interpolate"
    
    def test_analyze_split_requirements_impossible(self, carreras_historicas_sample):
        splits_objetivo = ["split_inexistente"]
        resultado = analyze_split_requirements(splits_objetivo, carreras_historicas_sample)
        
        assert "split_inexistente" in resultado["splits_imposibles"]
        assert len(resultado["splits_finales"]) == 0


# tests/unit/test_glue_utils.py (añadir al final)

import pandas as pd
import numpy as np
from glue_utils import validar_calidad_datos


class TestValidarCalidadDatos:
    """Pruebas para validar_calidad_datos"""
    
    def test_validacion_datos_correctos(self):
        """Datos correctos deben pasar la validación"""
        df = pd.DataFrame({
            'athlete_id': [f'a{i}' for i in range(200)],
            'km_5': np.random.normal(1800, 100, 200),
            'km_10': np.random.normal(3600, 150, 200),
        })
        splits = ['km_5', 'km_10']
        
        resultado = validar_calidad_datos(df, splits)
        
        assert resultado['valido'] == True
        assert len(resultado['errores']) == 0
        assert resultado['metricas']['n_registros'] == 200
    
    def test_validacion_pocos_registros(self):
        """Pocos registros debe fallar"""
        df = pd.DataFrame({
            'athlete_id': [f'a{i}' for i in range(50)],
            'km_5': np.random.normal(1800, 100, 50),
        })
        splits = ['km_5']
        
        resultado = validar_calidad_datos(df, splits, umbral_min_registros=100)
        
        assert resultado['valido'] == False
        assert "Registros insuficientes" in resultado['errores'][0]
        assert resultado['metricas']['n_registros'] == 50
    
    def test_validacion_nulos_excesivos_error(self):
        """Nulos excesivos (>50%) debe fallar"""
        df = pd.DataFrame({
            'athlete_id': [f'a{i}' for i in range(200)],
            'km_5': [np.nan if i < 150 else 1800 for i in range(200)],  # 75% nulos
            'km_10': np.random.normal(3600, 150, 200),
        })
        splits = ['km_5', 'km_10']
        
        resultado = validar_calidad_datos(df, splits, umbral_nulos_error=0.5)
        
        assert resultado['valido'] == False
        assert "nulos" in resultado['errores'][0]
        assert resultado['metricas']['nulos_km_5'] > 0.7
    
    def test_validacion_nulos_warning(self):
        """Nulos moderados (20-50%) genera warning pero no falla"""
        df = pd.DataFrame({
            'athlete_id': [f'a{i}' for i in range(200)],
            'km_5': [np.nan if i < 60 else 1800 for i in range(200)],  # 30% nulos
            'km_10': np.random.normal(3600, 150, 200),
        })
        splits = ['km_5', 'km_10']
        
        resultado = validar_calidad_datos(df, splits, 
                                          umbral_nulos_warning=0.2,
                                          umbral_nulos_error=0.5)
        
        # Debe pasar (no error), pero tener warning
        assert resultado['valido'] == True
        assert len(resultado['warnings']) > 0
        assert "nulos" in resultado['warnings'][0]
    
    def test_validacion_split_inexistente(self):
        """Split que no existe en los datos debe fallar"""
        df = pd.DataFrame({
            'athlete_id': [f'a{i}' for i in range(200)],
            'km_5': np.random.normal(1800, 100, 200),
        })
        splits = ['km_5', 'km_inexistente']
        
        resultado = validar_calidad_datos(df, splits)
        
        assert resultado['valido'] == False
        assert "no encontrado" in resultado['errores'][0]
    
    def test_validacion_tiempos_negativos_warning(self):
        """Tiempos negativos generan warning"""
        df = pd.DataFrame({
            'athlete_id': [f'a{i}' for i in range(200)],
            'km_5': list(np.random.normal(1800, 100, 195)) + [-100, -200, -300, -400, -500],
        })
        splits = ['km_5']
        
        resultado = validar_calidad_datos(df, splits)
        
        assert len(resultado['warnings']) > 0
        assert "negativos" in resultado['warnings'][0]
        assert resultado['valido'] == True  # No falla, solo warning
    
    def test_validacion_tiempos_muy_grandes_warning(self):
        """Tiempos > 6 horas generan warning"""
        df = pd.DataFrame({
            'athlete_id': [f'a{i}' for i in range(200)],
            'km_5': list(np.random.normal(1800, 100, 195)) + [25000, 30000, 35000, 40000, 50000],
        })
        splits = ['km_5']
        
        resultado = validar_calidad_datos(df, splits, tiempo_maximo_segundos=21600)
        
        assert len(resultado['warnings']) > 0
        assert "horas" in resultado['warnings'][0]
        assert resultado['valido'] == True
    
    def test_validacion_todo_mal(self):
        """Múltiples problemas: pocos registros, nulos, split inexistente"""
        df = pd.DataFrame({
            'athlete_id': [f'a{i}' for i in range(30)],
            'km_5': [np.nan] * 30,  # todos nulos
        })
        splits = ['km_5', 'km_inexistente']
        
        resultado = validar_calidad_datos(df, splits, umbral_min_registros=100)
        
        assert resultado['valido'] == False
        assert len(resultado['errores']) >= 2  # múltiples errores
    
    def test_validacion_metricas_retornadas(self):
        """Verifica que las métricas se retornan correctamente"""
        df = pd.DataFrame({
            'athlete_id': [f'a{i}' for i in range(200)],
            'km_5': [1800] * 200,
            'km_10': [3600] * 200,
        })
        splits = ['km_5', 'km_10']
        
        resultado = validar_calidad_datos(df, splits)
        
        assert 'n_registros' in resultado['metricas']
        assert 'n_splits' in resultado['metricas']
        assert 'nulos_km_5' in resultado['metricas']
        assert 'nulos_km_10' in resultado['metricas']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])