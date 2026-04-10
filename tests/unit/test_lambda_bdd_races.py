"""
tests/unit/test_lambda_bdd_races_enhanced.py
Pruebas adicionales para robustecer la cobertura de lambda_bdd_races
"""

import sys
import os
import json
import pytest
import re
import importlib.util
from datetime import datetime
from unittest.mock import patch, MagicMock

# =============================================================
# IMPORTANTE: Añadir la ruta raíz para que encuentre glue_utils
# =============================================================
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../lambda/lambda_bdd_races/lambda_bdd_races.py'))
spec = importlib.util.spec_from_file_location("lambda_bdd", file_path)
lambda_bdd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lambda_bdd)
lambda_handler = lambda_bdd.lambda_handler


class TestLambdaBDDRacesEnhanced:
    """Pruebas adicionales para robustecer la cobertura"""
    
    # =============================================================
    # TESTS QUE EJECUTAN LA LAMBDA COMPLETA (necesitan mocks)
    # =============================================================
    
    @patch('boto3.client')
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    @patch('glue_utils.calcular_cobertura_carrera')
    @patch('glue_utils.buscar_mejor_combinacion_fallback')
    def test_lambda_mixed_valid_invalid_carreras(self, mock_buscar, mock_calcular, mock_cargar, mock_listar, mock_boto):
        """Mezcla de carreras válidas e inválidas - debe omitir solo las inválidas"""
        mock_boto.return_value = MagicMock()
        mock_listar.return_value = ["carrera-2024"]
        mock_cargar.return_value = {"events": [{"name": "Maraton", "splits": [{"distance": 5000}]}]}
        mock_calcular.return_value = {
            'cobertura': 1.0,
            'puntos_cubiertos': [5000],
            'puntos_faltantes': [],
            'evento_elegido': 'Maraton'
        }
        mock_buscar.return_value = {
            'seleccionadas': ['carrera-2024'],
            'cobertura_total': 1.0,
            'puntos_cubiertos': [5000],
            'puntos_faltantes': []
        }
        
        event = {
            "carreras": [
                {"nombre": "Valida1-2025", "splits": ["km_5"]},
                {"splits": ["km_10"]},
                {"nombre": "Valida2-2025", "splits": ["km_15"]},
                {"nombre": "Invalida-2025", "splits": []},
                {"nombre": "Valida3-2025", "splits": ["km_20", "km_25"]},
                {"nombre": None, "splits": ["km_30"]},
                {"nombre": "Valida4-2025", "splits": ["km_35"]}
            ]
        }
        result = lambda_handler(event, None)
        
        assert result["num_modelos"] == 4
        carreras_procesadas = [c["carrera_objetivo"] for c in result["carreras_config"]]
        assert carreras_procesadas == ["Valida1-2025", "Valida2-2025", "Valida3-2025", "Valida4-2025"]
    
    @patch('boto3.client')
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    @patch('glue_utils.calcular_cobertura_carrera')
    @patch('glue_utils.buscar_mejor_combinacion_fallback')
    def test_lambda_nombre_prioridad_sobre_carrera_nuevo_formato(self, mock_buscar, mock_calcular, mock_cargar, mock_listar, mock_boto):
        """En nuevo formato, 'nombre' tiene prioridad sobre 'carrera'"""
        mock_boto.return_value = MagicMock()
        mock_listar.return_value = ["carrera-2024"]
        mock_cargar.return_value = {"events": [{"name": "Maraton", "splits": [{"distance": 5000}]}]}
        mock_calcular.return_value = {
            'cobertura': 1.0,
            'puntos_cubiertos': [5000],
            'puntos_faltantes': [],
            'evento_elegido': 'Maraton'
        }
        mock_buscar.return_value = {
            'seleccionadas': ['carrera-2024'],
            'cobertura_total': 1.0,
            'puntos_cubiertos': [5000],
            'puntos_faltantes': []
        }
        
        event = {
            "carreras": [
                {
                    "nombre": "NombrePrioritario-2025",
                    "carrera": "NombreIgnorado-2025",
                    "splits": ["km_5"]
                }
            ]
        }
        result = lambda_handler(event, None)
        
        assert result["num_modelos"] == 1
        assert result["carreras_config"][0]["carrera_objetivo"] == "NombrePrioritario-2025"
    
    @patch('boto3.client')
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    @patch('glue_utils.calcular_cobertura_carrera')
    @patch('glue_utils.buscar_mejor_combinacion_fallback')
    def test_lambda_training_params_extreme_values(self, mock_buscar, mock_calcular, mock_cargar, mock_listar, mock_boto):
        """Valores extremos en training_params"""
        mock_boto.return_value = MagicMock()
        mock_listar.return_value = ["carrera-2024"]
        mock_cargar.return_value = {"events": [{"name": "Maraton", "splits": [{"distance": 5000}]}]}
        mock_calcular.return_value = {
            'cobertura': 1.0,
            'puntos_cubiertos': [5000],
            'puntos_faltantes': [],
            'evento_elegido': 'Maraton'
        }
        mock_buscar.return_value = {
            'seleccionadas': ['carrera-2024'],
            'cobertura_total': 1.0,
            'puntos_cubiertos': [5000],
            'puntos_faltantes': []
        }
        
        training_params = {
            "n_estimators": 0,
            "max_depth": 1000,
            "learning_rate": -0.5,
            "min_samples_split": 1,
            "subsample": 1.5,
            "param_con_puntos": 1.2e-10,
            "param_con_unicode": "🏃‍♂️ valor",
            "param_con_espacios": "valor con espacios",
            "param_vacio": None,
            "param_lista": [1, 2, 3],
            "param_dict": {"sub": "value"}
        }
        
        event = {
            "carrera": "Test-2025",
            "splits": ["km_5"],
            "training_params": training_params
        }
        
        result = lambda_handler(event, None)
        config = result["carreras_config"][0]
        
        assert config["training_params"] == training_params
    
    # =============================================================
    # TESTS DE VALIDACIÓN (fallan ANTES de la selección, no necesitan mocks)
    # =============================================================
    
    def test_lambda_empty_splits_list_explicit(self):
        """Splits como lista vacía explícita - debe fallar"""
        event = {"carreras": [{"nombre": "Carrera1-2025", "splits": []}]}
        with pytest.raises(Exception) as exc_info:
            lambda_handler(event, None)
        assert "No se pudo preparar ninguna carrera" in str(exc_info.value)
    
    def test_lambda_splits_none_value(self):
        """Splits como None - debe fallar con ValueError"""
        event = {"carreras": [{"nombre": "Carrera1-2025", "splits": None}]}
        with pytest.raises(ValueError) as exc_info:
            lambda_handler(event, None)
        assert "splits debe ser una lista" in str(exc_info.value)
    
    def test_lambda_splits_wrong_type_string(self):
        """splits como string - debe fallar"""
        event = {"carreras": [{"nombre": "Carrera1-2025", "splits": "km_5"}]}
        with pytest.raises(ValueError) as exc_info:
            lambda_handler(event, None)
        assert "splits debe ser una lista" in str(exc_info.value)
    
    def test_lambda_splits_wrong_type_int(self):
        """splits como entero - debe fallar"""
        event = {"carreras": [{"nombre": "Carrera1-2025", "splits": 123}]}
        with pytest.raises(ValueError) as exc_info:
            lambda_handler(event, None)
        assert "splits debe ser una lista" in str(exc_info.value)
    
    def test_lambda_tipo_modelo_wrong_type(self):
        """tipo_modelo como número - debe fallar"""
        event = {"carreras": [{"nombre": "Carrera1-2025", "splits": ["km_5"], "tipo_modelo": 123}]}
        with pytest.raises(ValueError) as exc_info:
            lambda_handler(event, None)
        assert "tipo_modelo debe ser" in str(exc_info.value)
    
    def test_lambda_training_params_wrong_type(self):
        """training_params como string - debe fallar"""
        event = {"carreras": [{"nombre": "Carrera1-2025", "splits": ["km_5"], "training_params": "no_es_dict"}]}
        with pytest.raises(ValueError) as exc_info:
            lambda_handler(event, None)
        assert "training_params debe ser un diccionario" in str(exc_info.value)
    
    # =============================================================
    # TESTS CON NOMBRES NULL o VACÍOS (necesitan mocks porque la lambda intenta seleccionar)
    # =============================================================
    
    @patch('boto3.client')
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    @patch('glue_utils.calcular_cobertura_carrera')
    @patch('glue_utils.buscar_mejor_combinacion_fallback')
    def test_lambda_nombre_none_usar_carrera(self, mock_buscar, mock_calcular, mock_cargar, mock_listar, mock_boto):
        """Si 'nombre' es None, usa 'carrera'"""
        mock_boto.return_value = MagicMock()
        mock_listar.return_value = ["carrera-2024"]
        mock_cargar.return_value = {"events": [{"name": "Maraton", "splits": [{"distance": 5000}]}]}
        mock_calcular.return_value = {
            'cobertura': 1.0,
            'puntos_cubiertos': [5000],
            'puntos_faltantes': [],
            'evento_elegido': 'Maraton'
        }
        mock_buscar.return_value = {
            'seleccionadas': ['carrera-2024'],
            'cobertura_total': 1.0,
            'puntos_cubiertos': [5000],
            'puntos_faltantes': []
        }
        
        event = {
            "carreras": [
                {
                    "nombre": None,
                    "carrera": "CarreraBackup-2025",
                    "splits": ["km_5"]
                }
            ]
        }
        result = lambda_handler(event, None)
        
        assert result["num_modelos"] == 1
        assert result["carreras_config"][0]["carrera_objetivo"] == "CarreraBackup-2025"
    
    @patch('boto3.client')
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    @patch('glue_utils.calcular_cobertura_carrera')
    @patch('glue_utils.buscar_mejor_combinacion_fallback')
    def test_lambda_nombre_vacio_usar_carrera(self, mock_buscar, mock_calcular, mock_cargar, mock_listar, mock_boto):
        """Si 'nombre' es string vacío, usa 'carrera'"""
        mock_boto.return_value = MagicMock()
        mock_listar.return_value = ["carrera-2024"]
        mock_cargar.return_value = {"events": [{"name": "Maraton", "splits": [{"distance": 5000}]}]}
        mock_calcular.return_value = {
            'cobertura': 1.0,
            'puntos_cubiertos': [5000],
            'puntos_faltantes': [],
            'evento_elegido': 'Maraton'
        }
        mock_buscar.return_value = {
            'seleccionadas': ['carrera-2024'],
            'cobertura_total': 1.0,
            'puntos_cubiertos': [5000],
            'puntos_faltantes': []
        }
        
        event = {
            "carreras": [
                {
                    "nombre": "",
                    "carrera": "CarreraBackup-2025",
                    "splits": ["km_5"]
                }
            ]
        }
        result = lambda_handler(event, None)
        
        assert result["num_modelos"] == 1
        assert result["carreras_config"][0]["carrera_objetivo"] == "CarreraBackup-2025"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])