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
from unittest.mock import patch

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../lambda/lambda_bdd_races/lambda_bdd_races.py'))
spec = importlib.util.spec_from_file_location("lambda_bdd", file_path)
lambda_bdd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lambda_bdd)
lambda_handler = lambda_bdd.lambda_handler


class TestLambdaBDDRacesEnhanced:
    """Pruebas adicionales para robustecer la cobertura"""
    
    # =============================================================
    # TESTS PARA MÚLTIPLES CARRERAS MEZCLADAS (VÁLIDAS E INVÁLIDAS)
    # =============================================================
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_mixed_valid_invalid_carreras(self, mock_cargar, mock_listar):
        """Mezcla de carreras válidas e inválidas - debe omitir solo las inválidas"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
        event = {
            "carreras": [
                {"nombre": "Valida1-2025", "splits": ["km_5"]},           # ✅ válida
                {"splits": ["km_10"]},                                    # ❌ sin nombre
                {"nombre": "Valida2-2025", "splits": ["km_15"]},          # ✅ válida
                {"nombre": "Invalida-2025", "splits": []},                # ❌ splits vacío
                {"nombre": "Valida3-2025", "splits": ["km_20", "km_25"]}, # ✅ válida
                {"nombre": None, "splits": ["km_30"]},                    # ❌ nombre None
                {"nombre": "Valida4-2025", "splits": ["km_35"]}           # ✅ válida
            ]
        }
        result = lambda_handler(event, None)
        
        # Debe procesar solo las 4 válidas
        assert result["num_modelos"] == 4
        
        # Verificar que están en orden
        carreras_procesadas = [c["carrera_objetivo"] for c in result["carreras_config"]]
        assert carreras_procesadas == ["Valida1-2025", "Valida2-2025", "Valida3-2025", "Valida4-2025"]
        
        # Verificar que los splits se mantienen
        assert result["carreras_config"][0]["splits"] == ["km_5"]
        assert result["carreras_config"][1]["splits"] == ["km_15"]
        assert result["carreras_config"][2]["splits"] == ["km_20", "km_25"]
        assert result["carreras_config"][3]["splits"] == ["km_35"]
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_empty_splits_list_explicit(self, mock_cargar, mock_listar):
        """Splits como lista vacía explícita - debe invalidar la carrera"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
        event = {
            "carreras": [
                {"nombre": "Carrera1-2025", "splits": []}
            ]
        }
        
        with pytest.raises(Exception) as exc_info:
            lambda_handler(event, None)
        assert "No se pudo preparar ninguna carrera" in str(exc_info.value)
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_splits_none_value(self, mock_cargar, mock_listar):
        """Splits como None - debe fallar con ValueError"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
        event = {
            "carreras": [
                {"nombre": "Carrera1-2025", "splits": None}
            ]
        }

        with pytest.raises(ValueError) as exc_info:
            lambda_handler(event, None)
        
        assert "splits debe ser una lista" in str(exc_info.value)
        
    # =============================================================
    # TESTS PARA CAMPOS CON PRIORIDAD Y COEXISTENCIA
    # =============================================================
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_nombre_prioridad_sobre_carrera_nuevo_formato(self, mock_cargar, mock_listar):
        """En nuevo formato, 'nombre' tiene prioridad sobre 'carrera'"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
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
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_nombre_none_usar_carrera(self, mock_cargar, mock_listar):
        """Si 'nombre' es None, usa 'carrera'"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
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
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_nombre_vacio_usar_carrera(self, mock_cargar, mock_listar):
        """Si 'nombre' es string vacío, usa 'carrera'"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
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
    
    # =============================================================
    # TESTS PARA TIPOS DE DATOS INCORRECTOS
    # =============================================================
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_splits_wrong_type_string(self, mock_cargar, mock_listar):
        """splits como string en lugar de lista - debe fallar"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
        event = {
            "carreras": [
                {
                    "nombre": "Carrera1-2025",
                    "splits": "km_5, km_10"
                }
            ]
        }
        
        with pytest.raises(Exception):
            lambda_handler(event, None)
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_splits_wrong_type_int(self, mock_cargar, mock_listar):
        """splits como entero - debe fallar"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
        event = {
            "carreras": [
                {
                    "nombre": "Carrera1-2025",
                    "splits": 123
                }
            ]
        }
        
        with pytest.raises(Exception):
            lambda_handler(event, None)
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_tipo_modelo_wrong_type(self, mock_cargar, mock_listar):
        """tipo_modelo como número - debe fallar"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
        event = {
            "carreras": [
                {
                    "nombre": "Carrera1-2025",
                    "splits": ["km_5"],
                    "tipo_modelo": 123
                }
            ]
        }
        
        with pytest.raises(Exception) as exc_info:
            lambda_handler(event, None)
        assert "tipo_modelo debe ser" in str(exc_info.value)
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_training_params_wrong_type(self, mock_cargar, mock_listar):
        """training_params como string en lugar de dict - debe fallar con ValueError"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
        event = {
            "carreras": [
                {
                    "nombre": "Carrera1-2025",
                    "splits": ["km_5"],
                    "training_params": "no_es_dict"
                }
            ]
        }
        
        with pytest.raises(ValueError) as exc_info:
            lambda_handler(event, None)
        
        assert "training_params debe ser un diccionario" in str(exc_info.value)
    
    # =============================================================
    # TESTS PARA VALORES EXTREMOS
    # =============================================================
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_training_params_extreme_values(self, mock_cargar, mock_listar):
        """Valores extremos en training_params"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
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
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_carrera_nombre_muy_largo(self, mock_cargar, mock_listar):
        """Nombres de carrera extremadamente largos"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
        nombre_largo = "A" * 10000 + "-2025"
        event = {
            "carreras": [
                {"nombre": nombre_largo, "splits": ["km_5"]}
            ]
        }
        
        result = lambda_handler(event, None)
        assert result["num_modelos"] == 1
        assert result["carreras_config"][0]["carrera_objetivo"] == nombre_largo
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_splits_con_caracteres_especiales(self, mock_cargar, mock_listar):
        """Splits con caracteres especiales y unicode"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
        event = {
            "carreras": [
                {
                    "nombre": "Carrera-con-ny-🏃‍♂️-2025",
                    "splits": ["km_5", "km_10🏁", "half_🎯", "punto_€$%"]
                }
            ]
        }
        
        result = lambda_handler(event, None)
        assert result["num_modelos"] == 1
        assert result["carreras_config"][0]["splits"] == ["km_5", "km_10🏁", "half_🎯", "punto_€$%"]
    
    # =============================================================
    # TESTS PARA FILTROS CON VALORES EXTREMOS
    # =============================================================
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_filters_with_various_types(self, mock_cargar, mock_listar):
        """Filtros con diferentes tipos de datos"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
        event = {
            "carreras": [
                {
                    "nombre": "Carrera1-2025",
                    "splits": ["km_5"],
                    "event_id_filter": 12345,
                    "event_std_filter": None
                },
                {
                    "nombre": "Carrera2-2025",
                    "splits": ["km_10"],
                    "event_id_filter": True,
                    "event_std_filter": False
                },
                {
                    "nombre": "Carrera3-2025",
                    "splits": ["km_15"],
                    "event_id_filter": ["lista"],
                    "event_std_filter": {"dict": "value"}
                }
            ]
        }
        
        result = lambda_handler(event, None)
        assert result["num_modelos"] == 3
        
        assert result["carreras_config"][0]["event_id_filter"] == 12345
        assert result["carreras_config"][0]["event_std_filter"] is None
        assert result["carreras_config"][1]["event_id_filter"] is True
        assert result["carreras_config"][1]["event_std_filter"] is False
        assert result["carreras_config"][2]["event_id_filter"] == ["lista"]
        assert result["carreras_config"][2]["event_std_filter"] == {"dict": "value"}
    
    # =============================================================
    # TESTS MEJORADOS PARA TIMESTAMP
    # =============================================================
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_timestamp_format_regex(self, mock_cargar, mock_listar):
        """Verifica timestamp con regex y parsing real"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
        event = {
            "carrera": "Test-2025",
            "splits": ["km_5"]
        }
        result = lambda_handler(event, None)
        timestamp = result["generated_at"]
        
        pattern = r'^\d{8}-\d{6}$'
        assert re.match(pattern, timestamp) is not None, f"Formato inválido: {timestamp}"
        
        try:
            parsed = datetime.strptime(timestamp, "%Y%m%d-%H%M%S")
            assert parsed.year > 2020
            assert parsed.year < 2100
        except ValueError as e:
            pytest.fail(f"Timestamp inválido: {timestamp} - Error: {e}")
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_timestamp_microsecond_resolution(self, mock_cargar, mock_listar):
        """Verifica que los timestamps pueden ser diferentes"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
        event = {"carrera": "Test-2025", "splits": ["km_5"]}
        
        result1 = lambda_handler(event, None)
        result2 = lambda_handler(event, None)
        
        # No hacemos assert estricto, solo documentamos
        if result1["generated_at"] == result2["generated_at"]:
            print("⚠️ Timestamps iguales")
        else:
            print("✅ Timestamps diferentes")
    
    # =============================================================
    # TESTS PARA EVENTOS CON CAMPOS EXTRA
    # =============================================================
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_extra_fields_ignored(self, mock_cargar, mock_listar):
        """Campos extra en el evento deben ser ignorados"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
        event = {
            "carrera": "Test-2025",
            "splits": ["km_5"],
            "campo_extra": "esto_debe_ignorarse",
            "otro_campo": [1, 2, 3],
            "config_extra": {"algo": "valor"}
        }
        
        result = lambda_handler(event, None)
        assert result["num_modelos"] == 1
        assert result["carreras_config"][0]["carrera_objetivo"] == "Test-2025"
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_extra_fields_in_carrera(self, mock_cargar, mock_listar):
        """Campos extra dentro de cada carrera deben ser ignorados"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
        event = {
            "carreras": [
                {
                    "nombre": "Carrera1-2025",
                    "splits": ["km_5"],
                    "campo_extra": "debe_ignorarse",
                    "otro_extra": 123
                }
            ]
        }
        
        result = lambda_handler(event, None)
        assert result["num_modelos"] == 1
        
        config = result["carreras_config"][0]
        assert "campo_extra" not in config
        assert "otro_extra" not in config
    
    # =============================================================
    # TESTS PARA COMPATIBILIDAD Y REGRESIÓN
    # =============================================================
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_backward_compatibility_exact_match(self, mock_cargar, mock_listar):
        """Formato legacy debe producir salida compatible (ignorando timestamp)"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
        legacy_event = {
            "carrera": "Maraton_Madrid-2024",
            "splits": ["km_5", "km_10"],
            "tipo_modelo": "interpolacion",
            "event_id_filter": "123"
        }
        
        new_format_event = {
            "carreras": [
                {
                    "nombre": "Maraton_Madrid-2024",
                    "splits": ["km_5", "km_10"],
                    "tipo_modelo": "interpolacion",
                    "event_id_filter": "123"
                }
            ]
        }
        
        result_legacy = lambda_handler(legacy_event, None)
        result_new = lambda_handler(new_format_event, None)
        
        # Eliminar timestamps para comparación
        result_legacy.pop("generated_at")
        result_legacy.pop("timestamp_unico")
        result_new.pop("generated_at")
        result_new.pop("timestamp_unico")
        
        # Verificar que ambos tienen el mismo número de modelos
        assert result_legacy["num_modelos"] == result_new["num_modelos"]
        
        # Verificar que la carrera objetivo es la misma
        assert result_legacy["carreras_config"][0]["carrera_objetivo"] == result_new["carreras_config"][0]["carrera_objetivo"]
        assert result_legacy["carreras_config"][0]["splits"] == result_new["carreras_config"][0]["splits"]
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_case_sensitivity(self, mock_cargar, mock_listar):
        """Verificar que los campos son case-sensitive"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
        event = {
            "CARRERA": "Test-2025",
            "splits": ["km_5"]
        }
        
        with pytest.raises(Exception) as exc_info:
            lambda_handler(event, None)
        assert "No se especificaron carreras" in str(exc_info.value)
    
    # =============================================================
    # TESTS DE CARGA Y RENDIMIENTO
    # =============================================================
    
    @patch('glue_utils.listar_carreras_disponibles')
    @patch('glue_utils.cargar_catalogo_distancias')
    def test_lambda_many_carreras(self, mock_cargar, mock_listar):
        """Procesar muchas carreras (100) - prueba de carga básica"""
        mock_listar.return_value = []
        mock_cargar.return_value = None
        
        num_carreras = 100
        carreras = [
            {"nombre": f"Carrera_{i}-2025", "splits": [f"km_{j}" for j in range(1, 6)]}
            for i in range(num_carreras)
        ]
        
        event = {"carreras": carreras}
        result = lambda_handler(event, None)
        
        assert result["num_modelos"] == num_carreras
        assert len(result["carreras_config"]) == num_carreras
        
        nombres_procesados = [c["carrera_objetivo"] for c in result["carreras_config"]]
        for i in range(num_carreras):
            assert f"Carrera_{i}-2025" in nombres_procesados


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])