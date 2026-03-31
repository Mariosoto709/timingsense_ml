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
    
    def test_lambda_mixed_valid_invalid_carreras(self):
        """Mezcla de carreras válidas e inválidas - debe omitir solo las inválidas"""
        event = {
            "carreras": [
                {"nombre": "Valida1", "splits": ["km_5"]},           # ✅ válida
                {"splits": ["km_10"]},                               # ❌ sin nombre
                {"nombre": "Valida2", "splits": ["km_15"]},          # ✅ válida
                {"nombre": "Invalida", "splits": []},                # ❌ splits vacío
                {"nombre": "Valida3", "splits": ["km_20", "km_25"]}, # ✅ válida
                {"nombre": None, "splits": ["km_30"]},               # ❌ nombre None
                {"nombre": "Valida4", "splits": ["km_35"]}           # ✅ válida
            ]
        }
        result = lambda_handler(event, None)
        
        # Debe procesar solo las 4 válidas
        assert result["num_modelos"] == 4
        
        # Verificar que están en orden
        carreras_procesadas = [c["carrera_objetivo"] for c in result["carreras_config"]]
        assert carreras_procesadas == ["Valida1", "Valida2", "Valida3", "Valida4"]
        
        # Verificar que los splits se mantienen
        assert result["carreras_config"][0]["splits"] == ["km_5"]
        assert result["carreras_config"][1]["splits"] == ["km_15"]
        assert result["carreras_config"][2]["splits"] == ["km_20", "km_25"]
        assert result["carreras_config"][3]["splits"] == ["km_35"]
    
    def test_lambda_empty_splits_list_explicit(self):
        """Splits como lista vacía explícita - debe invalidar la carrera"""
        event = {
            "carreras": [
                {"nombre": "Carrera1", "splits": []}
            ]
        }
        
        with pytest.raises(Exception) as exc_info:
            lambda_handler(event, None)
        assert "No se pudo preparar ninguna carrera" in str(exc_info.value)
    
    def test_lambda_splits_none_value(self):
        """Splits como None - debe fallar con ValueError"""
        event = {
            "carreras": [
                {"nombre": "Carrera1", "splits": None}
            ]
        }

        with pytest.raises(ValueError) as exc_info:
            lambda_handler(event, None)
        
        # ✅ Esperar el mensaje real que lanza la Lambda
        assert "splits debe ser una lista" in str(exc_info.value)
        
    # =============================================================
    # TESTS PARA CAMPOS CON PRIORIDAD Y COEXISTENCIA
    # =============================================================
    
    def test_lambda_nombre_prioridad_sobre_carrera_nuevo_formato(self):
        """En nuevo formato, 'nombre' tiene prioridad sobre 'carrera'"""
        event = {
            "carreras": [
                {
                    "nombre": "NombrePrioritario",
                    "carrera": "NombreIgnorado",  # No debería usarse
                    "splits": ["km_5"]
                }
            ]
        }
        result = lambda_handler(event, None)
        
        assert result["num_modelos"] == 1
        assert result["carreras_config"][0]["carrera_objetivo"] == "NombrePrioritario"
    
    def test_lambda_nombre_none_usar_carrera(self):
        """Si 'nombre' es None, usa 'carrera'"""
        event = {
            "carreras": [
                {
                    "nombre": None,
                    "carrera": "CarreraBackup",
                    "splits": ["km_5"]
                }
            ]
        }
        result = lambda_handler(event, None)
        
        assert result["num_modelos"] == 1
        assert result["carreras_config"][0]["carrera_objetivo"] == "CarreraBackup"
    
    def test_lambda_nombre_vacio_usar_carrera(self):
        """Si 'nombre' es string vacío, usa 'carrera'"""
        event = {
            "carreras": [
                {
                    "nombre": "",
                    "carrera": "CarreraBackup",
                    "splits": ["km_5"]
                }
            ]
        }
        result = lambda_handler(event, None)
        
        assert result["num_modelos"] == 1
        assert result["carreras_config"][0]["carrera_objetivo"] == "CarreraBackup"
    
    # =============================================================
    # TESTS PARA TIPOS DE DATOS INCORRECTOS
    # =============================================================
    
    def test_lambda_splits_wrong_type_string(self):
        """splits como string en lugar de lista - debe fallar"""
        event = {
            "carreras": [
                {
                    "nombre": "Carrera1",
                    "splits": "km_5, km_10"  # ❌ debería ser lista
                }
            ]
        }
        
        # Esto debería explotar porque no se puede iterar sobre un string?
        # Depende de cómo se implemente. Si espera lista, esto fallará.
        with pytest.raises(Exception):
            lambda_handler(event, None)
    
    def test_lambda_splits_wrong_type_int(self):
        """splits como entero - debe fallar"""
        event = {
            "carreras": [
                {
                    "nombre": "Carrera1",
                    "splits": 123  # ❌ debería ser lista
                }
            ]
        }
        
        with pytest.raises(Exception):
            lambda_handler(event, None)
    
    def test_lambda_tipo_modelo_wrong_type(self):
        """tipo_modelo como número - debe fallar"""
        event = {
            "carreras": [
                {
                    "nombre": "Carrera1",
                    "splits": ["km_5"],
                    "tipo_modelo": 123  # ❌ debería ser string
                }
            ]
        }
        
        with pytest.raises(Exception) as exc_info:
            lambda_handler(event, None)
        # El error puede ser de tipo o de validación
        assert "tipo_modelo debe ser" in str(exc_info.value) or "must be str" in str(exc_info.value)
    
    def test_lambda_training_params_wrong_type(self):
        """training_params como string en lugar de dict - debe fallar con ValueError"""
        event = {
            "carreras": [
                {
                    "nombre": "Carrera1",
                    "splits": ["km_5"],
                    "training_params": "no_es_dict"  # ❌ debería ser dict
                }
            ]
        }
        
        # ✅ Cambiamos AttributeError por ValueError
        with pytest.raises(ValueError) as exc_info:
            lambda_handler(event, None)
        
        # ✅ Opcional: verificar que el mensaje de error sea el correcto
        assert "training_params debe ser un diccionario" in str(exc_info.value)
    
    # =============================================================
    # TESTS PARA VALORES EXTREMOS
    # =============================================================
    
    def test_lambda_training_params_extreme_values(self):
        """Valores extremos en training_params"""
        training_params = {
            "n_estimators": 0,           # Límite inferior
            "max_depth": 1000,           # Valor muy alto
            "learning_rate": -0.5,       # Negativo (probablemente inválido pero debe pasarse)
            "min_samples_split": 1,      # Mínimo posible
            "subsample": 1.5,            # > 1.0
            "param_con_puntos": 1.2e-10, # Notación científica
            "param_con_unicode": "🏃‍♂️ valor",  # Unicode
            "param_con_espacios": "valor con espacios",
            "param_vacio": None,
            "param_lista": [1, 2, 3],    # Lista anidada
            "param_dict": {"sub": "value"}  # Dict anidado
        }
        
        event = {
            "carrera": "Test",
            "splits": ["km_5"],
            "training_params": training_params
        }
        
        result = lambda_handler(event, None)
        config = result["carreras_config"][0]
        
        # Debe pasar exactamente los mismos valores
        assert config["training_params"] == training_params
    
    def test_lambda_carrera_nombre_muy_largo(self):
        """Nombres de carrera extremadamente largos"""
        nombre_largo = "A" * 10000  # 10,000 caracteres
        event = {
            "carreras": [
                {"nombre": nombre_largo, "splits": ["km_5"]}
            ]
        }
        
        result = lambda_handler(event, None)
        assert result["num_modelos"] == 1
        assert result["carreras_config"][0]["carrera_objetivo"] == nombre_largo
    
    def test_lambda_splits_con_caracteres_especiales(self):
        """Splits con caracteres especiales y unicode"""
        event = {
            "carreras": [
                {
                    "nombre": "Carrera con ñ y 🏃‍♂️",
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
    
    def test_lambda_filters_with_various_types(self):
        """Filtros con diferentes tipos de datos"""
        event = {
            "carreras": [
                {
                    "nombre": "Carrera1",
                    "splits": ["km_5"],
                    "event_id_filter": 12345,      # Número en lugar de string
                    "event_std_filter": None       # None explícito
                },
                {
                    "nombre": "Carrera2",
                    "splits": ["km_10"],
                    "event_id_filter": True,        # Boolean
                    "event_std_filter": False
                },
                {
                    "nombre": "Carrera3",
                    "splits": ["km_15"],
                    "event_id_filter": ["lista"],   # Lista (inusual pero válido)
                    "event_std_filter": {"dict": "value"}
                }
            ]
        }
        
        result = lambda_handler(event, None)
        assert result["num_modelos"] == 3
        
        # Verificar que los valores se pasan tal cual
        assert result["carreras_config"][0]["event_id_filter"] == 12345
        assert result["carreras_config"][0]["event_std_filter"] is None
        assert result["carreras_config"][1]["event_id_filter"] is True
        assert result["carreras_config"][1]["event_std_filter"] is False
        assert result["carreras_config"][2]["event_id_filter"] == ["lista"]
        assert result["carreras_config"][2]["event_std_filter"] == {"dict": "value"}
    
    # =============================================================
    # TESTS MEJORADOS PARA TIMESTAMP
    # =============================================================
    
    def test_lambda_timestamp_format_regex(self):
        """Verifica timestamp con regex y parsing real"""
        event = {
            "carrera": "Test",
            "splits": ["km_5"]
        }
        result = lambda_handler(event, None)
        timestamp = result["generated_at"]
        
        # Validar con regex
        pattern = r'^\d{8}-\d{6}$'
        assert re.match(pattern, timestamp) is not None, f"Formato inválido: {timestamp}"
        
        # Validar que es una fecha real
        try:
            parsed = datetime.strptime(timestamp, "%Y%m%d-%H%M%S")
            # Verificar que el año es razonable (no 0000)
            assert parsed.year > 2020
            assert parsed.year < 2100
        except ValueError as e:
            pytest.fail(f"Timestamp inválido: {timestamp} - Error: {e}")
    
    def test_lambda_timestamp_microsecond_resolution(self):
        """Verifica que los timestamps pueden ser diferentes en microsegundos"""
        import time
        
        event = {"carrera": "Test", "splits": ["km_5"]}
        
        # Llamar muy rápido para ver si usa microsegundos
        result1 = lambda_handler(event, None)
        result2 = lambda_handler(event, None)
        
        # Podrían ser iguales si la resolución es de 1 segundo
        # Pero deberían ser diferentes si usa microsegundos
        # Este test documenta el comportamiento actual
        if result1["generated_at"] == result2["generated_at"]:
            print("⚠️ Timestamps iguales - la lambda no usa microsegundos")
        else:
            print("✅ Timestamps diferentes - buena resolución")
        
        # No hacemos assert porque podría pasar cualquiera de los dos
    
    # =============================================================
    # TESTS PARA EVENTOS CON CAMPOS EXTRA
    # =============================================================
    
    def test_lambda_extra_fields_ignored(self):
        """Campos extra en el evento deben ser ignorados"""
        event = {
            "carrera": "Test",
            "splits": ["km_5"],
            "campo_extra": "esto_debe_ignorarse",
            "otro_campo": [1, 2, 3],
            "config_extra": {"algo": "valor"}
        }
        
        result = lambda_handler(event, None)
        assert result["num_modelos"] == 1
        assert result["carreras_config"][0]["carrera_objetivo"] == "Test"
    
    def test_lambda_extra_fields_in_carrera(self):
        """Campos extra dentro de cada carrera deben ser ignorados"""
        event = {
            "carreras": [
                {
                    "nombre": "Carrera1",
                    "splits": ["km_5"],
                    "campo_extra": "debe_ignorarse",
                    "otro_extra": 123
                }
            ]
        }
        
        result = lambda_handler(event, None)
        assert result["num_modelos"] == 1
        
        # Los campos extra NO deben estar en la salida
        config = result["carreras_config"][0]
        assert "campo_extra" not in config
        assert "otro_extra" not in config
        
        # Solo deben estar los campos esperados
        expected_fields = {
            "carrera_objetivo", "splits", "event_id_filter", 
            "event_std_filter", "tipo_modelo", "training_params"
        }
        assert set(config.keys()) == expected_fields
    
    # =============================================================
    # TESTS PARA COMPATIBILIDAD Y REGRESIÓN
    # =============================================================
    
    def test_lambda_backward_compatibility_exact_match(self):
        """Formato legacy debe producir EXACTAMENTE la misma salida que antes"""
        legacy_event = {
            "carrera": "Maratón_Madrid_2024",
            "splits": ["km_5", "km_10"],
            "tipo_modelo": "interpolacion",
            "event_id_filter": "123"
        }
        
        new_format_event = {
            "carreras": [
                {
                    "nombre": "Maratón_Madrid_2024",
                    "splits": ["km_5", "km_10"],
                    "tipo_modelo": "interpolacion",
                    "event_id_filter": "123"
                }
            ]
        }
        
        result_legacy = lambda_handler(legacy_event, None)
        result_new = lambda_handler(new_format_event, None)
        
        # Comparar todo excepto timestamp que es diferente
        result_legacy.pop("generated_at")
        result_legacy.pop("timestamp_unico")
        result_new.pop("generated_at")
        result_new.pop("timestamp_unico")
        
        assert result_legacy == result_new
    
    def test_lambda_case_sensitivity(self):
        """Verificar que los campos son case-sensitive (como debe ser)"""
        event = {
            "CARRERA": "Test",  # Mayúsculas
            "splits": ["km_5"]
        }
        
        # No debería encontrar 'carrera' en minúsculas
        with pytest.raises(Exception) as exc_info:
            lambda_handler(event, None)
        assert "No se especificaron carreras" in str(exc_info.value)
    
    # =============================================================
    # TESTS DE CARGA Y RENDIMIENTO (concepto)
    # =============================================================
    
    def test_lambda_many_carreras(self):
        """Procesar muchas carreras (100) - prueba de carga básica"""
        num_carreras = 100
        carreras = [
            {"nombre": f"Carrera_{i}", "splits": [f"km_{j}" for j in range(1, 6)]}
            for i in range(num_carreras)
        ]
        
        event = {"carreras": carreras}
        result = lambda_handler(event, None)
        
        assert result["num_modelos"] == num_carreras
        assert len(result["carreras_config"]) == num_carreras
        
        # Verificar que todas están presentes
        nombres_procesados = [c["carrera_objetivo"] for c in result["carreras_config"]]
        for i in range(num_carreras):
            assert f"Carrera_{i}" in nombres_procesados

if __name__ == "__main__":
    # Ejecutar con verbose para ver los resultados
    pytest.main([__file__, "-v", "--tb=short"])