"""
tests/unit/test_lambda_bdd_races.py
Pruebas unitarias para la Lambda lambda_bdd_races
"""

import sys
import os
import json
import pytest

# Añadir la carpeta de la lambda al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../lambda/lambda_bdd_races')))

from lambda_bdd_races import lambda_handler

class TestLambdaBDDRaces:
    """Pruebas para la Lambda de configuración de entrenamiento"""
    
    # =============================================================
    # TESTS PARA FORMATO LEGACY (UN SOLO MODELO)
    # =============================================================
    
    def test_lambda_single_carrera_legacy_interpolacion(self):
        """Evento con formato antiguo - modelo interpolación"""
        event = {
            "carrera": "Maratón_Madrid_2024",
            "splits": ["km_5", "km_10", "half"],
            "tipo_modelo": "interpolacion"
        }
        result = lambda_handler(event, None)
        
        # Verificar estructura básica
        assert result["num_modelos"] == 1
        assert "generated_at" in result
        assert "timestamp_unico" in result
        assert result["timestamp_unico"] == result["generated_at"]
        
        # Verificar configuración de la carrera
        config = result["carreras_config"][0]
        assert config["carrera_objetivo"] == "Maratón_Madrid_2024"
        assert config["splits"] == ["km_5", "km_10", "half"]
        assert config["tipo_modelo"] == "interpolacion"
        assert config["event_id_filter"] is None
        assert config["event_std_filter"] is None
        assert config["training_params"] == {}
    
    def test_lambda_single_carrera_legacy_prediccion(self):
        """Evento con formato antiguo - modelo predicción"""
        event = {
            "carrera": "Maratón_Valencia_2024",
            "splits": ["km_5", "km_10", "km_15"],
            "tipo_modelo": "prediccion",
            "training_params": {
                "n_estimators": 200,
                "max_depth": 8
            }
        }
        result = lambda_handler(event, None)
        
        config = result["carreras_config"][0]
        assert config["carrera_objetivo"] == "Maratón_Valencia_2024"
        assert config["tipo_modelo"] == "prediccion"
        assert config["training_params"]["n_estimators"] == 200
        assert config["training_params"]["max_depth"] == 8
    
    def test_lambda_single_carrera_legacy_with_filters(self):
        """Evento con formato antiguo - con filtros"""
        event = {
            "carrera": "Maratón_Madrid_2024",
            "splits": ["km_5", "km_10", "half"],
            "event_id_filter": "12345",
            "event_std_filter": "ABC123",
            "tipo_modelo": "interpolacion"
        }
        result = lambda_handler(event, None)
        
        config = result["carreras_config"][0]
        assert config["event_id_filter"] == "12345"
        assert config["event_std_filter"] == "ABC123"
    
    def test_lambda_single_carrera_legacy_default_tipo_modelo(self):
        """Evento sin tipo_modelo → debe usar 'interpolacion' por defecto"""
        event = {
            "carrera": "Maratón_Madrid_2024",
            "splits": ["km_5", "km_10", "half"]
        }
        result = lambda_handler(event, None)
        
        config = result["carreras_config"][0]
        assert config["tipo_modelo"] == "interpolacion"
    
    # =============================================================
    # TESTS PARA FORMATO NUEVO (LISTA DE CARRERAS)
    # =============================================================
    
    def test_lambda_multiple_carreras(self):
        """Evento con lista de carreras"""
        event = {
            "carreras": [
                {
                    "nombre": "Carrera1", 
                    "splits": ["km_5"], 
                    "tipo_modelo": "interpolacion"
                },
                {
                    "nombre": "Carrera2", 
                    "splits": ["km_10"], 
                    "tipo_modelo": "prediccion",
                    "training_params": {"n_estimators": 150}
                }
            ]
        }
        result = lambda_handler(event, None)
        
        assert result["num_modelos"] == 2
        
        config1 = result["carreras_config"][0]
        assert config1["carrera_objetivo"] == "Carrera1"
        assert config1["tipo_modelo"] == "interpolacion"
        
        config2 = result["carreras_config"][1]
        assert config2["carrera_objetivo"] == "Carrera2"
        assert config2["tipo_modelo"] == "prediccion"
        assert config2["training_params"]["n_estimators"] == 150
    
    def test_lambda_multiple_carreras_con_filtros(self):
        """Evento con múltiples carreras y filtros"""
        event = {
            "carreras": [
                {
                    "nombre": "Carrera1",
                    "splits": ["km_5"],
                    "event_id_filter": "evt123",
                    "event_std_filter": "std456"
                },
                {
                    "nombre": "Carrera2",
                    "splits": ["km_10"]
                }
            ]
        }
        result = lambda_handler(event, None)
        
        assert result["num_modelos"] == 2
        
        config1 = result["carreras_config"][0]
        assert config1["event_id_filter"] == "evt123"
        assert config1["event_std_filter"] == "std456"
        
        config2 = result["carreras_config"][1]
        assert config2["event_id_filter"] is None
        assert config2["event_std_filter"] is None
    
    # =============================================================
    # TESTS PARA CASOS DE ERROR Y BORDE
    # =============================================================
    
    def test_lambda_empty_carreras(self):
        """Evento sin carreras (formato vacío)"""
        event = {}
        
        with pytest.raises(Exception) as exc_info:
            lambda_handler(event, None)
        assert "No se especificaron carreras" in str(exc_info.value)
    
    def test_lambda_empty_carreras_list(self):
        """Evento con lista de carreras vacía"""
        event = {"carreras": []}
        
        with pytest.raises(Exception) as exc_info:
            lambda_handler(event, None)
        assert "No se especificaron carreras" in str(exc_info.value)
    
    def test_lambda_missing_carrera_nombre(self):
        """Falta nombre de carrera en uno de los elementos"""
        event = {
            "carreras": [
                {"nombre": "Carrera1", "splits": ["km_5"]},
                {"splits": ["km_10"]}  # falta nombre
            ]
        }
        result = lambda_handler(event, None)
        
        # Solo debe procesar la que tiene nombre
        assert result["num_modelos"] == 1
        assert result["carreras_config"][0]["carrera_objetivo"] == "Carrera1"
    
    def test_lambda_missing_splits(self):
        """Falta lista de splits en una carrera"""
        event = {
            "carreras": [
                {"nombre": "Carrera1", "splits": ["km_5"]},
                {"nombre": "Carrera2"}  # falta splits
            ]
        }
        result = lambda_handler(event, None)
        
        assert result["num_modelos"] == 1
        assert result["carreras_config"][0]["carrera_objetivo"] == "Carrera1"
    
    def test_lambda_invalid_tipo_modelo(self):
        """tipo_modelo inválido"""
        event = {
            "carreras": [
                {"nombre": "Carrera1", "splits": ["km_5"], "tipo_modelo": "invalido"}
            ]
        }
        
        with pytest.raises(Exception) as exc_info:
            lambda_handler(event, None)
        assert "tipo_modelo debe ser" in str(exc_info.value)
    
    def test_lambda_invalid_tipo_modelo_legacy(self):
        """tipo_modelo inválido en formato legacy"""
        event = {
            "carrera": "Maratón_Madrid_2024",
            "splits": ["km_5"],
            "tipo_modelo": "invalido"
        }
        
        with pytest.raises(Exception) as exc_info:
            lambda_handler(event, None)
        assert "tipo_modelo debe ser" in str(exc_info.value)
    
    def test_lambda_all_carreras_invalid(self):
        """Todas las carreras son inválidas"""
        event = {
            "carreras": [
                {"splits": ["km_5"]},  # falta nombre
                {"nombre": "Carrera2"}  # falta splits
            ]
        }
        
        with pytest.raises(Exception) as exc_info:
            lambda_handler(event, None)
        assert "No se pudo preparar ninguna carrera" in str(exc_info.value)
    
    # =============================================================
    # TESTS PARA TIMESTAMP Y FORMATO DE SALIDA
    # =============================================================
    
    def test_lambda_timestamp_format(self):
        """Verifica que el timestamp tiene el formato correcto"""
        event = {
            "carrera": "Test",
            "splits": ["km_5"]
        }
        result = lambda_handler(event, None)
        
        timestamp = result["generated_at"]
        # Formato esperado: YYYYMMDD-HHMMSS (ej: 20260323-143022)
        assert len(timestamp) == 15  # 8 + 1 + 6
        assert timestamp[8] == "-"  # guión entre fecha y hora
        # Los primeros 8 caracteres deben ser números (fecha)
        assert timestamp[:8].isdigit()
        # Los últimos 6 caracteres deben ser números (hora)
        assert timestamp[9:].isdigit()
    
    def test_lambda_timestamp_unique(self):
        """Dos llamadas generan timestamps diferentes"""
        import time
        
        event = {"carrera": "Test", "splits": ["km_5"]}
        result1 = lambda_handler(event, None)
        time.sleep(1)  # esperar 1 segundo
        result2 = lambda_handler(event, None)
        
        assert result1["generated_at"] != result2["generated_at"]
    
    def test_lambda_output_structure(self):
        """Verifica la estructura completa de la salida"""
        event = {
            "carreras": [
                {"nombre": "Carrera1", "splits": ["km_5"], "tipo_modelo": "interpolacion"},
                {"nombre": "Carrera2", "splits": ["km_10"], "tipo_modelo": "prediccion"}
            ]
        }
        result = lambda_handler(event, None)
        
        # Verificar estructura
        assert "carreras_config" in result
        assert "num_modelos" in result
        assert "generated_at" in result
        assert "timestamp_unico" in result
        
        # Verificar tipo de datos
        assert isinstance(result["carreras_config"], list)
        assert isinstance(result["num_modelos"], int)
        assert isinstance(result["generated_at"], str)
        
        # Cada configuración debe tener los campos requeridos
        for config in result["carreras_config"]:
            assert "carrera_objetivo" in config
            assert "splits" in config
            assert "event_id_filter" in config
            assert "event_std_filter" in config
            assert "tipo_modelo" in config
            assert "training_params" in config
    
    # =============================================================
    # TESTS PARA TRAINING_PARAMS
    # =============================================================
    
    def test_lambda_training_params_passthrough(self):
        """Los training_params se pasan correctamente"""
        training_params = {
            "n_estimators": 300,
            "max_depth": 12,
            "learning_rate": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "n_folds": 3,
            "min_samples": 100
        }
        event = {
            "carrera": "Test",
            "splits": ["km_5"],
            "training_params": training_params
        }
        result = lambda_handler(event, None)
        
        config = result["carreras_config"][0]
        assert config["training_params"] == training_params
    
    def test_lambda_training_params_default(self):
        """Si no hay training_params, se usa diccionario vacío"""
        event = {
            "carrera": "Test",
            "splits": ["km_5"]
        }
        result = lambda_handler(event, None)
        
        config = result["carreras_config"][0]
        assert config["training_params"] == {}
    
    # =============================================================
    # TESTS PARA MEZCLA DE FORMATOS
    # =============================================================
    
    def test_lambda_legacy_and_new_format_conflict(self):
        """Si hay ambos formatos, nuevo tiene prioridad"""
        event = {
            "carrera": "LegacyCarrera",  # formato antiguo
            "splits": ["km_5"],
            "carreras": [  # formato nuevo
                {"nombre": "NewCarrera", "splits": ["km_10"]}
            ]
        }
        result = lambda_handler(event, None)
        
        # El nuevo formato debe tener prioridad
        assert result["num_modelos"] == 1
        assert result["carreras_config"][0]["carrera_objetivo"] == "NewCarrera"
        assert "LegacyCarrera" not in [c["carrera_objetivo"] for c in result["carreras_config"]]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])