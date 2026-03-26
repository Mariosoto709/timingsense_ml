"""
tests/unit/test_lambda_procesar_glue.py
Pruebas unitarias para la Lambda lambda_procesar_salida_glue
"""

import sys
import os
import json
import pytest
from unittest.mock import Mock, patch, MagicMock

# Añadir la carpeta de la lambda al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../lambda/lambda_procesar_salida_glue')))

from lambda_procesar_salida_glue import lambda_handler


class TestLambdaProcesarGlue:
    """Pruebas para la Lambda que procesa la salida de Glue"""
    
    # =============================================================
    # DATOS DE PRUEBA
    # =============================================================
    
    @pytest.fixture
    def mock_s3_client(self):
        """Crea un mock del cliente S3"""
        with patch('lambda_procesar_salida_glue.s3') as mock_s3:
            yield mock_s3
    
    @pytest.fixture
    def sample_salida_glue(self):
        """Datos de ejemplo de salida del job Glue"""
        return {
            "modelos": [
                {
                    "carrera": "Maratón_Madrid_2024",
                    "carpeta_modelo": "Maratón_Madrid_2024-20260323-143022",
                    "data_s3_path": "s3://timingsense-athena-output-2026/modelos/Maratón_Madrid_2024-20260323-143022/data/",
                    "tabla_generada": "modelo_Maratón_Madrid_2024_20260323_143022",
                    "splits": 12,
                    "carreras_usadas": 5
                }
            ]
        }
    
    # =============================================================
    # TESTS CON ARCHIVO EXISTENTE
    # =============================================================
    
    def test_lambda_with_valid_file(self, mock_s3_client, sample_salida_glue):
        """Caso normal: encuentra el archivo de salida correctamente"""
        # Configurar mock
        mock_s3_client.get_object.return_value = {
            'Body': MagicMock(read=lambda: json.dumps(sample_salida_glue).encode())
        }
        
        event = {
            "glue_job_run_id": "jr_123456",
            "glue_job_name": "create-training-table",
            "carrera_objetivo": "Maratón_Madrid_2024",
            "generated_at": "20260323-143022"
        }
        
        result = lambda_handler(event, None)
        
        # Verificar que se llamó a get_object con la clave correcta
        expected_key = "debug/salida_step_20260323_143022.json"
        mock_s3_client.get_object.assert_called_with(
            Bucket="timingsense-athena-output-2026",
            Key=expected_key
        )
        
        # Verificar estructura de salida
        assert "modelos" in result
        assert len(result["modelos"]) == 1
        
        # Verificar que se corrigió la carpeta correctamente
        modelo = result["modelos"][0]
        assert modelo["carrera"] == "Maratón_Madrid_2024"
        assert modelo["carpeta_modelo"] == "Maratón_Madrid_2024-20260323-143022"
        assert modelo["data_s3_path"] == "s3://timingsense-athena-output-2026/modelos/Maratón_Madrid_2024-20260323-143022/data/"
        assert modelo["tabla_generada"] == "modelo_Maratón_Madrid_2024_20260323_143022"
    
    def test_lambda_corrects_folder_path(self, mock_s3_client, sample_salida_glue):
        """Verifica que corrige correctamente la carpeta del modelo"""
        # Simular que el archivo tiene una carpeta incorrecta
        salida_incorrecta = {
            "modelos": [
                {
                    "carrera": "Maratón_Madrid_2024",
                    "carpeta_modelo": "carpeta_incorrecta",
                    "data_s3_path": "s3://.../carpeta_incorrecta/data/",
                    "tabla_generada": "tabla_incorrecta"
                }
            ]
        }
        mock_s3_client.get_object.return_value = {
            'Body': MagicMock(read=lambda: json.dumps(salida_incorrecta).encode())
        }
        
        event = {
            "glue_job_run_id": "jr_123456",
            "glue_job_name": "create-training-table",
            "carrera_objetivo": "Maratón_Madrid_2024",
            "generated_at": "20260323-143022"
        }
        
        result = lambda_handler(event, None)
        
        modelo = result["modelos"][0]
        # Debe haber corregido la carpeta
        assert modelo["carpeta_modelo"] == "Maratón_Madrid_2024-20260323-143022"
        assert "carpeta_incorrecta" not in modelo["data_s3_path"]
    
    def test_lambda_with_multiple_models(self, mock_s3_client):
        """Caso con múltiples modelos en la respuesta"""
        salida_multiple = {
            "modelos": [
                {"carrera": "Carrera1", "carpeta_modelo": "old1", "data_s3_path": "old_path1"},
                {"carrera": "Carrera2", "carpeta_modelo": "old2", "data_s3_path": "old_path2"}
            ]
        }
        mock_s3_client.get_object.return_value = {
            'Body': MagicMock(read=lambda: json.dumps(salida_multiple).encode())
        }

        event = {
            "glue_job_run_id": "jr_123456",
            "glue_job_name": "create-training-table",
            "carrera_objetivo": "Carrera1",
            "generated_at": "20260323-143022"
        }

        result = lambda_handler(event, None)

        assert len(result["modelos"]) == 2
        
        # Verificar primer modelo
        modelo1 = result["modelos"][0]
        assert modelo1["carrera"] == "Carrera1"
        assert modelo1["carpeta_modelo"] == "Carrera1-20260323-143022"
        assert modelo1["data_s3_path"] == f"s3://timingsense-athena-output-2026/modelos/Carrera1-20260323-143022/data/"
        assert modelo1["tabla_generada"] == f"modelo_Carrera1_20260323_143022"
        
        # Verificar segundo modelo
        modelo2 = result["modelos"][1]
        assert modelo2["carrera"] == "Carrera2"
        assert modelo2["carpeta_modelo"] == "Carrera2-20260323-143022"
        assert modelo2["data_s3_path"] == f"s3://timingsense-athena-output-2026/modelos/Carrera2-20260323-143022/data/"
        assert modelo2["tabla_generada"] == f"modelo_Carrera2_20260323_143022"
    
    # =============================================================
    # TESTS CON ARCHIVO NO ENCONTRADO - FALLBACK
    # =============================================================
    
    def test_lambda_without_generated_at(self, mock_s3_client):
        """Falta generated_at → respuesta por defecto"""
        event = {
            "glue_job_run_id": "jr_123456",
            "glue_job_name": "create-training-table",
            "carrera_objetivo": "Maratón_Madrid_2024"
            # falta generated_at
        }
        
        result = lambda_handler(event, None)
        
        # Debe devolver lista vacía
        assert result["modelos"] == []
        # No debe llamar a S3
        mock_s3_client.get_object.assert_not_called()
    
    def test_lambda_file_not_found_fallback(self, mock_s3_client):
        """Archivo no encontrado → usa fallback con archivo más reciente"""
        # Simular que el archivo específico no existe
        mock_s3_client.get_object.side_effect = Exception("NoSuchKey")
        
        # Simular list_objects_v2 para encontrar archivos
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'debug/salida_step_20260323_143022.json', 'LastModified': '2026-03-23T14:30:22'},
                {'Key': 'debug/salida_step_20260322_100000.json', 'LastModified': '2026-03-22T10:00:00'}
            ]
        }
        
        # Simular que el archivo más reciente existe
        mock_s3_client.get_object.side_effect = [
            Exception("NoSuchKey"),  # primera llamada falla
            {  # segunda llamada (archivo más reciente) funciona
                'Body': MagicMock(read=lambda: json.dumps({
                    "modelos": [{"carrera": "OldCarrera"}]
                }).encode())
            }
        ]
        
        event = {
            "glue_job_run_id": "jr_123456",
            "glue_job_name": "create-training-table",
            "carrera_objetivo": "Maratón_Madrid_2024",
            "generated_at": "20260323-143022"
        }
        
        result = lambda_handler(event, None)
        
        # Debe haber llamado a list_objects_v2
        mock_s3_client.list_objects_v2.assert_called()
        
        # Debe haber corregido las carpetas con el generated_at correcto
        assert len(result["modelos"]) == 1
        modelo = result["modelos"][0]
        assert modelo["carpeta_modelo"] == "Maratón_Madrid_2024-20260323-143022"
    
    def test_lambda_no_files_found_fallback(self, mock_s3_client):
        """No hay archivos en S3 → fallback final"""
        # Simular que el archivo específico no existe
        mock_s3_client.get_object.side_effect = Exception("NoSuchKey")
        
        # Simular list_objects_v2 sin archivos
        mock_s3_client.list_objects_v2.return_value = {}
        
        event = {
            "glue_job_run_id": "jr_123456",
            "glue_job_name": "create-training-table",
            "carrera_objetivo": "Maratón_Madrid_2024",
            "generated_at": "20260323-143022"
        }
        
        result = lambda_handler(event, None)
        
        # Debe devolver el fallback final con datos básicos
        assert len(result["modelos"]) == 1
        modelo = result["modelos"][0]
        assert modelo["carrera"] == "Maratón_Madrid_2024"
        assert modelo["carpeta_modelo"] == "Maratón_Madrid_2024-20260323-143022"
        assert modelo["splits"] == 12
        assert modelo["carreras_usadas"] == 1
    
    # =============================================================
    # TESTS PARA MANEJO DE ERRORES
    # =============================================================
    
    def test_lambda_invalid_json(self, mock_s3_client):
        """Archivo con JSON inválido"""
        mock_s3_client.get_object.return_value = {
            'Body': MagicMock(read=lambda: b'invalid json {')
        }
        
        event = {
            "glue_job_run_id": "jr_123456",
            "glue_job_name": "create-training-table",
            "carrera_objetivo": "Maratón_Madrid_2024",
            "generated_at": "20260323-143022"
        }
        
        result = lambda_handler(event, None)
        
        # Debe hacer fallback
        assert len(result["modelos"]) == 1
        assert result["modelos"][0]["carrera"] == "Maratón_Madrid_2024"
    
    def test_lambda_empty_models_list(self, mock_s3_client):
        """Archivo con lista de modelos vacía"""
        mock_s3_client.get_object.return_value = {
            'Body': MagicMock(read=lambda: json.dumps({"modelos": []}).encode())
        }
        
        event = {
            "glue_job_run_id": "jr_123456",
            "glue_job_name": "create-training-table",
            "carrera_objetivo": "Maratón_Madrid_2024",
            "generated_at": "20260323-143022"
        }
        
        result = lambda_handler(event, None)
        
        assert result["modelos"] == []
    
    def test_lambda_missing_models_key(self, mock_s3_client):
        """Archivo sin clave 'modelos'"""
        mock_s3_client.get_object.return_value = {
            'Body': MagicMock(read=lambda: json.dumps({"otros": []}).encode())
        }
        
        event = {
            "glue_job_run_id": "jr_123456",
            "glue_job_name": "create-training-table",
            "carrera_objetivo": "Maratón_Madrid_2024",
            "generated_at": "20260323-143022"
        }
        
        result = lambda_handler(event, None)
        
        # Debe devolver lista vacía
        assert result["modelos"] == []
    
    # =============================================================
    # TESTS PARA LOGGING (opcional, verifica que no rompe)
    # =============================================================
    
    def test_lambda_logs_events(self, mock_s3_client, capsys):
        """Verifica que la Lambda imprime logs (no rompe)"""
        mock_s3_client.get_object.return_value = {
            'Body': MagicMock(read=lambda: json.dumps({"modelos": []}).encode())
        }
        
        event = {
            "glue_job_run_id": "jr_123456",
            "glue_job_name": "create-training-table",
            "carrera_objetivo": "Maratón_Madrid_2024",
            "generated_at": "20260323-143022"
        }
        
        result = lambda_handler(event, None)
        
        # Capturar logs
        captured = capsys.readouterr()
        assert "🔍 Procesando salida de Glue" in captured.out
        assert "Evento recibido:" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])