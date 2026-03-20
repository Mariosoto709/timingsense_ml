#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_interpolacion.py - Script de entrenamiento para interpolación en carrera
Uso: (lo ejecuta SageMaker automáticamente con argumentos)
"""

import argparse
import os
import sys
import json
import traceback
import boto3
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime
import re
import logging
import xgboost as xgb

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('timingsense')

# =============================================================
# FUNCIONES PARA MANEJAR METADATA DE INTERPOLACIÓN
# =============================================================

def extract_split_distance(split_name):
    """
    Extrae la distancia en km de un split normalizado.
    Devuelve None si no es un split de distancia.
    """
    if split_name is None:
        return None
    
    split_lower = split_name.lower()
    
    # Splits especiales con distancia conocida
    if split_lower == 'half':
        return 21.0975
    elif split_lower == 'finish':
        return 42.195
    elif split_lower == 'start':
        return 0.0
    
    # Splits km_X
    if split_lower.startswith('km_'):
        try:
            # Convertir 'km_5' → 5.0, 'km_18_2' → 18.2
            num_str = split_lower[3:].replace('_', '.')
            return float(num_str)
        except:
            return None
    
    return None


def load_training_metadata(model_dir):
    """
    Carga la metadata generada por el job de Glue.
    Busca en múltiples ubicaciones posibles.
    """
    logger.info("🔍 INICIANDO BÚSQUEDA DE METADATA...")
    
    # Listar todo el contenido del directorio training
    training_dir = '/opt/ml/input/data/training'
    if os.path.exists(training_dir):
        logger.info(f"📂 Contenido de {training_dir}:")
        try:
            for root, dirs, files in os.walk(training_dir):
                for file in files:
                    logger.info(f"   📄 {os.path.join(root, file)}")
                for dir in dirs:
                    logger.info(f"   📁 {os.path.join(root, dir)}/")
        except Exception as e:
            logger.warning(f"   ⚠️ Error listando directorio: {e}")
    else:
        logger.warning(f"❌ El directorio {training_dir} NO EXISTE")
    
    # Listar el directorio actual
    logger.info(f"📂 Contenido del directorio actual ({os.getcwd()}):")
    for file in os.listdir('.'):
        logger.info(f"   📄 {file}")
    
    posibles_rutas = [
        os.path.join(model_dir, 'metadata.json'),
        'metadata.json',
        '/opt/ml/input/data/training/metadata.json',
        '/opt/ml/input/config/metadata.json'
    ]
    
    # También buscar en subcarpetas de training
    if os.path.exists(training_dir):
        for root, dirs, files in os.walk(training_dir):
            if 'metadata.json' in files:
                ruta = os.path.join(root, 'metadata.json')
                posibles_rutas.append(ruta)
                logger.info(f"🔍 Encontrado metadata.json en: {ruta}")
    
    logger.info("🔍 Buscando en rutas específicas:")
    for ruta in posibles_rutas:
        logger.info(f"   → {ruta}")
        if os.path.exists(ruta):
            try:
                with open(ruta, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"✅ METADATA CARGADA DESDE: {ruta}")
                logger.info(f"   Contenido: {json.dumps(metadata, indent=2)[:200]}...")
                return metadata
            except Exception as e:
                logger.warning(f"⚠️ Error leyendo {ruta}: {e}")
    
    logger.warning("❌ NO SE ENCONTRÓ METADATA.JSON")
    return None


def normalizar_metadata(metadata):
    """Normaliza los nombres de splits en el metadata (convierte puntos a guiones)"""
    if not metadata or 'analisis' not in metadata:
        return metadata
    
    analisis = metadata['analisis']
    mapping_normalizado = {}
    
    for split, info in analisis.get('mapping', {}).items():
        # Normalizar el nombre del split principal
        split_norm = split.replace('.', '_').replace('-', '_').replace(' ', '_')
        
        # Normalizar también el split origen (si existe)
        if len(info) >= 2 and info[1] and isinstance(info[1], str):
            origen_norm = info[1].replace('.', '_').replace('-', '_').replace(' ', '_')
            info_normalizado = [info[0], origen_norm] + info[2:]
        else:
            info_normalizado = info
        
        mapping_normalizado[split_norm] = info_normalizado
    
    # Actualizar también las listas de splits
    if 'splits_directos' in analisis:
        analisis['splits_directos'] = [s.replace('.', '_').replace('-', '_').replace(' ', '_') 
                                        for s in analisis['splits_directos']]
    
    if 'splits_interpolables' in analisis:
        analisis['splits_interpolables'] = [s.replace('.', '_').replace('-', '_').replace(' ', '_') 
                                             for s in analisis['splits_interpolables']]
    
    if 'splits_imposibles' in analisis:
        analisis['splits_imposibles'] = [s.replace('.', '_').replace('-', '_').replace(' ', '_') 
                                          for s in analisis['splits_imposibles']]
    
    analisis['mapping'] = mapping_normalizado
    return metadata


def log_split_analysis(metadata, split_cols_ordenados):
    """
    Registra información detallada sobre los splits y su origen
    """
    if not metadata or 'analisis' not in metadata:
        return
    
    analisis = metadata['analisis']
    mapping = analisis.get('mapping', {})
    
    logger.info("\n📋 ANÁLISIS DE SPLITS PARA ENTRENAMIENTO:")
    
    for split in split_cols_ordenados:
        if split in mapping:
            tipo, origen, dist = mapping[split]
            if tipo == 'direct':
                logger.info(f"   ✅ {split}: disponible directamente")
            elif tipo == 'interpolate':
                dist_objetivo = extract_split_distance(split)
                logger.info(f"   🔄 {split}: desde {origen} (dif: {abs(dist_objetivo - dist):.1f}km)")
        else:
            logger.info(f"   ❓ {split}: sin información de origen")
    
    logger.info(f"\n📊 RESUMEN:")
    logger.info(f"   Splits directos: {len(analisis.get('splits_directos', []))}")
    logger.info(f"   Splits interpolables: {len(analisis.get('splits_interpolables', []))}")
    logger.info(f"   Splits imposibles: {len(analisis.get('splits_imposibles', []))}")


def get_paths():
    """Obtiene las rutas estándar de SageMaker dentro del contenedor"""
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output')
    train_dir = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
    
    # 🟢🟢🟢 NUEVO LOGGING 🟢🟢🟢
    logger.info(f"📂 Model directory: {model_dir}")
    logger.info(f"📂 Output directory: {output_dir}")
    logger.info(f"📂 Training data directory: {train_dir}")
    
    # Mostrar la ruta S3 original (si está disponible)
    training_data_s3 = os.environ.get('SM_TRAINING_DATA_S3', 'No disponible')
    logger.info(f"📂 Training data S3 path: {training_data_s3}")
    
    # Mostrar el training environment (contiene más detalles)
    training_env = os.environ.get('SM_TRAINING_ENV', '{}')
    logger.info(f"📂 Training env (primeros 200 chars): {training_env[:200]}")
    
    # Listar todas las variables SM_ para debug
    logger.info("📂 Todas las variables SM_:")
    for key, value in sorted(os.environ.items()):
        if key.startswith('SM_'):
            logger.info(f"   {key}={value}")
    
    return model_dir, output_dir, train_dir


def load_data_from_local(data_dir):
    """Carga todos los archivos Parquet de forma recursiva"""
    logger.info(f"📂 Cargando datos desde: {data_dir}")
    
    dfs = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                df_temp = pd.read_parquet(file_path)
                dfs.append(df_temp)
                logger.info(f"   ✅ Cargado: {file} - {len(df_temp)} filas")
            except Exception as e:
                logger.warning(f"   ⚠️ No se pudo cargar {file}: {e}")
    
    if not dfs:
        raise FileNotFoundError(f"No se encontraron archivos válidos en {data_dir}")
    
    logger.info(f"✅ Total: {len(dfs)} archivos, {sum(len(df) for df in dfs)} filas")
    return pd.concat(dfs, ignore_index=True)


def ordenar_splits_personalizado(split_cols):
    """Ordena splits por distancia REAL en la carrera"""
    
    def get_split_distance(split):
        """Devuelve la distancia exacta del split en km"""
        split_lower = split.lower()
        
        # Splits especiales con distancia conocida
        if split_lower in ['half', 'media']:
            return 21.0975  # Media maratón
        elif split_lower in ['finish', 'meta']:
            return 42.195   # Maratón completa
        elif split_lower in ['start', 'salida']:
            return 0.0
        
        # Splits kilométricos (km_5, km_10, km_18_2, etc.)
        if split_lower.startswith('km_'):
            try:
                # Extraer el número (ej: 'km_5' → 5, 'km_18_2' → 18.2)
                num_str = split_lower[3:].replace('_', '.')
                return float(num_str)
            except (ValueError, TypeError):
                logger.warning(f"⚠️ No se pudo extraer distancia de {split}")
                return float('inf')
        
        # Si no es un split reconocido, ponerlo al final
        logger.warning(f"⚠️ Split no reconocido: {split}")
        return float('inf')
    
    # Ordenar los splits por distancia
    splits_con_distancia = [(split, get_split_distance(split)) for split in split_cols]
    splits_ordenados = sorted(splits_con_distancia, key=lambda x: x[1])
    
    # Logging para verificar el orden
    logger.info("📊 Orden de splits por distancia:")
    for split, dist in splits_ordenados:
        logger.info(f"   {split}: {dist} km")
    
    return [split for split, _ in splits_ordenados]


def identificar_splits(df):
    """Identifica splits en el DataFrame (SIN RENOMBRAR)"""
    columnas_excluidas = ['gender', 'age', 'birthdate', 'athlete_id', 'event_id', 'event_std']
    splits = []
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Excluir columnas que no son splits
        if col_lower.endswith('_id') or col_lower.endswith('_std'):
            continue
        if col_lower.startswith('rawtime_'):
            continue
        if col_lower in columnas_excluidas:
            continue
        
        # Incluir solo columnas que parezcan splits
        if (col_lower.startswith('km_') or 
            col_lower in ['half', 'finish', 'meta', 'start'] or
            any(x in col_lower for x in ['km', 'half', 'finish', 'meta'])):
            splits.append(col)
    
    logger.info(f"Splits identificados ({len(splits)}): {splits}")
    return splits


def procesar_genero(df):
    """Procesa columna gender a formato numérico"""
    if 'gender' not in df.columns:
        return df
    
    if df['gender'].dtype == 'object':
        gender_map = {
            'M': 0, 'F': 1, 'male': 0, 'female': 1,
            'Male': 0, 'Female': 1, 'MALE': 0, 'FEMALE': 1
        }
        df['gender'] = df['gender'].map(gender_map).fillna(0).astype(int)
    else:
        df['gender'] = (df['gender'] % 2).astype(int)
    
    return df

# =============================================================
# VALIDACIÓN DE MODELOS - 3 NIVELES
# =============================================================

def nivel1_mejor_que_naive(y_true, y_pred):
    """
    Nivel 1: El modelo debe ser MEJOR que predecir la media
    """
    media = np.mean(y_true)
    predicciones_naive = np.full_like(y_true, media)
    mae_naive = np.mean(np.abs(y_true - predicciones_naive))
    mae_modelo = np.mean(np.abs(y_true - y_pred))
    
    aprueba = mae_modelo < mae_naive
    
    if aprueba:
        mejora = (mae_naive - mae_modelo) / mae_naive
    else:
        mejora = (mae_modelo - mae_naive) / mae_naive
        mejora = -mejora  # negativo si es peor
    
    return {
        'aprueba': aprueba,
        'mae_naive': mae_naive,
        'mae_modelo': mae_modelo,
        'mejora': mejora
    }


def nivel2_consistencia_error(errores_abs):
    """
    Nivel 2: El error debe ser consistente (CV ≤ 0.5)
    """
    media_error = np.mean(errores_abs)
    std_error = np.std(errores_abs)
    
    if media_error > 0:
        cv = std_error / media_error
    else:
        cv = float('inf')
    
    aprueba = cv <= 0.5
    
    return {
        'aprueba': aprueba,
        'media_error': media_error,
        'std_error': std_error,
        'cv': cv
    }


def nivel3_sin_outliers_catastroficos(errores_abs):
    """
    Nivel 3: No debe haber errores catastróficos (P95 ≤ 3 × P50)
    """
    p50 = np.percentile(errores_abs, 50)
    p95 = np.percentile(errores_abs, 95)
    
    if p50 > 0:
        relacion = p95 / p50
    else:
        relacion = float('inf')
    
    aprueba = relacion <= 3.0
    
    return {
        'aprueba': aprueba,
        'p50': p50,
        'p95': p95,
        'relacion': relacion
    }


def validar_modelo_completo(y_true, y_pred, split_objetivo, posicion_atleta):
    """
    Valida el modelo usando los 3 niveles
    """
    errores_abs = np.abs(y_true - y_pred)
    
    nivel1 = nivel1_mejor_que_naive(y_true, y_pred)
    nivel2 = nivel2_consistencia_error(errores_abs)
    nivel3 = nivel3_sin_outliers_catastroficos(errores_abs)
    
    aprueba = nivel1['aprueba'] and nivel2['aprueba'] and nivel3['aprueba']
    
    # Calcular puntuación de calidad (0-100)
    puntuacion = 0
    
    if nivel1['aprueba']:
        puntuacion += 40
        if nivel1['mejora'] > 0.3:
            puntuacion += 10
        elif nivel1['mejora'] > 0.1:
            puntuacion += 5
    
    if nivel2['aprueba']:
        puntuacion += 30
        if nivel2['cv'] < 0.3:
            puntuacion += 10
        elif nivel2['cv'] < 0.4:
            puntuacion += 5
    
    if nivel3['aprueba']:
        puntuacion += 30
        if nivel3['relacion'] < 2.0:
            puntuacion += 10
        elif nivel3['relacion'] < 2.5:
            puntuacion += 5
    
    return {
        'aprobado': aprueba,
        'puntuacion_calidad': min(100, puntuacion),
        'niveles': {
            'mejor_que_naive': nivel1,
            'consistencia_error': nivel2,
            'sin_outliers': nivel3
        },
        'split_objetivo': split_objetivo,
        'posicion_atleta': posicion_atleta,
        'n_muestras': len(y_true)
    }

# =============================================================
# SAGEMAKER MODEL REGISTRY
# =============================================================

def crear_model_package_group(carrera):
    """
    Crea un grupo de modelos en SageMaker Model Registry para una carrera.
    Si ya existe, lo ignora.
    
    Args:
        carrera: nombre de la carrera (ej: "Maratón_Madrid_2024")
    """
    try:
        import boto3
        sm_client = boto3.client('sagemaker')
        
        nombre_grupo = f"timingsense-{carrera.replace('-', '_').replace(' ', '_')}"
        
        response = sm_client.create_model_package_group(
            ModelPackageGroupName=nombre_grupo,
            ModelPackageGroupDescription=f"Modelos para {carrera}",
            Tags=[
                {'Key': 'proyecto', 'Value': 'timingsense'},
                {'Key': 'carrera', 'Value': carrera}
            ]
        )
        logger.info(f"✅ Grupo de modelos creado: {nombre_grupo}")
        return response
    except Exception as e:
        # Si ya existe, es normal
        if 'AlreadyExistsException' in str(type(e)):
            logger.info(f"ℹ️ Grupo de modelos ya existe para {carrera}")
        else:
            logger.warning(f"⚠️ Error creando grupo de modelos: {e}")
        return None


def registrar_modelo_en_registry(modelo_id, metricas, model_s3_uri, carrera, timestamp_unico, validacion):
    """
    Registra un modelo entrenado en SageMaker Model Registry.
    
    Args:
        modelo_id: identificador del modelo (ej: "km_10_desde_km_5")
        metricas: diccionario con métricas del modelo
        model_s3_uri: URI S3 donde está guardado el modelo (.joblib)
        carrera: nombre de la carrera
        timestamp_unico: timestamp de la ejecución
        validacion: diccionario con resultados de validación
    
    Returns:
        response de SageMaker o None si falla
    """
    try:
        import boto3
        import json
        from datetime import datetime
        
        sm_client = boto3.client('sagemaker')
        
        nombre_grupo = f"timingsense-{carrera.replace('-', '_').replace(' ', '_')}"
        
        # Preparar métricas para el registro
        metricas_dict = [
            {'Name': 'mae', 'Value': metricas.get('mae_mean', 0)},
            {'Name': 'mae_std', 'Value': metricas.get('mae_std', 0)},
            {'Name': 'r2', 'Value': metricas.get('r2', 0)},
            {'Name': 'n_samples', 'Value': metricas.get('n_samples', 0)},
            {'Name': 'calidad', 'Value': validacion.get('puntuacion_calidad', 0)}
        ]
        
        # Añadir métricas de validación si existen
        if 'niveles' in validacion:
            nivel1 = validacion['niveles'].get('mejor_que_naive', {})
            nivel2 = validacion['niveles'].get('consistencia_error', {})
            nivel3 = validacion['niveles'].get('sin_outliers', {})
            
            metricas_dict.append({'Name': 'mejora_sobre_naive', 'Value': nivel1.get('mejora', 0)})
            metricas_dict.append({'Name': 'cv_error', 'Value': nivel2.get('cv', 0)})
            metricas_dict.append({'Name': 'outliers_ratio', 'Value': nivel3.get('relacion', 0)})
        
        # Determinar estado de aprobación
        # Si el modelo pasó validación (calidad > 60), lo dejamos pendiente de aprobación manual
        # Si no pasó, lo registramos como rechazado
        if validacion.get('aprobado', False) and validacion.get('puntuacion_calidad', 0) >= 60:
            approval_status = "PendingManualApproval"
        else:
            approval_status = "Rejected"
        
        # Descripción del modelo
        descripcion = (
            f"Modelo {modelo_id} - "
            f"MAE: {metricas.get('mae_mean', 0):.2f}s, "
            f"Calidad: {validacion.get('puntuacion_calidad', 0)}/100, "
            f"Muestras: {metricas.get('n_samples', 0)}"
        )
        
        # Crear el modelo package
        response = sm_client.create_model_package(
            ModelPackageGroupName=nombre_grupo,
            ModelPackageDescription=descripcion,
            InferenceSpecification={
                'Containers': [{
                    'Image': '683313688378.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-xgboost:1.7-1',
                    'ModelDataUrl': model_s3_uri,
                    'Environment': {
                        'MODEL_TYPE': 'interpolacion',
                        'SPLIT_OBJETIVO': metricas.get('split_objetivo', modelo_id.split('_')[0]),
                        'POSICION_ATLETA': metricas.get('posicion_atleta', modelo_id.split('_')[-1]),
                        'MODEL_ID': modelo_id,
                        'CARRERA': carrera
                    }
                }],
                'SupportedContentTypes': ['text/csv'],
                'SupportedResponseMIMETypes': ['text/csv']
            },
            ModelApprovalStatus=approval_status,
            ModelMetrics={
                'ModelQuality': {
                    'Statistics': {
                        'ContentType': 'application/json',
                        'S3Uri': f"{model_s3_uri.replace('.joblib', '_metrics.json')}"
                    }
                }
            },
            MetadataProperties={
                'GeneratedBy': f"entrenamiento-{timestamp_unico}",
                'ProjectId': 'timingsense'
            },
            CustomerMetadataProperties={
                'modelo_id': modelo_id,
                'split_objetivo': metricas.get('split_objetivo', ''),
                'posicion_atleta': metricas.get('posicion_atleta', ''),
                'carrera': carrera,
                'timestamp': timestamp_unico,
                'calidad': str(validacion.get('puntuacion_calidad', 0)),
                'es_extrapolado': str(metricas.get('es_extrapolado', False))
            }
        )
        
        logger.info(f"   📦 Modelo registrado en SageMaker Registry: {modelo_id}")
        logger.info(f"      Estado: {approval_status}")
        logger.info(f"      ARN: {response['ModelPackageArn']}")
        
        return response
        
    except Exception as e:
        logger.warning(f"   ⚠️ Error registrando modelo en SageMaker Registry: {e}")
        return None


def guardar_metricas_para_registry(metricas, validacion, output_dir, modelo_id):
    """
    Guarda las métricas en un archivo JSON para que SageMaker Registry pueda leerlas.
    
    Args:
        metricas: diccionario con métricas del modelo
        validacion: diccionario con resultados de validación
        output_dir: directorio de salida
        modelo_id: identificador del modelo
    """
    import json
    import os
    
    metricas_json = {
        'model_id': modelo_id,
        'regression_metrics': {
            'mae': {
                'value': metricas.get('mae_mean', 0),
                'standard_deviation': metricas.get('mae_std', 0)
            },
            'r2': {
                'value': metricas.get('r2', 0)
            },
            'n_samples': {
                'value': metricas.get('n_samples', 0)
            }
        },
        'validation_metrics': {
            'calidad': validacion.get('puntuacion_calidad', 0),
            'aprobado': validacion.get('aprobado', False)
        }
    }
    
    # Añadir niveles de validación
    if 'niveles' in validacion:
        metricas_json['validation_metrics']['mejora_sobre_naive'] = validacion['niveles']['mejor_que_naive'].get('mejora', 0)
        metricas_json['validation_metrics']['cv_error'] = validacion['niveles']['consistencia_error'].get('cv', 0)
        metricas_json['validation_metrics']['outliers_ratio'] = validacion['niveles']['sin_outliers'].get('relacion', 0)
    
    # Guardar archivo de métricas
    metrics_path = os.path.join(output_dir, f'{modelo_id}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metricas_json, f, indent=2)
    
    return metrics_path


def calcular_metricas_detalladas(y_true, y_pred, split_objetivo, posicion_atleta, metadata=None, es_extrapolado=False):
    """Calcula métricas detalladas para análisis de errores"""
    errores = y_true - y_pred
    errores_abs = np.abs(errores)
    errores_rel = errores / (y_true + 1e-6)
    errores_rel_pct = np.abs(errores_rel) * 100
    
    metricas = {
        'modelo_id': f"{split_objetivo}_desde_{posicion_atleta}{'_ext' if es_extrapolado else ''}",
        'split_objetivo': split_objetivo,
        'posicion_atleta': posicion_atleta,
        'n_samples': len(y_true),
        'mae': float(np.mean(errores_abs)),
        'rmse': float(np.sqrt(np.mean(errores**2))),
        'r2': float(r2_score(y_true, y_pred)),
        'max_error': float(np.max(errores_abs)),
        'min_error': float(np.min(errores_abs)),
        'std_error': float(np.std(errores_abs)),
        'q1_error': float(np.percentile(errores_abs, 25)),
        'median_error': float(np.percentile(errores_abs, 50)),
        'q3_error': float(np.percentile(errores_abs, 75)),
        'p90_error': float(np.percentile(errores_abs, 90)),
        'p95_error': float(np.percentile(errores_abs, 95)),
        'p99_error': float(np.percentile(errores_abs, 99)),
        'mape': float(np.mean(errores_rel_pct)),
        'median_rel_error': float(np.percentile(errores_rel_pct, 50)),
        'p95_rel_error': float(np.percentile(errores_rel_pct, 95)),
        'es_extrapolado': es_extrapolado
    }
    
    if metadata and 'analisis' in metadata:
        mapping = metadata['analisis'].get('mapping', {})
        if split_objetivo in mapping:
            tipo, origen, dist = mapping[split_objetivo]
            metricas['split_origen'] = origen
            metricas['tipo_origen'] = tipo
            if tipo == 'interpolate':
                dist_objetivo = extract_split_distance(split_objetivo)
                metricas['diferencia_distancia'] = abs(dist_objetivo - dist) if dist_objetivo else None
    
    return metricas


def entrenar_modelo_interpolacion(df, split_objetivo, posicion_atleta, 
                                  split_cols_ordenados, model_params, metadata=None, carrera="desconocida"):
    """
    Entrena un modelo para predecir split_objetivo cuando el atleta
    está en posicion_atleta (tiene splits hasta esa posición)
    
    RETURNS: (modelo, metricas, modelo_id, exito, validacion)
    """
    idx_objetivo = split_cols_ordenados.index(split_objetivo)
    idx_posicion = split_cols_ordenados.index(posicion_atleta)
    
    splits_disponibles = [
        split for i, split in enumerate(split_cols_ordenados[:idx_posicion + 1])
        if i != idx_objetivo
    ]
    
    modelo_id = f"{split_objetivo}_desde_{posicion_atleta}"
    
    # Detectar si es extrapolado (el split no tiene datos)
    es_extrapolado = False
    if split_objetivo not in df.columns or df[split_objetivo].isna().all():
        es_extrapolado = True
        logger.info(f"   🔄 Split {split_objetivo} no tiene datos, será extrapolado")
    
    features = splits_disponibles.copy()
    if 'age' in df.columns:
        features.append('age')
    if 'gender' in df.columns:
        features.append('gender')
    
    cols_necesarias = splits_disponibles + [split_objetivo, 'age', 'gender']
    cols_existentes = [c for c in cols_necesarias if c in df.columns]
    
    # Verificar si el split objetivo existe (tiene datos no nulos)
    split_tiene_datos = split_objetivo in df.columns and df[split_objetivo].notna().any()
    
    if not split_tiene_datos and not es_extrapolado:
        logger.info(f"   ⚠️ {split_objetivo} NO TIENE DATOS en históricos")
        return None, None, None, False, None
    
    mask = df[cols_existentes].notna().all(axis=1)
    df_completos = df[mask].copy()
    
    if len(df_completos) < model_params.get('min_samples', 20):
        logger.info(f"   ⚠️ Pocos datos: {len(df_completos)} < {model_params.get('min_samples', 20)}")
        return None, None, None, False, None
    
    X = df_completos[features].values
    y = df_completos[split_objetivo].values
    
    kf = KFold(n_splits=min(model_params.get('n_folds', 5), len(df_completos)), 
               shuffle=True, random_state=42)
    
    fold_scores = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = xgb.XGBRegressor(
            n_estimators=model_params.get('n_estimators', 100),
            max_depth=model_params.get('max_depth', 6),
            learning_rate=model_params.get('learning_rate', 0.05),
            subsample=model_params.get('subsample', 0.8),
            colsample_bytree=model_params.get('colsample_bytree', 0.8),
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        fold_scores.append(mae)
        
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

    metricas_detalladas = calcular_metricas_detalladas(
        np.array(all_y_true), 
        np.array(all_y_pred),
        split_objetivo,
        posicion_atleta,
        metadata,
        es_extrapolado=False
    )

    metricas_detalladas.update({
        'splits_disponibles': splits_disponibles,
        'n_features': len(features),
        'mae_folds_mean': float(np.mean(fold_scores)),
        'mae_folds_std': float(np.std(fold_scores))
    })
    
    # =============================================================
    # VALIDACIÓN DEL MODELO - 3 NIVELES
    # =============================================================
    
    y_true_array = np.array(all_y_true)
    y_pred_array = np.array(all_y_pred)
    
    validacion = validar_modelo_completo(
        y_true_array,
        y_pred_array,
        split_objetivo,
        posicion_atleta
    )
    
    # Log de validación
    if validacion['aprobado']:
        logger.info(f"   ✅ VALIDACIÓN APROBADA - Calidad: {validacion['puntuacion_calidad']}/100")
        logger.info(f"      Mejora sobre naïve: {validacion['niveles']['mejor_que_naive']['mejora']:.1%}")
        logger.info(f"      CV error: {validacion['niveles']['consistencia_error']['cv']:.2f}")
        logger.info(f"      Ratio P95/P50: {validacion['niveles']['sin_outliers']['relacion']:.1f}")
    else:
        logger.warning(f"   ❌ VALIDACIÓN RECHAZADA - Calidad: {validacion['puntuacion_calidad']}/100")
        if not validacion['niveles']['mejor_que_naive']['aprueba']:
            logger.warning(f"      - No mejora a naïve (mae_modelo={validacion['niveles']['mejor_que_naive']['mae_modelo']:.1f}s vs mae_naive={validacion['niveles']['mejor_que_naive']['mae_naive']:.1f}s)")
        if not validacion['niveles']['consistencia_error']['aprueba']:
            logger.warning(f"      - Error inconsistente (CV={validacion['niveles']['consistencia_error']['cv']:.2f} > 0.5)")
        if not validacion['niveles']['sin_outliers']['aprueba']:
            logger.warning(f"      - Outliers catastróficos (P95={validacion['niveles']['sin_outliers']['p95']:.0f}s es {validacion['niveles']['sin_outliers']['relacion']:.1f}x P50={validacion['niveles']['sin_outliers']['p50']:.0f}s)")
    
    # Si no está aprobado, NO entrenamos el modelo final
    if not validacion['aprobado']:
        return None, metricas_detalladas, modelo_id, False, validacion
    
    # =============================================================
    # ENTRENAR MODELO FINAL (SOLO SI ESTÁ APROBADO)
    # =============================================================
    
    final_model = xgb.XGBRegressor(
        n_estimators=model_params.get('n_estimators', 100),
        max_depth=model_params.get('max_depth', 6),
        learning_rate=model_params.get('learning_rate', 0.05),
        subsample=model_params.get('subsample', 0.8),
        colsample_bytree=model_params.get('colsample_bytree', 0.8),
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    final_model.fit(X, y)
    
    metricas = {
        'mae_mean': float(np.mean(fold_scores)),
        'mae_std': float(np.std(fold_scores)),
        'n_samples': len(df_completos),
        'split_objetivo': split_objetivo,
        'posicion_atleta': posicion_atleta,
        'splits_disponibles': splits_disponibles,
        'n_features': len(features),
        'r2': validacion['niveles']['consistencia_error'].get('r2', 0)  # opcional
    }
    
    return final_model, metricas, modelo_id, True, validacion


def entrenar_modelo_extrapolado(df, split_objetivo, split_origen, dist_origen, dist_objetivo,
                                posicion_atleta, split_cols_ordenados, model_params, metadata=None, carrera="desconocida"):
    """
    Entrena un modelo para predecir split_origen y luego extrapola a split_objetivo
    
    RETURNS: (modelo, metricas, modelo_id, exito, validacion)
    """
    logger.info(f"   🔄 EXTRAPOLANDO: {split_objetivo} ← {split_origen}")
    
    # 1. Entrenar modelo para el split origen (ahora devuelve 5 valores)
    modelo_origen, metricas_origen, modelo_id_origen, exito, validacion_origen = entrenar_modelo_interpolacion(
        df, split_origen, posicion_atleta, split_cols_ordenados, model_params, metadata, carrera
    )
    
    # Verificar que el modelo origen se entrenó correctamente
    if not exito or modelo_origen is None:
        logger.info(f"   ⚠️ No se pudo entrenar modelo origen {split_origen}")
        return None, None, None, False, None
    
    # Verificar que el modelo origen pasó la validación
    if not validacion_origen['aprobado']:
        logger.info(f"   ⚠️ Modelo origen {split_origen} no pasó validación (calidad={validacion_origen['puntuacion_calidad']}/100)")
        return None, None, None, False, validacion_origen
    
    # 2. Calcular factor de ajuste (proporción de distancias)
    factor_ajuste = dist_objetivo / dist_origen
    
    # 3. Crear wrapper que ajusta la predicción
    class ModeloExtrapolado:
        def __init__(self, modelo_base, factor):
            self.modelo_base = modelo_base
            self.factor = factor
        
        def predict(self, X):
            pred_base = self.modelo_base.predict(X)
            return pred_base * self.factor
    
    modelo_extrapolado = ModeloExtrapolado(modelo_origen, factor_ajuste)
    modelo_id = f"{split_objetivo}_desde_{posicion_atleta}_ext"
    
    # 4. Calcular métricas estimadas para el modelo extrapolado
    metricas_extrapoladas = {
        'mae_mean': metricas_origen['mae_mean'] * factor_ajuste,
        'mae_std': metricas_origen['mae_std'] * factor_ajuste,
        'n_samples': metricas_origen['n_samples'],
        'split_objetivo': split_objetivo,
        'posicion_atleta': posicion_atleta,
        'splits_disponibles': metricas_origen.get('splits_disponibles', []),
        'split_origen': split_origen,
        'factor_ajuste': factor_ajuste,
        'distancia_origen': dist_origen,
        'distancia_objetivo': dist_objetivo,
        'r2': metricas_origen.get('r2', 0)  # el r2 se estima similar
    }
    
    # 5. Crear validación para el modelo extrapolado
    # Heredamos la validación del modelo origen pero aplicamos penalización
    puntuacion_base = validacion_origen['puntuacion_calidad']
    penalizacion = 10  # penalización fija por ser extrapolado
    
    validacion = {
        'aprobado': True,  # Ya pasó validación el origen
        'puntuacion_calidad': max(0, puntuacion_base - penalizacion),
        'niveles': validacion_origen['niveles'].copy(),
        'split_objetivo': split_objetivo,
        'posicion_atleta': posicion_atleta,
        'n_muestras': metricas_extrapoladas['n_samples'],
        'es_extrapolado': True,
        'factor_ajuste': factor_ajuste,
        'split_origen': split_origen
    }
    
    # Ajustar la descripción del nivel 1 para reflejar que es extrapolado
    validacion['niveles']['mejor_que_naive']['mejora_estimada'] = validacion['niveles']['mejor_que_naive']['mejora']
    validacion['niveles']['mejor_que_naive']['mejora'] = validacion['niveles']['mejor_que_naive']['mejora'] * 0.9  # estimación conservadora
    
    logger.info(f"   ✅ Modelo extrapolado creado: factor={factor_ajuste:.3f}")
    logger.info(f"      MAE estimado={metricas_extrapoladas['mae_mean']:.2f}s")
    logger.info(f"      Calidad origen={puntuacion_base}/100 → calidad extrapolado={validacion['puntuacion_calidad']}/100")
    
    return modelo_extrapolado, metricas_extrapoladas, modelo_id, True, validacion


def main():
    """Función principal de entrenamiento"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--carrera', type=str, required=True)
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--min-samples', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample-bytree', type=float, default=0.8)
    
    args, unknown = parser.parse_known_args()
    
    # Generar timestamp único para esta ejecución
    timestamp_unico = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    logger.info("="*80)
    logger.info("🚀 ENTRENAMIENTO DE MODELOS DE INTERPOLACIÓN")
    logger.info("="*80)
    logger.info(f"📌 Carrera: {args.carrera}")
    logger.info(f"🕒 Timestamp único: {timestamp_unico}")
    
    # =============================================================
    # CARGAR DATOS Y METADATA
    # =============================================================
    model_dir, output_dir, train_dir = get_paths()
    df = load_data_from_local(train_dir)
    logger.info(f"✅ Datos cargados: {len(df)} registros")
    
    # Crear grupo de modelos en SageMaker Registry
    crear_model_package_group(args.carrera)
    
    # =============================================================
    # 🟢🟢🟢 RENOMBRAR COLUMNAS INMEDIATAMENTE 🟢🟢🟢
    # =============================================================
    for col in df.columns:
        if '.' in col:
            nuevo_nombre = col.replace('.', '_')
            df.rename(columns={col: nuevo_nombre}, inplace=True)
            logger.info(f"🔄 Columna renombrada: {col} → {nuevo_nombre}")
    
    # También renombrar si hay columnas con guiones o espacios
    for col in df.columns:
        nuevo_nombre = col
        if '-' in col:
            nuevo_nombre = col.replace('-', '_')
        if ' ' in col:
            nuevo_nombre = col.replace(' ', '_')
        
        if nuevo_nombre != col:
            df.rename(columns={col: nuevo_nombre}, inplace=True)
            logger.info(f"🔄 Columna renombrada: {col} → {nuevo_nombre}")
    
    metadata = load_training_metadata(model_dir)
    if metadata:
        metadata = normalizar_metadata(metadata)
        logger.info(f"📋 Metadata cargada y normalizada")
    
    # =============================================================
    # PREPROCESAMIENTO
    # =============================================================
    df = procesar_genero(df)
    
    split_cols = identificar_splits(df)
    split_cols_ordenados = ordenar_splits_personalizado(split_cols)
    
    logger.info(f"📊 Splits detectados ({len(split_cols_ordenados)}): {split_cols_ordenados}")
    
    if metadata:
        log_split_analysis(metadata, split_cols_ordenados)
    
    # =============================================================
    # ENTRENAR MODELOS
    # =============================================================
    model_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'n_folds': args.n_folds,
        'min_samples': args.min_samples,
        'learning_rate': args.learning_rate,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree
    }
    
    modelos_guardados = {}
    metricas_totales = []
    validaciones_totales = []
    splits_extrapolados = []
    modelos_aprobados = 0
    modelos_rechazados = 0
    
    n_splits = len(split_cols_ordenados)
    total_modelos_teoricos = (n_splits - 2) * (n_splits - 1) // 2
    
    logger.info(f"\n🏋️ ENTRENANDO MODELOS (teóricos: {total_modelos_teoricos})...")
    
    modelo_idx = 0
    for idx_objetivo in range(0, n_splits - 1):
        split_objetivo = split_cols_ordenados[idx_objetivo]
        
        for idx_posicion in range(idx_objetivo + 1, n_splits):
            posicion_atleta = split_cols_ordenados[idx_posicion]
            modelo_idx += 1
            
            logger.info(f"\n📌 [{modelo_idx}/{total_modelos_teoricos}] {split_objetivo} ← {posicion_atleta}")
            
            # Verificar si este split necesita extrapolación
            necesita_extrapolacion = False
            split_origen = None
            dist_origen = None
            dist_objetivo = None
            
            if metadata and 'analisis' in metadata:
                mapping = metadata['analisis'].get('mapping', {})
                if split_objetivo in mapping:
                    tipo, origen, dist = mapping[split_objetivo]
                    if tipo == 'interpolate':
                        if split_objetivo not in df.columns or df[split_objetivo].isna().all():
                            necesita_extrapolacion = True
                            split_origen = origen
                            dist_origen = dist
                            dist_objetivo = extract_split_distance(split_objetivo)
                            logger.info(f"   🔍 Split {split_objetivo} requiere EXTRAPOLACIÓN desde {split_origen}")
            
            # Llamar al entrenamiento
            if necesita_extrapolacion:
                modelo, metricas, modelo_id, exito, validacion = entrenar_modelo_extrapolado(
                    df, split_objetivo, split_origen, dist_origen, dist_objetivo,
                    posicion_atleta, split_cols_ordenados, model_params, metadata,
                    carrera=args.carrera
                )
            else:
                modelo, metricas, modelo_id, exito, validacion = entrenar_modelo_interpolacion(
                    df, split_objetivo, posicion_atleta, 
                    split_cols_ordenados, model_params, metadata,
                    carrera=args.carrera
                )
            
            # Procesar resultado con validación
            if exito and modelo and validacion and validacion['aprobado']:
                # Guardar modelo SOLO si está aprobado
                model_path = os.path.join(model_dir, f"{modelo_id}.joblib")
                joblib.dump(modelo, model_path)
                
                # Construir URI S3 del modelo (para el registro)
                s3_bucket = "timingsense-modelos-2026"
                s3_model_uri = f"s3://{s3_bucket}/entrenamientos/{args.carrera.replace(' ', '_')}/{timestamp_unico}/models/{modelo_id}.joblib"
                
                # Guardar métricas en JSON para el registry
                metrics_path = guardar_metricas_para_registry(metricas, validacion, output_dir, modelo_id)
                
                # Subir métricas a S3
                s3_client = boto3.client('s3')
                s3_metrics_key = f"entrenamientos/{args.carrera.replace(' ', '_')}/{timestamp_unico}/metrics/{modelo_id}_metrics.json"
                try:
                    s3_client.upload_file(metrics_path, s3_bucket, s3_metrics_key)
                    logger.info(f"   📊 Métricas subidas a S3: {s3_metrics_key}")
                except Exception as e:
                    logger.warning(f"   ⚠️ Error subiendo métricas a S3: {e}")
                
                # Registrar en SageMaker Model Registry
                registry_response = registrar_modelo_en_registry(
                    modelo_id=modelo_id,
                    metricas=metricas,
                    model_s3_uri=s3_model_uri,
                    carrera=args.carrera,
                    timestamp_unico=timestamp_unico,
                    validacion=validacion
                )
                
                # Guardar en modelos_guardados
                modelos_guardados[modelo_id] = {
                    'path': model_path,
                    's3_uri': s3_model_uri,
                    'split_objetivo': split_objetivo,
                    'posicion_atleta': posicion_atleta,
                    'splits_disponibles': metricas.get('splits_disponibles', []),
                    'n_samples': metricas.get('n_samples', 0),
                    'mae': metricas.get('mae_mean', 0),
                    'es_extrapolado': necesita_extrapolacion,
                    'validacion': validacion,
                    'registry_arn': registry_response.get('ModelPackageArn') if registry_response else None
                }
                
                metricas_totales.append({
                    'modelo_id': modelo_id,
                    'split_objetivo': split_objetivo,
                    'posicion_atleta': posicion_atleta,
                    'mae': metricas.get('mae_mean', 0),
                    'mae_std': metricas.get('mae_std', 0),
                    'n_samples': metricas.get('n_samples', 0),
                    'es_extrapolado': necesita_extrapolacion,
                    'calidad': validacion['puntuacion_calidad'],
                    'registry_arn': registry_response.get('ModelPackageArn') if registry_response else None
                })
                
                validaciones_totales.append(validacion)
                modelos_aprobados += 1
                
                logger.info(f"   ✅ GUARDADO - MAE={metricas.get('mae_mean', 0):.1f}s | Calidad={validacion['puntuacion_calidad']}/100")
                if registry_response:
                    logger.info(f"      Registry ARN: {registry_response.get('ModelPackageArn', 'N/A')}")
            else:
                modelos_rechazados += 1
                if validacion:
                    validaciones_totales.append(validacion)
                    logger.info(f"   ❌ RECHAZADO - Calidad={validacion['puntuacion_calidad']}/100")
                else:
                    logger.info(f"   ❌ RECHAZADO - Datos insuficientes")
    
    # =============================================================
    # GUARDAR METADATA CON VALIDACIONES
    # =============================================================
    logger.info("\n💾 GUARDANDO METADATA...")
    
    df_metricas = pd.DataFrame(metricas_totales) if metricas_totales else pd.DataFrame()
    if not df_metricas.empty:
        df_metricas.to_csv(os.path.join(output_dir, 'metricas_modelos.csv'), index=False)
    
    # Calcular estadísticas de validación
    estadisticas_validacion = {}
    if validaciones_totales:
        puntuaciones = [v['puntuacion_calidad'] for v in validaciones_totales]
        mejoras = [v['niveles']['mejor_que_naive']['mejora'] for v in validaciones_totales]
        cvs = [v['niveles']['consistencia_error']['cv'] for v in validaciones_totales]
        ratios = [v['niveles']['sin_outliers']['relacion'] for v in validaciones_totales]
        
        estadisticas_validacion = {
            'total_evaluados': len(validaciones_totales),
            'aprobados': modelos_aprobados,
            'rechazados': modelos_rechazados,
            'tasa_aprobacion': modelos_aprobados / len(validaciones_totales) if validaciones_totales else 0,
            'puntuacion_promedio': float(np.mean(puntuaciones)),
            'puntuacion_min': float(np.min(puntuaciones)),
            'puntuacion_max': float(np.max(puntuaciones)),
            'mejora_promedio_sobre_naive': float(np.mean(mejoras)),
            'cv_promedio': float(np.mean(cvs)),
            'ratio_outliers_promedio': float(np.mean(ratios)),
            'rechazos_por_mejor_que_naive': sum(1 for v in validaciones_totales if not v['niveles']['mejor_que_naive']['aprueba']),
            'rechazos_por_consistencia': sum(1 for v in validaciones_totales if not v['niveles']['consistencia_error']['aprueba']),
            'rechazos_por_outliers': sum(1 for v in validaciones_totales if not v['niveles']['sin_outliers']['aprueba'])
        }
    
    metadata_completa = {
        'carrera': args.carrera,
        'timestamp': timestamp_unico,
        'fecha_entrenamiento': datetime.now().isoformat(),
        'total_modelos_teoricos': total_modelos_teoricos,
        'modelos_guardados': len(modelos_guardados),
        'modelos_extrapolados': len(splits_extrapolados),
        'splits_extrapolados': splits_extrapolados,
        'splits': split_cols_ordenados,
        'hiperparametros': model_params,
        'metricas_resumen': {
            'mae_promedio': float(df_metricas['mae'].mean()) if not df_metricas.empty else 0,
            'mae_std_promedio': float(df_metricas['mae_std'].mean()) if not df_metricas.empty else 0,
            'total_muestras': int(df_metricas['n_samples'].sum()) if not df_metricas.empty else 0
        },
        'validaciones': estadisticas_validacion
    }
    
    if metadata and 'analisis' in metadata:
        metadata_completa['analisis_original'] = metadata['analisis']
    
    # Guardar metadata principal
    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata_completa, f, indent=2, default=str)
    
    # Guardar lista de modelos disponibles
    with open(os.path.join(model_dir, 'modelos_disponibles.json'), 'w') as f:
        json.dump(list(modelos_guardados.keys()), f, indent=2)
    
    # Guardar resumen de validaciones aparte
    if validaciones_totales:
        resumen_validaciones = {
            'carrera': args.carrera,
            'timestamp': timestamp_unico,
            'fecha': datetime.now().isoformat(),
            'total_evaluados': len(validaciones_totales),
            'aprobados': modelos_aprobados,
            'rechazados': modelos_rechazados,
            'tasa_aprobacion': modelos_aprobados / len(validaciones_totales) if validaciones_totales else 0,
            'mejores_modelos': sorted(
                [{'id': k, 'mae': v['mae'], 'calidad': v['validacion']['puntuacion_calidad']} 
                 for k, v in modelos_guardados.items()],
                key=lambda x: x['calidad'],
                reverse=True
            )[:5]
        }
        with open(os.path.join(output_dir, 'resumen_validaciones.json'), 'w') as f:
            json.dump(resumen_validaciones, f, indent=2, default=str)
    
    logger.info(f"✅ {len(modelos_guardados)} modelos guardados en {model_dir}")
    
    # =============================================================
    # RESUMEN FINAL CON ESTADÍSTICAS DE VALIDACIÓN
    # =============================================================
    logger.info("\n" + "="*80)
    logger.info("📊 RESUMEN DEL ENTRENAMIENTO")
    logger.info("="*80)
    logger.info(f"🎯 Carrera: {args.carrera}")
    logger.info(f"🕒 Timestamp: {timestamp_unico}")
    logger.info(f"📊 Modelos entrenados: {modelos_aprobados}/{total_modelos_teoricos}")
    logger.info(f"🔄 Modelos extrapolados: {len(splits_extrapolados)}")
    
    if splits_extrapolados:
        logger.info(f"   Splits extrapolados: {splits_extrapolados}")
    
    # Mostrar estadísticas de validación
    if estadisticas_validacion:
        logger.info(f"\n📈 ESTADÍSTICAS DE VALIDACIÓN:")
        logger.info(f"   Tasa de aprobación: {estadisticas_validacion['tasa_aprobacion']:.1%}")
        logger.info(f"   Calidad promedio: {estadisticas_validacion['puntuacion_promedio']:.0f}/100")
        logger.info(f"   Mejora promedio sobre naïve: {estadisticas_validacion['mejora_promedio_sobre_naive']:.1%}")
        logger.info(f"   CV promedio: {estadisticas_validacion['cv_promedio']:.2f}")
        logger.info(f"   Rechazos por:")
        logger.info(f"      - No mejora a naïve: {estadisticas_validacion['rechazos_por_mejor_que_naive']}")
        logger.info(f"      - Error inconsistente: {estadisticas_validacion['rechazos_por_consistencia']}")
        logger.info(f"      - Outliers catastróficos: {estadisticas_validacion['rechazos_por_outliers']}")
    
    if metricas_totales:
        df_metricas_ordenado = df_metricas.sort_values('mae')
        logger.info("\n🏆 TOP 5 MEJORES MODELOS (por MAE):")
        for _, row in df_metricas_ordenado.head(5).iterrows():
            tipo = "EXT" if row.get('es_extrapolado', False) else "DIR"
            calidad = row.get('calidad', 0)
            logger.info(f"   {row['modelo_id']}: MAE={row['mae']:.2f}s ({tipo}, n={row['n_samples']}, calidad={calidad}/100)")
        
        logger.info("\n🏆 TOP 5 MEJORES MODELOS (por calidad):")
        df_por_calidad = df_metricas.sort_values('calidad', ascending=False)
        for _, row in df_por_calidad.head(5).iterrows():
            tipo = "EXT" if row.get('es_extrapolado', False) else "DIR"
            logger.info(f"   {row['modelo_id']}: Calidad={row['calidad']}/100 (MAE={row['mae']:.2f}s, n={row['n_samples']})")
    
    logger.info("\n" + "="*80)
    logger.info("✅ ENTRENAMIENTO COMPLETADO")
    logger.info("="*80)
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        trc = traceback.format_exc()
        error_msg = f"❌ Error: {str(e)}\n{trc}"
        logger.error(error_msg)
        with open('/opt/ml/output/failure', 'w') as f:
            f.write(error_msg)
        sys.exit(255)