"""
train_prediccion.py - Script de entrenamiento para modelos de predicción con intervalos de confianza
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
# CONFIGURACIÓN S3
# =============================================================
s3_client = boto3.client('s3')
S3_BUCKET_MODELOS = 'timingsense-modelos-2026'

# =============================================================
# FUNCIONES AUXILIARES
# =============================================================

def extract_split_distance(split_name):
    """Extrae la distancia en km de un split"""
    if split_name is None:
        return None
    
    split_lower = split_name.lower()
    
    if split_lower == 'half':
        return 21.0975
    elif split_lower == 'finish':
        return 42.195
    elif split_lower == 'start':
        return 0.0
    
    if split_lower.startswith('km_'):
        try:
            num_str = split_lower[3:].replace('_', '.')
            return float(num_str)
        except:
            return None
    
    return None


def ordenar_splits_por_distancia(split_cols):
    """Ordena splits por distancia real"""
    
    def get_distance(split):
        split_lower = split.lower()
        if split_lower in ['half', 'media']:
            return 21.0975
        elif split_lower in ['finish', 'meta']:
            return 42.195
        elif split_lower.startswith('km_'):
            try:
                num_str = split_lower[3:].replace('_', '.')
                return float(num_str)
            except:
                return float('inf')
        return float('inf')
    
    splits_con_distancia = [(split, get_distance(split)) for split in split_cols]
    splits_ordenados = sorted(splits_con_distancia, key=lambda x: x[1])
    
    logger.info("📊 Orden de splits por distancia:")
    for split, dist in splits_ordenados:
        logger.info(f"   {split}: {dist} km")
    
    return [split for split, _ in splits_ordenados]


def get_paths():
    """Obtiene las rutas estándar de SageMaker"""
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output')
    train_dir = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
    
    logger.info(f"📂 Model directory: {model_dir}")
    logger.info(f"📂 Output directory: {output_dir}")
    logger.info(f"📂 Training data directory: {train_dir}")
    
    return model_dir, output_dir, train_dir


def load_data_from_local(data_dir):
    """Carga todos los archivos Parquet"""
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
    
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"✅ Total: {len(df)} registros")
    return df


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


def identificar_splits(df):
    """Identifica columnas que son splits"""
    columnas_excluidas = ['gender', 'age', 'birthdate', 'athlete_id', 
                          'event_id', 'event_std', 'race_id']
    splits = []
    
    for col in df.columns:
        col_lower = col.lower()
        
        if col_lower.endswith('_id') or col_lower.endswith('_std'):
            continue
        if col_lower.startswith('rawtime_'):
            continue
        if col_lower in columnas_excluidas:
            continue
        if col_lower.startswith('km_') or col_lower in ['half', 'finish', 'meta']:
            splits.append(col)
    
    logger.info(f"📊 Splits identificados ({len(splits)}): {splits}")
    return splits


def crear_prefijo_str(input_splits):
    """Crea el string de prefijo para el nombre del modelo"""
    partes = []
    for s in input_splits:
        if 'km_' in s:
            num = s.replace('km_', '').replace('_', '.')
            partes.append(f"{num}k")
        else:
            partes.append(s)
    return "_".join(partes)

# =============================================================
# VALIDACIÓN DE MODELOS - 3 NIVELES
# =============================================================

def nivel1_mejor_que_naive(y_true, y_pred):
    """Nivel 1: El modelo debe ser MEJOR que predecir la media"""
    media = np.mean(y_true)
    predicciones_naive = np.full_like(y_true, media)
    mae_naive = np.mean(np.abs(y_true - predicciones_naive))
    mae_modelo = np.mean(np.abs(y_true - y_pred))
    
    aprueba = mae_modelo < mae_naive
    
    if aprueba:
        mejora = (mae_naive - mae_modelo) / mae_naive
    else:
        mejora = (mae_modelo - mae_naive) / mae_naive
        mejora = -mejora
    
    return {
        'aprueba': aprueba,
        'mae_naive': mae_naive,
        'mae_modelo': mae_modelo,
        'mejora': mejora
    }


def nivel2_consistencia_error(errores_abs):
    """Nivel 2: El error debe ser consistente (CV ≤ 0.5)"""
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
    """Nivel 3: No debe haber errores catastróficos (P95 ≤ 3 × P50)"""
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


def validar_modelo_completo(y_true, y_pred, input_splits_str, output_split):
    """Valida el modelo usando los 3 niveles"""
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
        'input_splits': input_splits_str,
        'output_split': output_split,
        'n_muestras': len(y_true)
    }


# =============================================================
# SAGEMAKER MODEL REGISTRY
# =============================================================

def crear_model_package_group(carrera):
    """Crea un grupo de modelos en SageMaker Model Registry"""
    try:
        sm_client = boto3.client('sagemaker')
        nombre_grupo = f"timingsense-prediccion-{carrera.replace('-', '_').replace(' ', '_')}"
        
        response = sm_client.create_model_package_group(
            ModelPackageGroupName=nombre_grupo,
            ModelPackageGroupDescription=f"Modelos de predicción para {carrera}",
            Tags=[
                {'Key': 'proyecto', 'Value': 'timingsense'},
                {'Key': 'carrera', 'Value': carrera},
                {'Key': 'tipo', 'Value': 'prediccion'}
            ]
        )
        logger.info(f"✅ Grupo de modelos creado: {nombre_grupo}")
        return response
    except Exception as e:
        if 'AlreadyExistsException' in str(type(e)):
            logger.info(f"ℹ️ Grupo de modelos ya existe para {carrera}")
        else:
            logger.warning(f"⚠️ Error creando grupo: {e}")
        return None


def registrar_modelo_prediccion(modelo_id, metricas, model_s3_uri, carrera, timestamp_unico, validacion, quantile):
    """Registra un modelo de predicción en SageMaker Model Registry"""
    try:
        sm_client = boto3.client('sagemaker')
        nombre_grupo = f"timingsense-prediccion-{carrera.replace('-', '_').replace(' ', '_')}"
        
        # Preparar métricas
        metricas_dict = [
            {'Name': 'mae_test', 'Value': metricas.get('mae_test', 0)},
            {'Name': 'cobertura_test', 'Value': metricas.get('cobertura_test', 0)},
            {'Name': 'n_samples_train', 'Value': metricas.get('n_samples_train', 0)},
            {'Name': 'calidad', 'Value': validacion.get('puntuacion_calidad', 0)}
        ]
        
        # Añadir métricas de validación
        if 'niveles' in validacion:
            nivel1 = validacion['niveles'].get('mejor_que_naive', {})
            nivel2 = validacion['niveles'].get('consistencia_error', {})
            nivel3 = validacion['niveles'].get('sin_outliers', {})
            
            metricas_dict.append({'Name': 'mejora_sobre_naive', 'Value': nivel1.get('mejora', 0)})
            metricas_dict.append({'Name': 'cv_error', 'Value': nivel2.get('cv', 0)})
            metricas_dict.append({'Name': 'outliers_ratio', 'Value': nivel3.get('relacion', 0)})
        
        # Estado de aprobación
        if validacion.get('aprobado', False) and validacion.get('puntuacion_calidad', 0) >= 60:
            approval_status = "PendingManualApproval"
        else:
            approval_status = "Rejected"
        
        descripcion = (
            f"Modelo {modelo_id} - "
            f"MAE: {metricas.get('mae_test', 0):.2f}s, "
            f"Cobertura: {metricas.get('cobertura_test', 0):.1f}%, "
            f"Calidad: {validacion.get('puntuacion_calidad', 0)}/100"
        )
        
        response = sm_client.create_model_package(
            ModelPackageGroupName=nombre_grupo,
            ModelPackageDescription=descripcion,
            InferenceSpecification={
                'Containers': [{
                    'Image': '683313688378.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-xgboost:1.7-1',
                    'ModelDataUrl': model_s3_uri,
                    'Environment': {
                        'MODEL_TYPE': 'prediccion',
                        'QUANTILE': str(quantile),
                        'INPUT_SPLITS': metricas.get('input_splits', ''),
                        'OUTPUT_SPLIT': metricas.get('output_split', ''),
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
                'GeneratedBy': f"entrenamiento-prediccion-{timestamp_unico}",
                'ProjectId': 'timingsense'
            },
            CustomerMetadataProperties={
                'modelo_id': modelo_id,
                'input_splits': metricas.get('input_splits', ''),
                'output_split': metricas.get('output_split', ''),
                'carrera': carrera,
                'timestamp': timestamp_unico,
                'calidad': str(validacion.get('puntuacion_calidad', 0)),
                'quantile': str(quantile),
                'tipo': 'prediccion'
            }
        )
        
        logger.info(f"   📦 Modelo registrado: {modelo_id} (q{quantile})")
        logger.info(f"      Estado: {approval_status}")
        
        return response
        
    except Exception as e:
        logger.warning(f"   ⚠️ Error registrando modelo: {e}")
        return None


def guardar_metricas_prediccion_json(metricas, validacion, output_dir, modelo_id):
    """Guarda métricas en JSON para SageMaker Registry"""
    metricas_json = {
        'model_id': modelo_id,
        'type': 'prediccion',
        'regression_metrics': {
            'mae_test': {
                'value': metricas.get('mae_test', 0)
            },
            'cobertura_test': {
                'value': metricas.get('cobertura_test', 0)
            },
            'n_samples_train': {
                'value': metricas.get('n_samples_train', 0)
            },
            'n_samples_test': {
                'value': metricas.get('n_samples_test', 0)
            }
        },
        'validation_metrics': {
            'calidad': validacion.get('puntuacion_calidad', 0),
            'aprobado': validacion.get('aprobado', False)
        }
    }
    
    if 'niveles' in validacion:
        metricas_json['validation_metrics']['mejora_sobre_naive'] = validacion['niveles']['mejor_que_naive'].get('mejora', 0)
        metricas_json['validation_metrics']['cv_error'] = validacion['niveles']['consistencia_error'].get('cv', 0)
        metricas_json['validation_metrics']['outliers_ratio'] = validacion['niveles']['sin_outliers'].get('relacion', 0)
    
    metrics_path = os.path.join(output_dir, f'{modelo_id}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metricas_json, f, indent=2)
    
    return metrics_path


def entrenar_modelos_quantiles(df, split_cols_ordenados, carrera, quantiles, params, model_dir, output_dir, timestamp_unico):
    """
    Entrena modelos XGBoost con reg:squarederror y calcula cuantiles empíricos
    CON VALIDACIÓN DE 3 NIVELES Y REGISTRO EN SAGEMAKER
    """
    resultados = []
    modelos_guardados = []
    validaciones_totales = []
    modelos_aprobados = 0
    modelos_rechazados = 0
    
    n_splits = len(split_cols_ordenados)
    total_combinaciones = (n_splits - 1) * n_splits // 2
    combinacion_idx = 0
    
    logger.info(f"\n🏋️ ENTRENANDO {total_combinaciones} COMBINACIONES...")
    
    for i in range(1, n_splits):
        input_splits = split_cols_ordenados[:i]
        futuros = split_cols_ordenados[i:]
        
        prefijo_str = crear_prefijo_str(input_splits)
        input_splits_str = "_".join(input_splits)
        
        for futuro in futuros:
            combinacion_idx += 1
            logger.info(f"\n📌 [{combinacion_idx}/{total_combinaciones}] {input_splits} → {futuro}")
            
            # Preparar datos
            cols_req = input_splits + [futuro]
            if 'age' in df.columns:
                cols_req.append('age')
            if 'gender' in df.columns:
                cols_req.append('gender')
            
            df_conf = df.dropna(subset=cols_req).copy()
            if len(df_conf) < params['min_samples']:
                logger.info(f"   ⚠️ Pocos datos: {len(df_conf)} < {params['min_samples']}")
                continue
            
            features = input_splits.copy()
            if 'age' in df_conf.columns:
                features.append('age')
            if 'gender' in df_conf.columns:
                features.append('gender')
            
            X = df_conf[features].values
            y = df_conf[futuro].values
            
            # Dividir en TRAIN (80%) y TEST (20%)
            athletes = df_conf['athlete_id'].unique()
            from sklearn.model_selection import train_test_split
            
            train_athletes, test_athletes = train_test_split(
                athletes, test_size=0.2, random_state=42
            )
            
            train_mask = df_conf['athlete_id'].isin(train_athletes)
            test_mask = df_conf['athlete_id'].isin(test_athletes)
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            # Entrenar modelo
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            model.fit(X_train, y_train)
            
            # Calcular residuos en TRAIN
            y_train_pred = model.predict(X_train)
            residuos_train = y_train - y_train_pred
            
            q10 = np.percentile(residuos_train, 10)
            q50 = np.percentile(residuos_train, 50)
            q90 = np.percentile(residuos_train, 90)
            
            # Evaluar en TEST
            y_test_pred = model.predict(X_test)
            mae_test = mean_absolute_error(y_test, y_test_pred)
            
            # =============================================================
            # VALIDACIÓN DE 3 NIVELES
            # =============================================================
            validacion = validar_modelo_completo(y_test, y_test_pred, input_splits_str, futuro)
            
            # Log de validación
            if validacion['aprobado']:
                logger.info(f"   ✅ VALIDACIÓN APROBADA - Calidad: {validacion['puntuacion_calidad']}/100")
                logger.info(f"      Mejora sobre naïve: {validacion['niveles']['mejor_que_naive']['mejora']:.1%}")
                logger.info(f"      CV error: {validacion['niveles']['consistencia_error']['cv']:.2f}")
                logger.info(f"      Ratio P95/P50: {validacion['niveles']['sin_outliers']['relacion']:.1f}")
            else:
                logger.warning(f"   ❌ VALIDACIÓN RECHAZADA - Calidad: {validacion['puntuacion_calidad']}/100")
                if not validacion['niveles']['mejor_que_naive']['aprueba']:
                    logger.warning(f"      - No mejora a naïve")
                if not validacion['niveles']['consistencia_error']['aprueba']:
                    logger.warning(f"      - Error inconsistente (CV={validacion['niveles']['consistencia_error']['cv']:.2f} > 0.5)")
                if not validacion['niveles']['sin_outliers']['aprueba']:
                    logger.warning(f"      - Outliers catastróficos (ratio={validacion['niveles']['sin_outliers']['relacion']:.1f} > 3)")
            
            # Verificar cobertura real en test
            test_inferior = y_test_pred + q10
            test_superior = y_test_pred + q90
            cobertura = np.mean((y_test >= test_inferior) & (y_test <= test_superior)) * 100
            
            logger.info(f"   ✅ MAE en test: {mae_test:.2f} seg")
            logger.info(f"   📊 Cobertura real del intervalo 80% en test: {cobertura:.1f}%")
            
            # =============================================================
            # GUARDAR MODELOS Y REGISTRAR (SOLO SI APROBADOS)
            # =============================================================
            
            for q_val, q_label in [(q10, '10'), (q50, '50'), (q90, '90')]:
                modelo_id = f"pred_{input_splits_str}_to_{futuro.replace('km_','')}_q{q_label}"
                filename = f"{modelo_id}.joblib"
                model_path = os.path.join(model_dir, filename)
                
                model_data = {
                    'modelo': model,
                    'features': features,
                    'input_splits': input_splits,
                    'output_split': futuro,
                    'cuantiles': {'q10': q10, 'q50': q50, 'q90': q90},
                    'tipo': f'q{q_label}',
                    'mejor_mae': mae_test,
                    'cobertura_test': cobertura
                }
                joblib.dump(model_data, model_path)
                
                # Solo registrar si el modelo base pasó validación
                if validacion['aprobado']:
                    # Subir métricas a S3
                    metricas_modelo = {
                        'mae_test': mae_test,
                        'cobertura_test': cobertura,
                        'n_samples_train': len(X_train),
                        'n_samples_test': len(X_test),
                        'input_splits': input_splits_str,
                        'output_split': futuro
                    }
                    
                    metrics_path = guardar_metricas_prediccion_json(metricas_modelo, validacion, output_dir, modelo_id)
                    s3_metrics_key = f"entrenamientos/prediccion/{carrera.replace(' ', '_')}/{timestamp_unico}/metrics/{modelo_id}_metrics.json"
                    try:
                        s3_client.upload_file(metrics_path, S3_BUCKET_MODELOS, s3_metrics_key)
                    except Exception as e:
                        logger.warning(f"   ⚠️ Error subiendo métricas: {e}")
                    
                    # Construir URI S3
                    s3_model_uri = f"s3://{S3_BUCKET_MODELOS}/entrenamientos/prediccion/{carrera.replace(' ', '_')}/{timestamp_unico}/models/{filename}"
                    
                    # Registrar en SageMaker
                    registry_response = registrar_modelo_prediccion(
                        modelo_id=modelo_id,
                        metricas=metricas_modelo,
                        model_s3_uri=s3_model_uri,
                        carrera=carrera,
                        timestamp_unico=timestamp_unico,
                        validacion=validacion,
                        quantile=q_label
                    )
                    
                    modelos_guardados.append({
                        'modelo_id': modelo_id,
                        'registry_arn': registry_response.get('ModelPackageArn') if registry_response else None
                    })
                    
                    logger.info(f"   💾 Modelo guardado: {modelo_id}")
            
            # Estadísticas de validación
            if validacion['aprobado']:
                modelos_aprobados += 1
            else:
                modelos_rechazados += 1
            
            validaciones_totales.append(validacion)
            
            # Guardar métricas
            resultados.append({
                'input_splits': input_splits_str,
                'output_split': futuro,
                'mae_test': mae_test,
                'cobertura_test': cobertura,
                'q10_train': q10,
                'q50_train': q50,
                'q90_train': q90,
                'n_samples_train': len(X_train),
                'n_samples_test': len(X_test),
                'n_atletas_train': len(train_athletes),
                'n_atletas_test': len(test_athletes),
                'validacion_aprobado': validacion['aprobado'],
                'calidad': validacion['puntuacion_calidad']
            })
    
    # Mostrar resumen de validación
    if validaciones_totales:
        aprobados = sum(1 for v in validaciones_totales if v['aprobado'])
        calidad_promedio = np.mean([v['puntuacion_calidad'] for v in validaciones_totales])
        logger.info(f"\n📊 RESUMEN DE VALIDACIÓN:")
        logger.info(f"   Modelos evaluados: {len(validaciones_totales)}")
        logger.info(f"   Modelos aprobados: {aprobados}")
        logger.info(f"   Modelos rechazados: {len(validaciones_totales) - aprobados}")
        logger.info(f"   Tasa de aprobación: {aprobados/len(validaciones_totales):.1%}")
        logger.info(f"   Calidad promedio: {calidad_promedio:.0f}/100")
    
    return pd.DataFrame(resultados), modelos_guardados, validaciones_totales


# =============================================================
# MAIN
# =============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--carrera', type=str, required=True)
    parser.add_argument('--tipo_modelo', type=str, default='prediccion')
    parser.add_argument('--n-estimators', type=int, default=200)
    parser.add_argument('--max-depth', type=int, default=10)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--min-samples', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample-bytree', type=float, default=0.8)
    
    args, unknown = parser.parse_known_args()
    
    # Generar timestamp único
    timestamp_unico = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    logger.info("="*80)
    logger.info("🚀 ENTRENAMIENTO DE MODELOS DE PREDICCIÓN CON INTERVALOS")
    logger.info("="*80)
    logger.info(f"📌 Carrera: {args.carrera}")
    logger.info(f"📌 Tipo: {args.tipo_modelo}")
    logger.info(f"🕒 Timestamp: {timestamp_unico}")
    
    try:
        # =============================================================
        # CARGAR DATOS
        # =============================================================
        model_dir, output_dir, train_dir = get_paths()
        df = load_data_from_local(train_dir)
        
        # Crear grupo en Model Registry
        crear_model_package_group(args.carrera)
        
        # Renombrar columnas
        for col in df.columns:
            if '.' in col:
                nuevo_nombre = col.replace('.', '_')
                df.rename(columns={col: nuevo_nombre}, inplace=True)
            if '-' in col:
                nuevo_nombre = col.replace('-', '_')
                df.rename(columns={col: nuevo_nombre}, inplace=True)
        
        # Preprocesamiento
        df = procesar_genero(df)
        
        # Identificar y ordenar splits
        split_cols = identificar_splits(df)
        if not split_cols:
            raise ValueError("No se identificaron splits en los datos")
            
        split_cols_ordenados = ordenar_splits_por_distancia(split_cols)
        logger.info(f"📊 Splits ordenados ({len(split_cols_ordenados)}): {split_cols_ordenados}")
        
        # =============================================================
        # PARÁMETROS DE ENTRENAMIENTO
        # =============================================================
        params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'subsample': args.subsample,
            'colsample_bytree': args.colsample_bytree,
            'n_folds': args.n_folds,
            'min_samples': args.min_samples
        }
        
        quantiles = [0.1, 0.5, 0.9]
        
        # =============================================================
        # ENTRENAR MODELOS CON VALIDACIÓN
        # =============================================================
        metricas_df, modelos_guardados, validaciones_totales = entrenar_modelos_quantiles(
            df, split_cols_ordenados, args.carrera, quantiles, params, 
            model_dir, output_dir, timestamp_unico
        )
        
        # =============================================================
        # GUARDAR METADATA
        # =============================================================
        if not metricas_df.empty:
            metricas_path = os.path.join(output_dir, 'metricas_prediccion.csv')
            metricas_df.to_csv(metricas_path, index=False)
            logger.info(f"✅ Métricas guardadas en: {metricas_path}")
            
            # Estadísticas de validación
            if validaciones_totales:
                aprobados = sum(1 for v in validaciones_totales if v['aprobado'])
                calidad_promedio = np.mean([v['puntuacion_calidad'] for v in validaciones_totales])
                
                estadisticas_validacion = {
                    'total_evaluados': len(validaciones_totales),
                    'aprobados': aprobados,
                    'rechazados': len(validaciones_totales) - aprobados,
                    'tasa_aprobacion': aprobados / len(validaciones_totales),
                    'calidad_promedio': float(calidad_promedio),
                    'mejora_promedio': float(np.mean([v['niveles']['mejor_que_naive']['mejora'] for v in validaciones_totales])),
                    'cv_promedio': float(np.mean([v['niveles']['consistencia_error']['cv'] for v in validaciones_totales]))
                }
            else:
                estadisticas_validacion = {}
        
        total_esperado = (len(split_cols_ordenados) - 1) * len(split_cols_ordenados) // 2
        total_entrenado = len(metricas_df) if not metricas_df.empty else 0
        
        metadata = {
            'carrera': args.carrera,
            'tipo_modelo': args.tipo_modelo,
            'timestamp': timestamp_unico,
            'fecha_entrenamiento': datetime.now().isoformat(),
            'quantiles': quantiles,
            'splits': split_cols_ordenados,
            'total_combinaciones_esperadas': total_esperado,
            'combinaciones_entrenadas': total_entrenado,
            'modelos_totales': total_entrenado * len(quantiles),
            'hiperparametros': params,
            'bucket_modelos': S3_BUCKET_MODELOS,
            'validaciones': estadisticas_validacion,
            'modelos_registrados': len(modelos_guardados)
        }
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Guardar resumen de validaciones
        if validaciones_totales:
            resumen_validaciones = {
                'carrera': args.carrera,
                'timestamp': timestamp_unico,
                'tipo': 'prediccion',
                'total_evaluados': len(validaciones_totales),
                'aprobados': aprobados,
                'rechazados': len(validaciones_totales) - aprobados,
                'tasa_aprobacion': aprobados / len(validaciones_totales),
                'calidad_promedio': float(calidad_promedio),
                'mejores_modelos': sorted(
                    [{'id': f"{r['input_splits']}_to_{r['output_split']}", 
                      'mae': r['mae_test'], 
                      'calidad': r['calidad']} 
                     for r in metricas_df.to_dict('records')],
                    key=lambda x: x['calidad'],
                    reverse=True
                )[:5]
            }
            resumen_path = os.path.join(output_dir, 'resumen_validaciones.json')
            with open(resumen_path, 'w') as f:
                json.dump(resumen_validaciones, f, indent=2, default=str)
        
        # =============================================================
        # RESUMEN FINAL
        # =============================================================
        logger.info("\n" + "="*80)
        logger.info("📊 RESUMEN FINAL")
        logger.info("="*80)
        logger.info(f"🎯 Carrera: {args.carrera}")
        logger.info(f"📊 Splits: {len(split_cols_ordenados)}")
        logger.info(f"📊 Combinaciones posibles: {total_esperado}")
        logger.info(f"📊 Combinaciones entrenadas: {total_entrenado}")
        logger.info(f"📊 Modelos guardados: {total_entrenado * len(quantiles)}")
        
        if estadisticas_validacion:
            logger.info(f"\n📈 ESTADÍSTICAS DE VALIDACIÓN:")
            logger.info(f"   Tasa de aprobación: {estadisticas_validacion['tasa_aprobacion']:.1%}")
            logger.info(f"   Calidad promedio: {estadisticas_validacion['calidad_promedio']:.0f}/100")
            logger.info(f"   Mejora promedio sobre naïve: {estadisticas_validacion['mejora_promedio']:.1%}")
            logger.info(f"   CV promedio: {estadisticas_validacion['cv_promedio']:.2f}")
        
        if not metricas_df.empty:
            logger.info("\n🏆 TOP 5 MEJORES MODELOS (por MAE):")
            top5 = metricas_df.nsmallest(5, 'mae_test')
            for _, row in top5.iterrows():
                calidad = row.get('calidad', 0)
                logger.info(f"   {row['input_splits']} → {row['output_split']}: "
                           f"MAE={row['mae_test']:.2f}s, "
                           f"Cobertura={row['cobertura_test']:.1f}%, "
                           f"Calidad={calidad}/100")
        
        logger.info("\n" + "="*80)
        logger.info("✅ ENTRENAMIENTO COMPLETADO")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error en entrenamiento: {str(e)}")
        raise


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