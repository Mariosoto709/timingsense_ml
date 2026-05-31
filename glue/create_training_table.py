"""
Job principal de Glue - Orquesta la creación de tablas de entrenamiento
VERSIÓN MEJORADA CON:
- Checkpoint para recuperación parcial
- Reintentos con backoff
- Cache con hash para evitar reprocesar
- Escritura segura con verificación
"""

import json
import sys
import hashlib
import re
import time
import os
from datetime import datetime
from urllib.parse import urlparse

from awsglue.utils import getResolvedOptions
import boto3
import pandas as pd

# Clientes AWS
s3 = boto3.client('s3')
athena = boto3.client('athena')

# Constantes
S3_ATHENA_OUTPUT = 's3://timingsense-training-data'
DATABASE = 'timingsense_training'
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]  # segundos


# ============================================================
# FUNCIONES DE REINTENTO
# ============================================================

def ejecutar_con_reintento(func, operation_name, max_retries=MAX_RETRIES):
    """Ejecuta una función con reintentos y backoff exponencial"""
    for i in range(max_retries):
        try:
            print(f"   🔄 Ejecutando {operation_name} (intento {i+1}/{max_retries})...")
            return func()
        except Exception as e:
            if i == max_retries - 1:
                print(f"   ❌ {operation_name} falló después de {max_retries} intentos")
                raise
            wait_time = RETRY_BACKOFF[i] if i < len(RETRY_BACKOFF) else 2 ** i
            print(f"   ⚠️ {operation_name} falló: {e}. Reintentando en {wait_time}s...")
            time.sleep(wait_time)


# ============================================================
# FUNCIONES DE S3 (CON REINTENTO)
# ============================================================

def path_exists_s3(path):
    """Verifica si una ruta S3 existe"""
    try:
        parsed = urlparse(path)
        response = s3.list_objects_v2(
            Bucket=parsed.netloc,
            Prefix=parsed.path.lstrip('/'),
            MaxKeys=1
        )
        return response.get('KeyCount', 0) > 0
    except Exception as e:
        return False


def copiar_path_s3(origen, destino):
    """Copia archivos/directorios en S3"""
    src_parsed = urlparse(origen)
    dst_parsed = urlparse(destino)
    
    prefix = src_parsed.path.lstrip('/')
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    
    continuation_token = None
    copied_count = 0
    
    while True:
        params = {'Bucket': src_parsed.netloc, 'Prefix': prefix, 'MaxKeys': 1000}
        if continuation_token:
            params['ContinuationToken'] = continuation_token
        response = s3.list_objects_v2(**params)
        
        for obj in response.get('Contents', []):
            src_key = obj['Key']
            dst_key = src_key.replace(prefix, dst_parsed.path.lstrip('/'))
            if dst_key.startswith('/'):
                dst_key = dst_key[1:]
            s3.copy_object(
                Bucket=dst_parsed.netloc,
                CopySource={'Bucket': src_parsed.netloc, 'Key': src_key},
                Key=dst_key
            )
            copied_count += 1
        
        if not response.get('IsTruncated'):
            break
        continuation_token = response.get('NextContinuationToken')
    
    print(f"   ✅ Copiados {copied_count} objetos")
    return True


def eliminar_path_s3(path):
    """Elimina una ruta S3"""
    parsed = urlparse(path)
    prefix = parsed.path.lstrip('/')
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    
    continuation_token = None
    while True:
        params = {'Bucket': parsed.netloc, 'Prefix': prefix, 'MaxKeys': 1000}
        if continuation_token:
            params['ContinuationToken'] = continuation_token
        response = s3.list_objects_v2(**params)
        
        for obj in response.get('Contents', []):
            s3.delete_object(Bucket=parsed.netloc, Key=obj['Key'])
        
        if not response.get('IsTruncated'):
            break
        continuation_token = response.get('NextContinuationToken')
    
    print(f"   🧹 Eliminado: {path}")


def guardar_dataset_seguro(df, output_path, spark):
    """
    Guarda un DataFrame con escritura segura:
    1. Temporal
    2. Verificación
    3. Backup de versión anterior
    4. Copia a destino
    5. Verificación final
    """
    temp_path = f"{output_path}.temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_path = f"{output_path}.backup"
    
    print(f"   📝 Escribiendo temporal: {temp_path}")
    
    # 1. Escribir temporal
    df.write.mode("overwrite").parquet(temp_path)
    
    # 2. Verificar temporal
    print(f"   🔍 Verificando temporal...")
    df_temp = spark.read.parquet(temp_path)
    temp_count = df_temp.count()
    
    if temp_count == 0:
        raise Exception("Temporal vacío")
    
    print(f"   ✅ Temporal válido: {temp_count:,} registros")
    
    # 3. Backup si existe
    if path_exists_s3(output_path):
        print(f"   📦 Creando backup: {backup_path}")
        if path_exists_s3(backup_path):
            eliminar_path_s3(backup_path)
        copiar_path_s3(output_path, backup_path)
    
    # 4. Copiar a destino
    print(f"   📋 Copiando a destino...")
    if path_exists_s3(output_path):
        eliminar_path_s3(output_path)
    copiar_path_s3(temp_path, output_path)
    
    # 5. Verificar destino
    print(f"   🔍 Verificando destino...")
    df_final = spark.read.parquet(output_path)
    final_count = df_final.count()
    
    if final_count != temp_count:
        # Rollback
        print(f"   ❌ Conteo不一致! Restaurando backup...")
        if path_exists_s3(backup_path):
            eliminar_path_s3(output_path)
            copiar_path_s3(backup_path, output_path)
        raise Exception(f"Verificación falló: temp={temp_count}, final={final_count}")
    
    print(f"   ✅ Destino verificado: {final_count:,} registros")
    
    # 6. Limpiar temporal
    eliminar_path_s3(temp_path)
    
    return True


# ============================================================
# FUNCIONES DE ATHENA (CON REINTENTO)
# ============================================================

def execute_athena_query(query, database=DATABASE):
    """Ejecuta una query en Athena y espera resultado"""
    def _execute():
        response = athena.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': database},
            ResultConfiguration={'OutputLocation': S3_ATHENA_OUTPUT}
        )
        
        execution_id = response['QueryExecutionId']
        
        # Esperar a que termine
        while True:
            status = athena.get_query_execution(QueryExecutionId=execution_id)
            state = status['QueryExecution']['Status']['State']
            
            if state == 'SUCCEEDED':
                # Para CTAS, obtener ubicación
                if 'CREATE TABLE' in query.upper():
                    result = athena.get_query_results(QueryExecutionId=execution_id)
                    # La ubicación está en los resultados o metadata
                    return S3_ATHENA_OUTPUT
                return True
            elif state == 'FAILED':
                reason = status['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                raise Exception(f"Query falló: {reason}")
            elif state == 'CANCELLED':
                raise Exception("Query cancelada")
            
            time.sleep(2)
    
    return ejecutar_con_reintento(_execute, f"Athena query")


def tabla_existe(table_name, database=DATABASE):
    """Verifica si una tabla existe en Glue"""
    def _execute():
        try:
            athena.get_table_metadata(
                CatalogName='awsglue',
                DatabaseName=database,
                TableName=table_name
            )
            return True
        except Exception:
            return False
    
    return ejecutar_con_reintento(_execute, f"Verificar tabla {table_name}")


def create_glue_table(table_name, columns, database=DATABASE, partitioned=False):
    """Crea una tabla en Glue"""
    def _execute():
        # Construir DDL
        cols_sql = ",\n    ".join(columns)
        partition_sql = "PARTITIONED BY (race_id string, event_id string)" if partitioned else ""
        
        create_query = f"""
        CREATE EXTERNAL TABLE IF NOT EXISTS {database}.{table_name} (
            {cols_sql}
        )
        {partition_sql}
        STORED AS PARQUET
        LOCATION 's3://{S3_ATHENA_OUTPUT}/{database}/{table_name}/'
        """
        
        execute_athena_query(create_query)
        return True
    
    return ejecutar_con_reintento(_execute, f"Crear tabla {table_name}")


def drop_table(table_name, database=DATABASE):
    """Elimina una tabla"""
    def _execute():
        drop_query = f"DROP TABLE IF EXISTS {database}.{table_name}"
        execute_athena_query(drop_query)
        return True
    
    return ejecutar_con_reintento(_execute, f"Eliminar tabla {table_name}")


# ============================================================
# FUNCIONES DE CHECKPOINT
# ============================================================

class TrainingCheckpoint:
    """Maneja checkpoints para recuperación parcial"""
    
    def __init__(self, output_base, ejecucion_id):
        self.output_base = output_base.rstrip('/')
        self.ejecucion_id = ejecucion_id
        self.checkpoint_path = f"{self.output_base}/checkpoints/{ejecucion_id}.json"
        self.data = self._cargar()
    
    def _cargar(self):
        """Carga checkpoint existente"""
        default_data = {
            "ejecucion_id": self.ejecucion_id,
            "timestamp_start": datetime.now().isoformat(),
            "timestamp_last_update": None,
            "timestamp_end": None,
            "status": "RUNNING",
            "total_carreras": 0,
            "carreras_procesadas": [],
            "carreras_fallidas": [],
            "carreras_omitidas": []
        }
        
        try:
            if path_exists_s3(self.checkpoint_path):
                parsed = urlparse(self.checkpoint_path)
                response = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip('/'))
                loaded = json.loads(response['Body'].read().decode('utf-8'))
                if loaded.get("ejecucion_id") == self.ejecucion_id:
                    print(f"📂 Checkpoint recuperado: {len(loaded.get('carreras_procesadas', []))} carreras ya procesadas")
                    return loaded
        except Exception as e:
            print(f"⚠️ Error cargando checkpoint: {e}")
        
        print("📂 No se encontró checkpoint previo, comenzando desde cero")
        return default_data
    
    def init_carreras(self, carreras):
        """Inicializa las carreras a procesar"""
        self.data["total_carreras"] = len(carreras)
        self.data["carreras_pendientes"] = [
            c for c in carreras 
            if c not in self.data["carreras_procesadas"] 
            and c not in self.data["carreras_fallidas"]
        ]
        self._guardar()
    
    def marcar_procesada(self, carrera, resultado=None):
        """Marca una carrera como procesada exitosamente"""
        if carrera not in self.data["carreras_procesadas"]:
            self.data["carreras_procesadas"].append(carrera)
        if carrera in self.data.get("carreras_pendientes", []):
            self.data["carreras_pendientes"].remove(carrera)
        if resultado:
            self.data[f"resultado_{carrera}"] = resultado
        self._guardar()
    
    def marcar_fallida(self, carrera, error):
        """Marca una carrera como fallida"""
        if carrera not in [f['carrera'] for f in self.data["carreras_fallidas"]]:
            self.data["carreras_fallidas"].append({
                "carrera": carrera,
                "error": str(error)[:200],
                "timestamp": datetime.now().isoformat()
            })
        if carrera in self.data.get("carreras_pendientes", []):
            self.data["carreras_pendientes"].remove(carrera)
        self._guardar()
    
    def marcar_omitida(self, carrera, razon):
        """Marca una carrera como omitida (cache hit)"""
        if carrera not in [f['carrera'] for f in self.data["carreras_omitidas"]]:
            self.data["carreras_omitidas"].append({
                "carrera": carrera,
                "razon": razon,
                "timestamp": datetime.now().isoformat()
            })
        if carrera in self.data.get("carreras_pendientes", []):
            self.data["carreras_pendientes"].remove(carrera)
        self._guardar()
    
    def get_pendientes(self):
        """Retorna lista de carreras pendientes"""
        return self.data.get("carreras_pendientes", [])
    
    def _guardar(self):
        """Guarda checkpoint en S3"""
        self.data["timestamp_last_update"] = datetime.now().isoformat()
        
        try:
            parsed = urlparse(self.checkpoint_path)
            s3.put_object(
                Bucket=parsed.netloc,
                Key=parsed.path.lstrip('/'),
                Body=json.dumps(self.data, indent=2, default=str),
                ContentType='application/json'
            )
        except Exception as e:
            print(f"⚠️ Error guardando checkpoint: {e}")
    
    def finish(self, status):
        """Finaliza checkpoint"""
        self.data["status"] = status
        self.data["timestamp_end"] = datetime.now().isoformat()
        self._guardar()
    
    def cleanup(self):
        """Limpia checkpoint después de ejecución exitosa"""
        try:
            if path_exists_s3(self.checkpoint_path):
                eliminar_path_s3(self.checkpoint_path)
            print("🧹 Checkpoint limpiado")
        except Exception as e:
            print(f"⚠️ Error limpiando checkpoint: {e}")


# ============================================================
# FUNCIONES DE CACHE POR HASH
# ============================================================

def get_dataset_hash(carreras_historicas, splits_objetivo, evento_objetivo):
    """Calcula hash único para un conjunto de carreras y splits"""
    data = {
        "carreras": sorted([c['race_id'] for c in carreras_historicas]),
        "eventos_historicos": sorted([c.get('evento', 'unknown') for c in carreras_historicas]),
        "evento_objetivo": evento_objetivo,
        "splits": sorted(splits_objetivo)
    }
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


def dataset_ya_existe(hash_dataset, output_base):
    """Verifica si ya existe un dataset con ese hash"""
    dataset_path = f"{output_base}/datasets/{hash_dataset}/"
    metadata_path = f"{dataset_path}metadata.json"
    
    if path_exists_s3(metadata_path):
        print(f"   💾 Dataset con hash {hash_dataset[:8]} ya existe, reutilizando")
        return dataset_path
    return None


def guardar_metadata_dataset(hash_dataset, output_base, metadata):
    """Guarda metadata del dataset"""
    dataset_path = f"{output_base}/datasets/{hash_dataset}/"
    metadata_path = f"{dataset_path}metadata.json"
    
    parsed = urlparse(metadata_path)
    s3.put_object(
        Bucket=parsed.netloc,
        Key=parsed.path.lstrip('/'),
        Body=json.dumps(metadata, indent=2, default=str),
        ContentType='application/json'
    )
    print(f"   📝 Metadata guardada en: {metadata_path}")
    
    return dataset_path


# ============================================================
# FUNCIONES PRINCIPALES
# ============================================================

def analyze_split_requirements(splits_originales, carreras_historicas):
    """
    Analiza qué splits son directos y cuáles necesitan interpolación.
    """
    splits_directos = []
    splits_interpolables = []
    splits_imposibles = []
    mapping = {}
    
    # Obtener todos los splits disponibles en carreras históricas
    splits_disponibles = set()
    for c in carreras_historicas:
        splits_disponibles.update(c.get('splits', []))
    
    # Normalizar splits originales
    splits_normalizados = [s.replace('.', '_') for s in splits_originales]
    
    for split_original, split_norm in zip(splits_originales, splits_normalizados):
        if split_norm in splits_disponibles:
            splits_directos.append(split_original)
            mapping[split_original] = {'tipo': 'directo', 'split_origen': split_norm}
        else:
            # Intentar encontrar split cercano para interpolación
            # (lógica simplificada, se puede expandir)
            splits_interpolables.append({
                'split_objetivo': split_original,
                'split_anterior': None,
                'split_posterior': None
            })
            mapping[split_original] = {'tipo': 'interpolable', 'disponible': False}
    
    return {
        'splits_directos': splits_directos,
        'splits_interpolables': splits_interpolables,
        'splits_imposibles': splits_imposibles,
        'splits_finales': splits_normalizados,
        'mapping': mapping
    }


def procesar_una_carrera(config, timestamp_unico, output_base, spark, checkpoint=None):
    """
    Procesa una carrera individual.
    Ahora con:
    - Cache por hash
    - Checkpoint
    - Escritura segura
    """
    carrera_objetivo = config["carrera_objetivo"]
    evento_objetivo = config.get("evento_objetivo")  # ← AÑADIR
    splits = config["splits"]
    carreras_historicas_detalle = config.get('carreras_historicas_detalle', [])
    tipo_seleccion = config.get('tipo_seleccion', 'desconocido')
    cobertura_total = config.get('cobertura_total', 0)
    
    print(f"\n{'='*60}")
    print(f"🏁 Procesando: {carrera_objetivo} / {evento_objetivo}")  # ← MODIFICAR
    print(f"{'='*60}")
    
    # Verificar checkpoint
    if checkpoint and carrera_objetivo in checkpoint.data.get("carreras_procesadas", []):
        print(f"   💾 Carrera ya procesada según checkpoint, omitiendo")
        return checkpoint.data.get(f"resultado_{carrera_objetivo}")
    
    # ============================================================
    # CACHE POR HASH
    # ============================================================
    hash_dataset = get_dataset_hash(carreras_historicas_detalle, splits, evento_objetivo)
    dataset_existente = dataset_ya_existe(hash_dataset, output_base)
    
    if dataset_existente:
        print(f"   💾 Dataset ya existe (hash: {hash_dataset[:8]}), reutilizando")
        
        # Cargar metadata existente
        metadata_path = f"{dataset_existente}metadata.json"
        parsed = urlparse(metadata_path)
        response = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip('/'))
        metadata = json.loads(response['Body'].read().decode('utf-8'))
        
        resultado = {
            "carrera": carrera_objetivo,
            "carpeta_modelo": metadata.get('carpeta_modelo'),
            "data_s3_path": metadata.get('data_s3_path'),
            "tabla_generada": metadata.get('tabla_generada'),
            "splits": metadata.get('splits', 0),
            "splits_originales": metadata.get('splits_originales', 0),
            "splits_interpolados": metadata.get('splits_interpolados', 0),
            "splits_imposibles": metadata.get('splits_imposibles', 0),
            "carreras_usadas": metadata.get('carreras_usadas', 0),
            "tipo_seleccion": metadata.get('tipo_seleccion', tipo_seleccion),
            "cobertura_total": metadata.get('cobertura_total', cobertura_total),
            "cache_hit": True
        }
        
        if checkpoint:
            checkpoint.marcar_procesada(carrera_objetivo, resultado)
        
        return resultado
    
    # ============================================================
    # TRANSFORMAR CARRERAS HISTÓRICAS AL FORMATO ESPERADO
    # ============================================================
    carreras_historicas = []
    for h in carreras_historicas_detalle:
        # Obtener splits reales de la carrera (desde S3)
        try:
            race_id = h['race_id']
            splits_path = f"s3://timingsense-races-processed-wide/current/wide/race_id={race_id}/metadata.json"
            parsed = urlparse(splits_path)
            response = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip('/'))
            metadata_carrera = json.loads(response['Body'].read().decode('utf-8'))
            
            splits_carrera = [s['nombre'] for s in metadata_carrera.get('splits', [])]
            
            carreras_historicas.append({
                'race_id': race_id,
                'event_id': h.get('evento', 'unknown'),
                'splits': splits_carrera
            })
        except Exception as e:
            print(f"   ⚠️ Error cargando splits de {h['race_id']}: {e}")
            carreras_historicas.append({
                'race_id': h['race_id'],
                'event_id': h.get('evento', 'unknown'),
                'splits': h.get('splits', [])
            })
    
    print(f"   Carreras históricas a usar: {len(carreras_historicas)}")
    
    # ============================================================
    # ANALIZAR REQUISITOS DE SPLITS
    # ============================================================
    analisis = analyze_split_requirements(splits, carreras_historicas)
    
    print(f"   Splits directos: {analisis['splits_directos']}")
    print(f"   Splits interpolables: {len(analisis['splits_interpolables'])}")
    
    # ============================================================
    # CREAR CARPETA Y GUARDAR DATOS
    # ============================================================
    carpeta_modelo = f"{carrera_objetivo}-{timestamp_unico}"
    data_s3_path = f"{output_base}/modelos/{carpeta_modelo}/data/"
    
    print(f"   📁 Carpeta modelo: {carpeta_modelo}")
    
    # Construir query SELECT
    select_cols = ["athlete_id", "event_id", "event_std", "gender", "age"]
    
    for split_final in analisis['splits_finales']:
        select_cols.append(f'"{split_final}" as "{split_final}"')
    
    select_clause = ",\n        ".join(select_cols)
    
    # Construir WHERE clause
    condiciones = [f"(race_id = '{c['race_id']}' AND event_id = '{c['event_id']}')" 
                   for c in carreras_historicas]
    where_clause = " OR ".join(condiciones)
    
    # Tabla fuente (wide data)
    tabla_fuente = "wide_data"  # Asumiendo que existe una vista/tabla unificada
    
    # ============================================================
    # GENERAR DATASET (usando Spark para escritura segura)
    # ============================================================
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder.getOrCreate()
    
    # Leer datos de las carreras históricas
    dfs = []
    for c in carreras_historicas:
        path = f"s3://timingsense-races-processed-wide/current/wide/race_id={c['race_id']}/"
        try:
            df = spark.read.parquet(path)
            # Filtrar por event_id si es necesario
            if c['event_id'] and c['event_id'] != 'unknown':
                df = df.filter(f"event_id = '{c['event_id']}'")
            dfs.append(df)
            print(f"   📖 Leída {c['race_id']}: {df.count()} registros")
        except Exception as e:
            print(f"   ⚠️ Error leyendo {c['race_id']}: {e}")
    
    if not dfs:
        raise Exception("No se pudieron leer datos de ninguna carrera histórica")
    
    # Unir todos los DataFrames
    df_final = dfs[0]
    for df in dfs[1:]:
        df_final = df_final.unionByName(df, allowMissingColumns=True)
    
    # Seleccionar solo las columnas necesarias
    cols_disponibles = [c for c in select_cols if c in df_final.columns]
    df_final = df_final.select(*cols_disponibles)
    
    # Validar que hay datos
    total_registros = df_final.count()
    if total_registros < 100:
        raise Exception(f"Datos insuficientes: {total_registros} filas")
    
    print(f"   📊 Total registros: {total_registros:,}")
    
    # ============================================================
    # ESCRIBIR CON SEGURIDAD
    # ============================================================
    guardar_dataset_seguro(df_final, data_s3_path, spark)
    
    # ============================================================
    # VALIDACIÓN POST-ESCRITURA
    # ============================================================
    print(f"   🔍 Validando dataset...")
    df_validacion = spark.read.parquet(data_s3_path)
    count_validacion = df_validacion.count()
    
    if count_validacion != total_registros:
        raise Exception(f"Validación falló: esperado {total_registros}, obtenido {count_validacion}")
    
    # Verificar nulos
    null_counts = {}
    for col in df_validacion.columns:
        null_count = df_validacion.filter(f"{col} IS NULL").count()
        if null_count > 0:
            null_counts[col] = null_count
    
    if null_counts:
        print(f"   ⚠️ Columnas con nulos: {null_counts}")
    
    print(f"   ✅ Dataset validado: {count_validacion:,} registros")
    
    # ============================================================
    # GUARDAR METADATA
    # ============================================================
    metadata = {
        "carrera": carrera_objetivo,
        "timestamp": timestamp_unico,
        "carpeta_modelo": carpeta_modelo,
        "data_s3_path": data_s3_path,
        "hash_dataset": hash_dataset,
        "splits_originales": splits,
        "splits_finales": analisis['splits_finales'],
        "splits_directos": analisis['splits_directos'],
        "splits_interpolados": len(analisis['splits_interpolables']),
        "splits_imposibles": len(analisis['splits_imposibles']),
        "carreras_usadas": len(carreras_historicas),
        "carreras_detalle": [{"race_id": c['race_id'], "event_id": c['event_id']} for c in carreras_historicas],
        "tipo_seleccion": tipo_seleccion,
        "cobertura_total": cobertura_total,
        "total_registros": total_registros,
        "columnas": df_final.columns
    }
    
    # Guardar metadata
    metadata_s3_path = f"{output_base}/datasets/{hash_dataset}/metadata.json"
    parsed = urlparse(metadata_s3_path)
    s3.put_object(
        Bucket=parsed.netloc,
        Key=parsed.path.lstrip('/'),
        Body=json.dumps(metadata, indent=2, default=str),
        ContentType='application/json'
    )
    
    print(f"   📝 Metadata guardada")
    
    resultado = {
        "carrera": carrera_objetivo,
        "carpeta_modelo": carpeta_modelo,
        "data_s3_path": data_s3_path,
        "tabla_generada": None,
        "splits": len(analisis['splits_finales']),
        "splits_originales": len(splits),
        "splits_interpolados": len(analisis['splits_interpolables']),
        "splits_imposibles": len(analisis['splits_imposibles']),
        "carreras_usadas": len(carreras_historicas),
        "tipo_seleccion": tipo_seleccion,
        "cobertura_total": cobertura_total,
        "total_registros": total_registros,
        "hash_dataset": hash_dataset,
        "cache_hit": False
    }
    
    return resultado


# ============================================================
# MAIN GLUE ENTRYPOINT
# ============================================================

def main():
    print("=" * 80)
    print("🚀 JOB GLUE - CREAR TABLAS DE ENTRENAMIENTO (VERSIÓN MEJORADA)")
    print("=" * 80)
    
    # Leer parámetros
    try:
        args = getResolvedOptions(sys.argv, ["carreras_json", "timestamp_unico", "output_base"])
        carreras_json = args["carreras_json"]
        timestamp_unico = args["timestamp_unico"]
        output_base = args["output_base"]
    except:
        try:
            args = getResolvedOptions(sys.argv, ["carreras_json", "timestamp_unico"])
            carreras_json = args["carreras_json"]
            timestamp_unico = args["timestamp_unico"]
            output_base = S3_ATHENA_OUTPUT
        except:
            args = getResolvedOptions(sys.argv, ["carreras_json"])
            carreras_json = args["carreras_json"]
            timestamp_unico = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_base = S3_ATHENA_OUTPUT
    
    carreras_config = json.loads(carreras_json)
    
    print(f"📥 Carreras a procesar: {len(carreras_config)}")
    print(f"🕒 Timestamp único: {timestamp_unico}")
    print(f"📂 Output base: {output_base}")
    
    # Inicializar Spark
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    
    # Inicializar Checkpoint
    ejecucion_id = f"training_{timestamp_unico}"
    checkpoint = TrainingCheckpoint(output_base, ejecucion_id)
    
    # Obtener lista de carreras a procesar
    todas_carreras = [c.get('carrera_objetivo') for c in carreras_config]
    checkpoint.init_carreras(todas_carreras)
    carreras_pendientes = checkpoint.get_pendientes()
    
    print(f"\n📊 Carreras pendientes: {len(carreras_pendientes)}")
    
    if not carreras_pendientes:
        print("✅ No hay carreras pendientes")
        checkpoint.finish("SUCCESS")
        checkpoint.cleanup()
        
        # Guardar salida
        salida = {"modelos": []}
        print(json.dumps(salida))
        return
    
    # Procesar cada carrera
    resultados = []
    errores = []
    
    for carrera in carreras_pendientes:
        # Encontrar la configuración de esta carrera
        config = next((c for c in carreras_config if c.get('carrera_objetivo') == carrera), None)
        
        if not config:
            print(f"⚠️ Configuración no encontrada para {carrera}")
            continue
        
        try:
            resultado = procesar_una_carrera(config, timestamp_unico, output_base, spark, checkpoint)
            if resultado:
                resultados.append(resultado)
                checkpoint.marcar_procesada(carrera, resultado)
                print(f"✅ {carrera} procesada correctamente")
        except Exception as e:
            error_msg = str(e)
            errores.append({"carrera": carrera, "error": error_msg})
            checkpoint.marcar_fallida(carrera, error_msg)
            print(f"❌ {carrera} falló: {error_msg}")
    
    # Resumen final
    print("\n" + "=" * 80)
    print("📊 RESUMEN FINAL")
    print("=" * 80)
    print(f"✅ Exitosas: {len(resultados)}")
    print(f"❌ Fallidas: {len(errores)}")
    
    if errores:
        print("\n⚠️ Errores:")
        for e in errores:
            print(f"   - {e['carrera']}: {e['error'][:100]}")
    
    # Finalizar checkpoint
    if errores:
        checkpoint.finish("PARTIAL")
        print("⚠️ Ejecución parcial - revisa errores")
    else:
        checkpoint.finish("SUCCESS")
        checkpoint.cleanup()
        print("✅ Ejecución completa exitosa")
    
    # Salida para Step Functions
    salida = {"modelos": resultados}
    print(json.dumps(salida))


if __name__ == "__main__":
    main()