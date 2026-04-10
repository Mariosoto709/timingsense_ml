# glue/create-training-table.py
"""
Job principal de Glue - Orquesta la creación de tablas de entrenamiento
"""

import json
import sys
import hashlib
import re
from datetime import datetime
from awsglue.utils import getResolvedOptions

# Importar utilidades locales
from glue_utils import analyze_split_requirements
from athena_utils import (
    execute_athena_query,
    tabla_existe,
    create_glue_table,
    s3,
    S3_ATHENA_OUTPUT,
    DATABASE
)


def crear_tabla_temporal_adaptativa(splits, analisis, carreras_historicas):
    """Crea una tabla temporal considerando splits directos e interpolables"""
    print(f"\n📋 Creando tabla temporal adaptativa...")
    
    splits_finales = analisis['splits_finales']
    print(f"   Splits finales en modelo: {splits_finales}")
    
    print(f"\n🔍 DEBUG - Splits en cada etapa:")
    print(f"   Splits originales (input): {splits}")
    print(f"   Splits finales (para modelo): {splits_finales}")
    
    splits_normalizados = [s.replace('.', '_') for s in splits]
    print(f"   Splits normalizados (con guión bajo): {splits_normalizados}")
    
    hash_splits = hashlib.md5('_'.join(sorted(splits_finales)).encode()).hexdigest()[:8]
    nombre_tabla_temp = f"temp_wide_{hash_splits}"
    
    if tabla_existe(nombre_tabla_temp):
        print(f"✅ Reutilizando tabla temporal existente: {nombre_tabla_temp}")
    else:
        print(f"🆕 Creando nueva tabla temporal: {nombre_tabla_temp}")
        
        columnas_splits = [f'"{s}" double' for s in splits_finales]
        print(f"   Columnas en tabla temporal: {columnas_splits}")
        
        try:
            create_glue_table(nombre_tabla_temp, columnas_splits, partitioned=True)
            print(f"✅ Tabla temporal {nombre_tabla_temp} creada via Glue")
        except Exception as e:
            print(f"❌ Error creando tabla con particiones: {e}")
            print("🔄 Intentando sin particiones...")
            create_glue_table(nombre_tabla_temp, columnas_splits, partitioned=False)
            print(f"✅ Tabla temporal {nombre_tabla_temp} creada (sin particiones)")
    
    print(f"\n🔧 Añadiendo particiones necesarias...")
    for carrera in carreras_historicas:
        add_partition_query = f"""
        ALTER TABLE {DATABASE}.{nombre_tabla_temp} 
        ADD PARTITION (race_id = '{carrera['race_id']}', event_id = '{carrera['event_id']}')
        """
        try:
            execute_athena_query(add_partition_query)
            print(f"   ✅ Partición añadida: {carrera['race_id']}/{carrera['event_id']}")
        except Exception as e:
            print(f"   ℹ️ Partición ya existente o error: {e}")
    
    return nombre_tabla_temp, splits_finales


def procesar_una_carrera(config, timestamp_unico):
    """
    Procesa una carrera individual.
    Las carreras históricas ya vienen seleccionadas desde la Lambda.
    """
    carrera_objetivo = config["carrera_objetivo"]
    splits = config["splits"]  # nombres originales de splits (ej. ["5K", "10K", "Media"])
    event_id_filter = config.get("event_id_filter")
    event_std_filter = config.get("event_std_filter")
    tipo_seleccion = config.get('tipo_seleccion', 'desconocido')
    cobertura_total = config.get('cobertura_total', 0)
    
    # =============================================================
    # OBTENER CARRERAS HISTÓRICAS DESDE LA CONFIGURACIÓN
    # =============================================================
    carreras_historicas_detalle = config.get('carreras_historicas_detalle', [])
    
    if not carreras_historicas_detalle:
        print(f"⚠️ No se proporcionaron carreras históricas para {carrera_objetivo}")
        return None
    
    # Transformar al formato que espera analyze_split_requirements
    carreras_historicas = []
    for h in carreras_historicas_detalle:
        carreras_historicas.append({
            'race_id': h['race_id'],
            'event_id': h.get('evento', 'unknown'),
            'splits': h['splits']  # lista de strings normalizados (ej. ["km_5", "km_10", "half"])
        })
    
    print(f"\n🔍 CONFIGURACIÓN RECIBIDA:")
    print(f"   Carrera objetivo: {carrera_objetivo}")
    print(f"   Splits solicitados: {splits}")
    print(f"   Tipo selección: {tipo_seleccion}")
    print(f"   Cobertura total: {cobertura_total:.1%}")
    print(f"   Carreras históricas a usar: {len(carreras_historicas)}")
    
    for c in carreras_historicas:
        print(f"      - {c['race_id']} (evento: {c['event_id']}, splits: {len(c['splits'])})")
    
    # =============================================================
    # ANALIZAR REQUISITOS DE SPLITS
    # =============================================================
    analisis = analyze_split_requirements(splits, carreras_historicas)
    
    print(f"\n📊 RESULTADO DEL ANÁLISIS:")
    print(f"   Splits directos: {analisis['splits_directos']}")
    print(f"   Splits interpolables: {[s['split_objetivo'] for s in analisis['splits_interpolables']]}")
    print(f"   Splits imposibles: {analisis['splits_imposibles']}")

    # =============================================================
    # CREAR CARPETA Y TABLA TEMPORAL
    # =============================================================
    carpeta_modelo = f"{carrera_objetivo}-{timestamp_unico}"
    data_s3_path = f"s3://{S3_ATHENA_OUTPUT}/modelos/{carpeta_modelo}/data/"

    print(f"\n🚀 Procesando {carrera_objetivo}")
    print(f"📁 Carpeta modelo: {carpeta_modelo}")

    tabla_fuente, splits_finales = crear_tabla_temporal_adaptativa(
        splits, analisis, carreras_historicas
    )

    # =============================================================
    # CONSTRUIR WHERE CLAUSE
    # =============================================================
    condiciones = [f"(race_id = '{c['race_id']}' AND event_id = '{c['event_id']}')" 
                   for c in carreras_historicas]
    where_clause = " OR ".join(condiciones)
    
    if event_id_filter and event_id_filter != 'None':
        where_clause = f"({where_clause}) AND event_id = '{event_id_filter}'"
    if event_std_filter and event_std_filter != 'None':
        where_clause = f"({where_clause}) AND event_std = '{event_std_filter}'"

    # =============================================================
    # CONSTRUIR SELECT CON LOS SPLITS FINALES
    # =============================================================
    select_cols = ["athlete_id", "event_id", "event_std", "gender", "age"]
    
    for split_final in splits_finales:
        split_original = next((s for s in splits if s.replace('.', '_') == split_final), split_final)
        select_cols.append(f'"{split_final}" as "{split_original}"')
    
    select_clause = ",\n        ".join(select_cols)

    # =============================================================
    # CREAR TABLA RESULTADO (CTAS)
    # =============================================================
    nombre_limpio = re.sub(r'[^a-zA-Z0-9_-]', '_', carrera_objetivo)
    nombre_tabla_resultado = f"modelo_{nombre_limpio}_{timestamp_unico}".replace('-', '_')
    print(f"\n📊 Tabla resultado: {nombre_tabla_resultado}")

    if not tabla_existe(nombre_tabla_resultado):
        ctas_query = f"""
        CREATE TABLE {DATABASE}.{nombre_tabla_resultado}
        WITH (
            format = 'PARQUET',
            write_compression = 'SNAPPY',
            external_location = '{data_s3_path}'
        ) AS
        SELECT {select_clause}
        FROM {DATABASE}.{tabla_fuente}
        WHERE {where_clause}
        """
        
        print(f"\n📝 Ejecutando CTAS query...")
        data_location = execute_athena_query(ctas_query)
        print(f"✅ Tabla creada en: {data_location}")

        # =============================================================
        # VALIDACIÓN DE CALIDAD DE DATOS
        # =============================================================
        print(f"🔍 Validando calidad dataset...")
        try:
            import pandas as pd
            df_sample = pd.read_parquet(data_location, nrows=1000)
            
            # Volumen mínimo
            if len(df_sample) < 100:
                raise ValueError(f"Datos insuficientes: {len(df_sample)} filas")
            
            # Nulos totales >20%
            null_ratio = df_sample.isnull().sum().sum() / (len(df_sample) * len(df_sample.columns))
            if null_ratio > 0.20:
                raise ValueError(f"Demasiados nulos: {null_ratio:.1%} ({df_sample.isnull().sum().sum()} nulos)")
            
            print(f"✅ Dataset OK: {len(df_sample)} filas, {null_ratio:.1%} nulos")
            
        except Exception as e:
            print(f"❌ VALIDACIÓN FALLIDA: {e}")
            # Limpiar tabla fallida
            drop_query = f"DROP TABLE {DATABASE}.{nombre_tabla_resultado}"
            execute_athena_query(drop_query)
            raise ValueError(f"Dataset inválido: {e}")

        print(f"📁 Datos guardados en: {data_s3_path}")

    # =============================================================
    # GUARDAR METADATA
    # =============================================================
    metadata = {
        "carrera": carrera_objetivo,
        "timestamp": timestamp_unico,
        "splits_originales": splits,
        "splits_finales": splits_finales,
        "analisis": {
            "splits_directos": analisis['splits_directos'],
            "splits_interpolables": analisis['splits_interpolables'],
            "splits_imposibles": analisis['splits_imposibles'],
            "mapping": analisis['mapping']
        },
        "data_s3_path": data_s3_path,
        "tabla_fuente": tabla_fuente,
        "tabla_generada": nombre_tabla_resultado,
        "carreras_utilizadas": [{"race_id": c['race_id'], "event_id": c['event_id']} for c in carreras_historicas],
        "carreras_usadas": len(carreras_historicas),
        "tipo_seleccion": tipo_seleccion,
        "cobertura_total": cobertura_total
    }

    s3.put_object(
        Bucket=S3_ATHENA_OUTPUT,
        Key=f"modelos/{carpeta_modelo}/data/metadata.json",
        Body=json.dumps(metadata, indent=2)
    )
    
    return {
        "carrera": carrera_objetivo,
        "carpeta_modelo": carpeta_modelo,
        "data_s3_path": data_s3_path,
        "tabla_generada": nombre_tabla_resultado,
        "splits": len(splits_finales),
        "splits_originales": len(splits),
        "splits_interpolados": len(analisis['splits_interpolables']),
        "splits_imposibles": len(analisis['splits_imposibles']),
        "carreras_usadas": len(carreras_historicas),
        "tipo_seleccion": tipo_seleccion,
        "cobertura_total": cobertura_total
    }


# ============================================================
# MAIN GLUE ENTRYPOINT
# ============================================================

try:
    args = getResolvedOptions(sys.argv, ["carreras_json", "timestamp_unico"])
    carreras_json = args["carreras_json"]
    timestamp_unico = args["timestamp_unico"]
except:
    args = getResolvedOptions(sys.argv, ["carreras_json"])
    carreras_json = args["carreras_json"]
    timestamp_unico = None

carreras_config = json.loads(carreras_json)

if not timestamp_unico:
    timestamp_unico = carreras_config[0].get('timestamp_unico') if carreras_config else None
    
if not timestamp_unico:
    timestamp_unico = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"⚠️ timestamp_unico no proporcionado, generado: {timestamp_unico}")

print("=" * 80)
print("🚀 JOB GLUE - CREAR TABLAS DE ENTRENAMIENTO")
print("=" * 80)
print(f"📥 Carreras a procesar: {len(carreras_config)}")
print(f"🕒 Timestamp único: {timestamp_unico}")

resultados = []
errores = []

for idx, config in enumerate(carreras_config, 1):
    print(f"\n{'='*60}")
    print(f"📌 Procesando carrera {idx}/{len(carreras_config)}")
    print(f"{'='*60}")
    
    try:
        resultado = procesar_una_carrera(config, timestamp_unico)
        if resultado:
            resultados.append(resultado)
            print(f"✅ Carrera {idx} procesada correctamente")
        else:
            errores.append({
                "carrera": config.get('carrera_objetivo'),
                "error": "No se encontraron datos históricos"
            })
            print(f"⚠️ Carrera {idx} sin datos históricos")
    except Exception as e:
        errores.append({
            "carrera": config.get('carrera_objetivo'),
            "error": str(e)
        })
        print(f"❌ Error en carrera {idx}: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("📊 RESUMEN FINAL")
print("=" * 80)
print(f"✅ Carreras exitosas: {len(resultados)}")
print(f"❌ Carreras fallidas: {len(errores)}")

if resultados:
    print("\n📋 Carpetas creadas:")
    for r in resultados:
        print(f"   - {r['carpeta_modelo']}/")

if errores:
    print("\n⚠️ Errores encontrados:")
    for e in errores:
        print(f"   - {e['carrera']}: {e['error']}")

print("\n" + "=" * 80)
print("✅ JOB FINALIZADO")
print("=" * 80)

# Salida para Step Functions
salida_final = {"modelos": resultados if resultados else []}

# Guardar en S3 para debug
try:
    if resultados:
        carpeta_ejemplo = resultados[0]['carpeta_modelo']
        partes = carpeta_ejemplo.split('-')
        timestamp_debug = partes[-2] + '_' + partes[-1]
    else:
        timestamp_debug = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    s3.put_object(
        Bucket=S3_ATHENA_OUTPUT,
        Key=f"debug/salida_step_{timestamp_debug}.json",
        Body=json.dumps(salida_final)
    )
except Exception:
    pass

sys.stdout.flush()
sys.stderr.flush()
print(json.dumps(salida_final))
sys.stdout.flush()
sys.stderr.flush()