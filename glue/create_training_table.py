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
from glue_utils import analyze_split_requirements, extract_split_distance
from athena_utils import (
    execute_athena_query,
    tabla_existe,
    create_glue_table,
    s3,
    S3_ATHENA_OUTPUT,
    DATABASE
)


# ============================================================
# FUNCIONES DE ORQUESTACIÓN
# ============================================================

def obtener_carreras_historicas(carrera_objetivo, splits, event_id_filter=None, event_std_filter=None, num_carreras_necesarias=5):
    """Obtiene carreras históricas desde S3"""
    try:
        response = s3.get_object(
            Bucket='timingsense-races-processed-wide',
            Key='athena_catalog/esquemas_carreras.json'
        )
        catalogo = json.loads(response['Body'].read())
    except Exception as e:
        print(f"❌ Error cargando catálogo: {e}")
        return []

    match = re.match(r"(.*)-(\d{4})$", carrera_objetivo)
    if not match:
        print(f"❌ Formato inválido: {carrera_objetivo}")
        return []

    base, year_str = match.groups()
    year_objetivo = int(year_str)

    set_splits_objetivo = set([s.lower() for s in splits])
    print(f"🔍 DEBUG - Splits objetivo normalizados: {set_splits_objetivo}")

    historicas = []
    for _, info in catalogo['carreras'].items():
        race_id = info['race_id']
        if race_id.startswith(f"{base}-"):
            try:
                año = int(race_id.split('-')[-1])
            except:
                continue
            
            set_splits_catalogo = set([s.lower() for s in info['splits']])
            
            if año < year_objetivo and set_splits_catalogo == set_splits_objetivo:
                pasa_filtro = True
                if event_id_filter and info['event_id'] != event_id_filter:
                    pasa_filtro = False
                if event_std_filter and info.get('event_std') != event_std_filter:
                    pasa_filtro = False
                if pasa_filtro:
                    historicas.append(info)
                    print(f"✅ Match encontrado: {race_id}")

    print(f"📊 Total carreras históricas encontradas: {len(historicas)}")
    return historicas[:num_carreras_necesarias]


def obtener_carreras_historicas_adaptativo(carrera_objetivo, splits, event_id_filter=None, event_std_filter=None, num_carreras_necesarias=5):
    """Versión adaptativa que encuentra carreras históricas con splits compatibles"""
    try:
        response = s3.get_object(
            Bucket='timingsense-races-processed-wide',
            Key='athena_catalog/esquemas_carreras.json'
        )
        catalogo = json.loads(response['Body'].read())
    except Exception as e:
        print(f"❌ Error cargando catálogo: {e}")
        return [], None

    match = re.match(r"(.*)-(\d{4})$", carrera_objetivo)
    if not match:
        print(f"❌ Formato inválido: {carrera_objetivo}")
        return [], None

    base, year_str = match.groups()
    year_objetivo = int(year_str)

    splits_objetivo_norm = [s.lower().replace('.', '_') for s in splits]
    print(f"🔍 DEBUG - Splits objetivo normalizados: {splits_objetivo_norm}")

    candidatas = []
    for _, info in catalogo['carreras'].items():
        race_id = info['race_id']
        if race_id.startswith(f"{base}-"):
            try:
                año = int(race_id.split('-')[-1])
            except:
                continue
            
            if año < year_objetivo:
                set_splits_catalogo = set([s.lower() for s in info['splits']])
                
                pasa_filtro = True
                if event_id_filter and info['event_id'] != event_id_filter:
                    pasa_filtro = False
                if event_std_filter and info.get('event_std') != event_std_filter:
                    pasa_filtro = False
                
                if pasa_filtro:
                    candidatas.append({
                        'info': info,
                        'año': año,
                        'splits': set_splits_catalogo
                    })
                    print(f"✅ Candidata encontrada: {race_id} - Splits: {sorted(set_splits_catalogo)}")

    if not candidatas:
        print(f"⚠️ No se encontraron candidatas para {base}")
        return [], None

    todas_historicas = [c['info'] for c in candidatas]
    analisis = analyze_split_requirements(splits_objetivo_norm, todas_historicas)
    
    print(f"\n📊 RESULTADO DEL ANÁLISIS:")
    print(f"   Splits directos: {analisis['splits_directos']}")
    print(f"   Splits interpolables: {[s['split_objetivo'] for s in analisis['splits_interpolables']]}")
    print(f"   Splits imposibles: {analisis['splits_imposibles']}")
    
    if analisis['splits_imposibles']:
        print(f"\n⚠️ Hay splits imposibles, filtrando candidatas...")
        candidatas_filtradas = []
        for cand in candidatas:
            tiene_distancia = any(extract_split_distance(s) is not None for s in cand['splits'])
            if tiene_distancia:
                candidatas_filtradas.append(cand)
                print(f"   ✅ {cand['info']['race_id']} - tiene splits de distancia")
            else:
                print(f"   ❌ {cand['info']['race_id']} - descartada (sin splits de distancia)")
        
        if candidatas_filtradas:
            candidatas = candidatas_filtradas
        else:
            print(f"❌ No quedan candidatas después del filtro")
            return [], analisis
    
    candidatas.sort(key=lambda x: x['año'], reverse=True)
    seleccionadas = candidatas[:num_carreras_necesarias]
    
    historicas = [c['info'] for c in seleccionadas]
    
    print(f"\n📊 Carreras seleccionadas ({len(historicas)}):")
    for c in seleccionadas:
        print(f"   - {c['info']['race_id']} ({c['año']})")
    
    return historicas, analisis


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
    """Procesa una carrera individual"""
    carrera_objetivo = config["carrera_objetivo"]
    splits = config["splits"]
    event_id_filter = config.get("event_id_filter")
    event_std_filter = config.get("event_std_filter")

    print(f"\n🔍 CONFIGURACIÓN RECIBIDA:")
    print(f"   Carrera: {carrera_objetivo}")
    print(f"   Splits: {splits}")

    carreras_historicas, analisis = obtener_carreras_historicas_adaptativo(
        carrera_objetivo=carrera_objetivo,
        splits=splits,
        event_id_filter=event_id_filter,
        event_std_filter=event_std_filter,
        num_carreras_necesarias=5
    )

    if not carreras_historicas:
        print(f"⚠️ No se encontraron carreras históricas para {carrera_objetivo}")
        return None

    carpeta_modelo = f"{carrera_objetivo}-{timestamp_unico}"
    data_s3_path = f"s3://{S3_ATHENA_OUTPUT}/modelos/{carpeta_modelo}/data/"

    print(f"\n🚀 Procesando {carrera_objetivo}")
    print(f"📁 Carpeta modelo: {carpeta_modelo}")

    tabla_fuente, splits_finales = crear_tabla_temporal_adaptativa(
        splits, analisis, carreras_historicas
    )

    condiciones = [f"(race_id = '{c['race_id']}' AND event_id = '{c['event_id']}')" 
                   for c in carreras_historicas]
    where_clause = " OR ".join(condiciones)
    
    if event_id_filter and event_id_filter != 'None':
        where_clause = f"({where_clause}) AND event_id = '{event_id_filter}'"
    if event_std_filter and event_std_filter != 'None':
        where_clause = f"({where_clause}) AND event_std = '{event_std_filter}'"

    # Construir SELECT
    select_cols = ["athlete_id", "event_id", "event_std", "gender", "age"]
    
    for split_final in splits_finales:
        split_original = next((s for s in splits if s.replace('.', '_') == split_final), split_final)
        select_cols.append(f'"{split_final}" as "{split_original}"')
    
    select_clause = ",\n        ".join(select_cols)

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
        print(f"📁 Datos guardados en: {data_s3_path}")

    # Guardar metadata
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
        "carreras_usadas": len(carreras_historicas)
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
        "carreras_usadas": len(carreras_historicas)
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
        timestamp_unico = partes[-2] + '-' + partes[-1]
        timestamp_debug = timestamp_unico.replace('-', '_')
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