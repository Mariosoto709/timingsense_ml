import json
import time
import boto3
import re
import hashlib
from datetime import datetime
from awsglue.utils import getResolvedOptions
import sys  
try:
    from urllib.parse import urlparse
except ImportError:
    urlparse = None

# ============================================================
# CONFIG
# ============================================================

REGION = "eu-north-1"
DATABASE = "timingsense_db"
WORKGROUP = "timingsense-ath"
S3_OUTPUT = "s3://timingsense-athena-output-2026/query-results/"
S3_PROCESSED_BUCKET = "timingsense-races-processed-wide"
S3_ATHENA_OUTPUT = "timingsense-athena-output-2026"

athena = boto3.client("athena", region_name=REGION)
s3 = boto3.client("s3")
glue = boto3.client("glue", region_name=REGION)

# ============================================================
# UTILIDADES ATHENA
# ============================================================

def wait_for_query(execution_id):
    while True:
        response = athena.get_query_execution(QueryExecutionId=execution_id)
        state = response["QueryExecution"]["Status"]["State"]

        if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            if state != "SUCCEEDED":
                reason = response["QueryExecution"]["Status"].get("StateChangeReason", "Unknown")
                raise Exception(f"Query falló: {reason}")
            return
        time.sleep(2)

def execute_athena_query(query):
    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": DATABASE},
        ResultConfiguration={"OutputLocation": S3_OUTPUT},
        WorkGroup=WORKGROUP
    )
    execution_id = response["QueryExecutionId"]
    wait_for_query(execution_id)
    result = athena.get_query_execution(QueryExecutionId=execution_id)
    return result["QueryExecution"]["ResultConfiguration"]["OutputLocation"]

# ============================================================
# FUNCIÓN PARA VERIFICAR SI UNA TABLA EXISTE
# ============================================================

def tabla_existe(nombre_tabla):
    try:
        glue.get_table(DatabaseName=DATABASE, Name=nombre_tabla)
        return True
    except glue.exceptions.EntityNotFoundException:
        return False
    except Exception as e:
        print(f"⚠️ Error verificando tabla {nombre_tabla}: {e}")
        return False

# ============================================================
# NUEVA FUNCIÓN: CREAR TABLA EN GLUE DIRECTAMENTE
# ============================================================

def create_glue_table(table_name, split_columns, partitioned=True):
    """
    Crea una tabla en el Catálogo de Datos de Glue apuntando a los datos Parquet.
    - table_name: nombre de la tabla a crear
    - split_columns: lista de strings como '"km_5" double' definiendo columnas de splits
    - partitioned: si True, la tabla se particiona por race_id y event_id
    """
    # Procesar columnas de splits
    columns = []
    for col_def in split_columns:
        # Extraer nombre y tipo de cada split
        match = re.match(r'"([^"]+)"\s+(\w+)', col_def)
        if match:
            col_name, col_type = match.groups()
        else:
            parts = col_def.split()
            col_name = parts[0].strip('"')
            col_type = parts[1] if len(parts) > 1 else 'string'
        columns.append({'Name': col_name, 'Type': col_type})

    # Columnas estándar que siempre tiene cualquier tabla
    standard_columns = [
        {'Name': 'athlete_id', 'Type': 'string'},
        {'Name': 'event_std', 'Type': 'string'},
        {'Name': 'gender', 'Type': 'string'},
        {'Name': 'age', 'Type': 'int'}
    ]

    # Todas las columnas = estándar + splits
    all_columns = standard_columns + columns

    # Configuración de almacenamiento (Parquet)
    storage_descriptor = {
        'Columns': all_columns,
        'Location': f's3://{S3_PROCESSED_BUCKET}/wide/',
        'InputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat',
        'OutputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat',
        'SerdeInfo': {
            'SerializationLibrary': 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe',
            'Parameters': {'serialization.format': '1'}
        },
        'Parameters': {'classification': 'parquet'}
    }

    # Definición completa de la tabla
    table_input = {
        'Name': table_name,
        'StorageDescriptor': storage_descriptor,
        'TableType': 'EXTERNAL_TABLE',
        'Parameters': {'EXTERNAL': 'TRUE'}
    }

    # Añadir particiones si es necesario
    if partitioned:
        table_input['PartitionKeys'] = [
            {'Name': 'race_id', 'Type': 'string'},
            {'Name': 'event_id', 'Type': 'string'}
        ]

    # Crear la tabla
    try:
        glue.create_table(DatabaseName=DATABASE, TableInput=table_input)
        print(f"✅ Tabla {table_name} creada en Glue")
    except glue.exceptions.AlreadyExistsException:
        print(f"ℹ️ Tabla {table_name} ya existe")
    except Exception as e:
        print(f"❌ Error creando tabla en Glue: {e}")
        raise

# ============================================================
# FUNCIÓN PARA CREAR TABLA TEMPORAL CON LOS SPLITS NECESARIOS
# ============================================================

def crear_tabla_temporal(splits, carreras_historicas):
    """
    Crea una tabla temporal que apunta a los datos wide existentes
    """
    print("\n📋 Creando/verificando tabla temporal...")
    
    # LOG 1: Mostrar splits originales
    print(f"   Splits originales (con punto): {splits}")
    
    # Normalizar splits para nombres de columna (reemplazar punto por guión bajo)
    splits_normalizados = [s.replace('.', '_') for s in splits]
    print(f"   Splits normalizados (con guión bajo): {splits_normalizados}")
    
    hash_splits = hashlib.md5('_'.join(sorted(splits_normalizados)).encode()).hexdigest()[:8]
    nombre_tabla_temp = f"temp_wide_{hash_splits}"
    
    # Verificar si la tabla ya existe
    if tabla_existe(nombre_tabla_temp):
        print(f"✅ Reutilizando tabla temporal existente: {nombre_tabla_temp}")
    else:
        print(f"🆕 Creando nueva tabla temporal: {nombre_tabla_temp}")
        
        # Construir columnas de splits - USAR NOMBRES NORMALIZADOS (guión bajo)
        columnas_splits = [f'"{s}" double' for s in splits_normalizados]
        print(f"   Columnas en tabla temporal: {columnas_splits}")
        
        create_query = f"""
        CREATE EXTERNAL TABLE {DATABASE}.{nombre_tabla_temp} (
            athlete_id string,
            event_std string,
            gender string,
            age int,
            {', '.join(columnas_splits)}
        )
        PARTITIONED BY (race_id string, event_id string)
        STORED AS PARQUET
        LOCATION 's3://{S3_PROCESSED_BUCKET}/wide/'
        """
        
        print(f"📝 Creando tabla temporal...")
        execute_athena_query(create_query)
        
        print(f"✅ Tabla temporal {nombre_tabla_temp} creada")
    
    # Añadir particiones necesarias
    print("\n🔧 Añadiendo particiones necesarias...")
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
    
    return nombre_tabla_temp

# ============================================================
# OBTENER CARRERAS HISTÓRICAS
# ============================================================

def obtener_carreras_historicas(carrera_objetivo, splits, event_id_filter=None, event_std_filter=None, num_carreras_necesarias=5):
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

    # Normalizar splits de la petición (a minúsculas)
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
            
            # Normalizar splits del catálogo (a minúsculas)
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

def extract_split_distance(split_name):
    """
    Extrae la distancia en km de un split normalizado.
    Acepta tanto km_18.2 como km_18_2
    """
    if split_name is None:
        return None
    
    split_lower = split_name.lower()
    
    # Splits especiales
    if split_lower == 'half':
        return 21.0975
    elif split_lower == 'finish':
        return 42.195
    elif split_lower == 'start':
        return 0.0
    
    # Splits km_X
    if split_lower.startswith('km_'):
        try:
            # Extraer parte numérica y NORMALIZAR
            numero_str = split_lower[3:]  # "18_2" o "18.2" o "5"
            # Convertir cualquier guión bajo a punto
            numero_str = numero_str.replace('_', '.')
            return float(numero_str)
        except (ValueError, TypeError):
            return None
    
    return None


def get_split_type(split_name):
    """
    Determina el tipo de split:
    - 'distance': split con distancia (km_X, half, finish, start)
    - 'other': cualquier otro tipo
    """
    return 'distance' if extract_split_distance(split_name) is not None else 'other'


def find_closest_split(historical_splits, target_distance):
    """
    Encuentra el split histórico más cercano a una distancia objetivo.
    Devuelve (split_name, distance)
    """
    if not historical_splits:
        return None, None
    
    # Extraer distancias de todos los splits históricos
    historical_with_dist = []
    for split in historical_splits:
        dist = extract_split_distance(split)
        if dist is not None:
            historical_with_dist.append((split, dist))
    
    if not historical_with_dist:
        return None, None
    
    # Encontrar el más cercano
    closest = min(historical_with_dist, key=lambda x: abs(x[1] - target_distance))
    return closest


def splits_son_equivalentes(split1, split2, tolerancia=0.001):
    """
    Determina si dos splits representan el mismo punto comparando sus distancias.
    """
    dist1 = extract_split_distance(split1)
    dist2 = extract_split_distance(split2)
    
    if dist1 is None or dist2 is None:
        return False
    
    return abs(dist1 - dist2) < tolerancia


def analyze_split_requirements(splits_objetivo, carreras_historicas):
    """
    Analiza qué splits de la nueva carrera están disponibles en datos históricos.
    
    Returns:
        dict con:
        - splits_directos: splits que existen directamente
        - splits_interpolables: splits km_X que se pueden interpolar
        - splits_imposibles: splits que no se pueden obtener
        - mapping: para cada split nuevo, de dónde obtenerlo
        - splits_finales: lista de splits que realmente se usarán en el modelo
    """
    print(f"\n🔍 ANALIZANDO REQUERIMIENTOS DE SPLITS:")
    
    # Normalizar splits objetivo (vienen con punto, convertir a guión bajo)
    splits_objetivo_norm = [s.replace('.', '_') for s in splits_objetivo]
    print(f"   Splits objetivo normalizados: {splits_objetivo_norm}")
    
    # Recopilar todos los splits disponibles en carreras históricas
    all_historical_splits = set()
    for carrera in carreras_historicas:
        all_historical_splits.update(carrera['splits'])
    
    print(f"   Splits históricos disponibles: {sorted(all_historical_splits)}")
    
    result = {
        'splits_directos': [],
        'splits_interpolables': [],
        'splits_imposibles': [],
        'mapping': {},
        'splits_finales': []
    }
    
    for split in splits_objetivo_norm:
        split_type = get_split_type(split)
        split_dist = extract_split_distance(split)
        
        # 🔥 CASO 1 MEJORADO: Buscar por distancia en lugar de nombre exacto
        split_encontrado = None
        if split_dist is not None:
            for hist_split in all_historical_splits:
                if splits_son_equivalentes(split, hist_split):
                    split_encontrado = hist_split
                    break
        
        # Caso 1: Split existe directamente en históricos (por nombre o por distancia)
        if split in all_historical_splits or split_encontrado is not None:
            nombre_real = split_encontrado if split_encontrado is not None else split
            result['splits_directos'].append(split)
            result['mapping'][split] = ('direct', nombre_real, split_dist)
            result['splits_finales'].append(split)
            if split in all_historical_splits:
                print(f"   ✅ DIRECT (por nombre): {split} existe en históricos")
            else:
                print(f"   ✅ DIRECT (por distancia): {split} ≡ {nombre_real} (dist={split_dist:.1f}km)")
        
        # Caso 2: Split de distancia que no existe (km_X, half, finish)
        elif split_type == 'distance' and split_dist is not None:
            # Encontrar el split histórico más cercano
            closest_split, closest_dist = find_closest_split(all_historical_splits, split_dist)
            
            if closest_split:
                result['splits_interpolables'].append({
                    'split_objetivo': split,
                    'split_origen': closest_split,
                    'distancia_objetivo': split_dist,
                    'distancia_origen': closest_dist,
                    'diferencia': abs(closest_dist - split_dist)
                })
                result['mapping'][split] = ('interpolate', closest_split, closest_dist)
                result['splits_finales'].append(split)
                print(f"   🔄 INTERPOLABLE: {split} (dist={split_dist:.1f}km) → desde {closest_split} (dist={closest_dist:.1f}km)")
            else:
                result['splits_imposibles'].append(split)
                print(f"   ❌ IMPOSIBLE: {split} - no hay splits de distancia en históricos")
        
        # Caso 3: Split no numérico que no existe
        else:
            result['splits_imposibles'].append(split)
            print(f"   ❌ IMPOSIBLE: {split} - no es un split de distancia y no existe en históricos")
    
    return result


def obtener_carreras_historicas_adaptativo(carrera_objetivo, splits, event_id_filter=None, event_std_filter=None, num_carreras_necesarias=5):
    """
    Versión adaptativa que encuentra carreras históricas con splits compatibles
    """
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

    # Normalizar splits de la petición (a minúsculas y con guión bajo)
    splits_objetivo_norm = [s.lower().replace('.', '_') for s in splits]
    print(f"🔍 DEBUG - Splits objetivo normalizados: {splits_objetivo_norm}")

    # Primera pasada: encontrar candidatas (misma carrera, años anteriores)
    candidatas = []
    for _, info in catalogo['carreras'].items():
        race_id = info['race_id']
        if race_id.startswith(f"{base}-"):
            try:
                año = int(race_id.split('-')[-1])
            except:
                continue
            
            if año < year_objetivo:
                # Normalizar splits del catálogo (ya vienen con guión bajo)
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

    # Analizar requerimientos con TODAS las candidatas primero
    todas_historicas = [c['info'] for c in candidatas]
    analisis = analyze_split_requirements(splits_objetivo_norm, todas_historicas)
    
    print(f"\n📊 RESULTADO DEL ANÁLISIS:")
    print(f"   Splits directos: {analisis['splits_directos']}")
    print(f"   Splits interpolables: {[s['split_objetivo'] for s in analisis['splits_interpolables']]}")
    print(f"   Splits imposibles: {analisis['splits_imposibles']}")
    
    # Si hay splits imposibles, filtrar candidatas que los tengan
    if analisis['splits_imposibles']:
        print(f"\n⚠️ Hay splits imposibles, filtrando candidatas...")
        candidatas_filtradas = []
        for cand in candidatas:
            # Verificar si esta candidata tiene ALGÚN split de distancia
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
    
    # Ordenar por año (más recientes primero) y tomar las necesarias
    candidatas.sort(key=lambda x: x['año'], reverse=True)
    seleccionadas = candidatas[:num_carreras_necesarias]
    
    historicas = [c['info'] for c in seleccionadas]
    
    print(f"\n📊 Carreras seleccionadas ({len(historicas)}):")
    for c in seleccionadas:
        print(f"   - {c['info']['race_id']} ({c['año']})")
    
    return historicas, analisis


def crear_tabla_temporal_adaptativa(splits, analisis, carreras_historicas):
    """
    Crea una tabla temporal considerando splits directos e interpolables
    VERSIÓN CORREGIDA - Usando GLUE API en lugar de Athena DDL
    """
    print(f"\n📋 Creando tabla temporal adaptativa...")
    
    # Splits que realmente usaremos (directos + interpolables)
    splits_finales = analisis['splits_finales']
    print(f"   Splits finales en modelo: {splits_finales}")
    
    # Al inicio de crear_tabla_temporal_adaptativa, después de definir splits_finales
    print(f"\n🔍 DEBUG - Splits en cada etapa:")
    print(f"   Splits originales (input): {splits}")
    print(f"   Splits finales (para modelo): {splits_finales}")
    print(f"   Mapping de análisis:")
    for split, (tipo, origen, dist) in analisis['mapping'].items():
        print(f"      {split}: {tipo} desde {origen} (dist={dist})")
        
    # LOG 1: Mostrar splits originales
    print(f"   Splits originales (con punto): {splits}")
    
    # Normalizar splits para nombres de columna
    splits_normalizados = [s.replace('.', '_') for s in splits]
    print(f"   Splits normalizados (con guión bajo): {splits_normalizados}")
    
    # Crear hash SOLO con los splits finales
    hash_splits = hashlib.md5('_'.join(sorted(splits_finales)).encode()).hexdigest()[:8]
    nombre_tabla_temp = f"temp_wide_{hash_splits}"
    
    # Verificar si la tabla ya existe
    if tabla_existe(nombre_tabla_temp):
        print(f"✅ Reutilizando tabla temporal existente: {nombre_tabla_temp}")
    else:
        print(f"🆕 Creando nueva tabla temporal: {nombre_tabla_temp}")
        
        # Construir columnas de splits - SOLO los finales
        columnas_splits = [f'"{s}" double' for s in splits_finales]
        print(f"   Columnas en tabla temporal: {columnas_splits}")
        
        # 🟢🟢🟢 USAR GLUE API EN LUGAR DE ATHENA DDL 🟢🟢🟢
        try:
            create_glue_table(nombre_tabla_temp, columnas_splits, partitioned=True)
            print(f"✅ Tabla temporal {nombre_tabla_temp} creada via Glue")
        except Exception as e:
            print(f"❌ Error creando tabla con particiones: {e}")
            # Si falla con particiones, intentar sin particiones
            print("🔄 Intentando sin particiones...")
            create_glue_table(nombre_tabla_temp, columnas_splits, partitioned=False)
            print(f"✅ Tabla temporal {nombre_tabla_temp} creada (sin particiones)")
    
    # Añadir particiones necesarias (solo si la tabla tiene particiones)
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


def procesar_una_carrera(config):
    carrera_objetivo = config["carrera_objetivo"]
    splits = config["splits"]
    event_id_filter = config.get("event_id_filter")
    event_std_filter = config.get("event_std_filter")

    print(f"\n🔍 CONFIGURACIÓN RECIBIDA:")
    print(f"   Carrera: {carrera_objetivo}")
    print(f"   Splits: {splits}")
    print(f"   event_id_filter: {event_id_filter}")
    print(f"   event_std_filter: {event_std_filter}")

    # PASO 1: Obtener carreras históricas con análisis adaptativo
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

    # Guardar el análisis en S3 para referencia
    carpeta_modelo = f"{carrera_objetivo}-{timestamp_unico}"
    data_s3_path = f"s3://{S3_ATHENA_OUTPUT}/modelos/{carpeta_modelo}/data/"

    print(f"\n🚀 Procesando {carrera_objetivo}")
    print(f"📁 Carpeta modelo: {carpeta_modelo}")
    print(f"📁 Ruta S3 datos: {data_s3_path}")

    # PASO 2: Crear tabla temporal con los splits finales
    tabla_fuente, splits_finales = crear_tabla_temporal_adaptativa(
        splits, 
        analisis, 
        carreras_historicas
    )

    # PASO 3: Construir WHERE clause con las carreras históricas
    condiciones = [f"(race_id = '{c['race_id']}' AND event_id = '{c['event_id']}')" 
                   for c in carreras_historicas]
    where_clause = " OR ".join(condiciones)
    
    if event_id_filter and event_id_filter != 'None':
        where_clause = f"({where_clause}) AND event_id = '{event_id_filter}'"
    if event_std_filter and event_std_filter != 'None':
        where_clause = f"({where_clause}) AND event_std = '{event_std_filter}'"
        
    # DESPUÉS DE PASO 2 (justo después de crear tabla_fuente)
    print(f"\n🔍 DEBUG - Verificando tabla temporal: {tabla_fuente}")
    
    # Consulta para ver las primeras filas de la tabla temporal
    debug_query = f"""
    SELECT athlete_id, race_id, event_id, gender, age, 
           "km_5", "km_10", "km_15", "km_18_2", "km_20", 
           "half", "km_25", "km_28", "km_30", "km_35", "km_40", "finish"
    FROM {DATABASE}.{tabla_fuente}
    WHERE {where_clause}
    LIMIT 5
    """
    
    print(f"📝 Ejecutando debug query en tabla temporal...")
    try:
        debug_location = execute_athena_query(debug_query)
        print(f"✅ Resultados guardados en: {debug_location}")
        
        # Leer resultados directamente
        s3 = boto3.client('s3')
        parsed = urlparse(debug_location)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        
        # Buscar archivo CSV de resultados
        response = s3.get_object(Bucket=bucket, Key=key)
        import csv
        content = response['Body'].read().decode('utf-8').splitlines()
        reader = csv.reader(content)
        headers = next(reader)
        rows = list(reader)
        
        print(f"\n📊 PRIMERAS 5 FILAS DE TABLA TEMPORAL:")
        print(f"   Headers: {headers}")
        for i, row in enumerate(rows[:5]):
            print(f"   Fila {i+1}: {row}")
    except Exception as e:
        print(f"⚠️ Error en debug query: {e}")

    # PASO 4: Construir SELECT clause con los splits finales
    select_cols = [
        "athlete_id",
        "event_id",
        "event_std",
        "gender",
        "age"
    ]
    
    print(f"\n🔧 Construyendo SELECT con splits finales:")
    for split_final in splits_finales:
        # Buscar el split original correspondiente (con punto) para el alias
        split_original = next((s for s in splits if s.replace('.', '_') == split_final), split_final)
        print(f"   Columna en tabla: '{split_final}' → Se usará como '{split_original}' en el modelo")
        # Usar el nombre normalizado (split_final) como columna fuente
        select_cols.append(f'"{split_final}" as "{split_original}"')
    
    select_clause = ",\n        ".join(select_cols)

    # PASO 5: Crear nombre de tabla resultado
    nombre_limpio = re.sub(r'[^a-zA-Z0-9_-]', '_', carrera_objetivo)
    nombre_tabla_resultado = f"modelo_{nombre_limpio}_{timestamp_unico}".replace('-', '_')
    print(f"\n📊 Tabla resultado: {nombre_tabla_resultado}")

    # PASO 6: Verificar si la tabla resultado ya existe
    if tabla_existe(nombre_tabla_resultado):
        print(f"⚠️ La tabla {nombre_tabla_resultado} ya existe. Usando datos existentes.")
        print(f"📁 Datos disponibles en: {data_s3_path}")
    else:
        # PASO 7: Crear tabla CTAS con los datos de entrenamiento
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
        
        print(f"\n📝 CTAS query (primeras 200 caracteres):")
        print(ctas_query[:200] + "...")
        
        print(f"\n📝 Ejecutando CTAS query...")
        data_location = execute_athena_query(ctas_query)
        print(f"✅ Tabla creada en: {data_location}")
        print(f"📁 Datos guardados en: {data_s3_path}")
        
        # DESPUÉS DE PASO 7 (después de ejecutar CTAS)
        print(f"\n🔍 DEBUG - Verificando tabla resultado: {nombre_tabla_resultado}")
        
        # Consulta para ver las primeras filas de la tabla resultado
        debug_result_query = f"""
        SELECT athlete_id, event_id, gender, age, 
               "km_5", "km_10", "km_15", "km_18.2", "km_20", 
               "half", "km_25", "km_28", "km_30", "km_35", "km_40", "finish"
        FROM {DATABASE}.{nombre_tabla_resultado}
        LIMIT 5
        """
        
        print(f"📝 Ejecutando debug query en tabla resultado...")
        try:
            debug_result_location = execute_athena_query(debug_result_query)
            print(f"✅ Resultados guardados en: {debug_result_location}")
            
            # Leer resultados
            parsed = urlparse(debug_result_location)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            
            response = s3.get_object(Bucket=bucket, Key=key)
            import csv
            content = response['Body'].read().decode('utf-8').splitlines()
            reader = csv.reader(content)
            headers = next(reader)
            rows = list(reader)
            
            print(f"\n📊 PRIMERAS 5 FILAS DE TABLA RESULTADO:")
            print(f"   Headers: {headers}")
            for i, row in enumerate(rows[:5]):
                print(f"   Fila {i+1}: {row}")
            
            # Verificar específicamente km_18.2
            km_18_2_idx = headers.index('km_18.2') if 'km_18.2' in headers else -1
            if km_18_2_idx >= 0:
                print(f"\n✅ Verificación km_18.2:")
                for i, row in enumerate(rows[:5]):
                    print(f"   Atleta {i+1}: km_18.2 = {row[km_18_2_idx]}")
                    
        except Exception as e:
            print(f"⚠️ Error en debug query: {e}")

    # PASO 8: Guardar metadata completa
    metadata = {
        "carrera": carrera_objetivo,
        "timestamp": timestamp_unico,
        "splits_originales": splits,
        "splits_normalizados": [s.replace('.', '_') for s in splits],
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
        "carreras_utilizadas": [
            {
                "race_id": c['race_id'],
                "event_id": c['event_id']
            } for c in carreras_historicas
        ],
        "carreras_usadas": len(carreras_historicas)
    }

    # Guardar metadata en S3
    s3.put_object(
        Bucket=S3_ATHENA_OUTPUT,
        Key=f"modelos/{carpeta_modelo}/data/metadata.json",
        Body=json.dumps(metadata, indent=2)
    )
    
    print(f"✅ Metadata guardada en: s3://{S3_ATHENA_OUTPUT}/modelos/{carpeta_modelo}/metadata.json")

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

# Intentar obtener ambos argumentos a la vez
try:
    args = getResolvedOptions(sys.argv, ["carreras_json", "timestamp_unico"])
    carreras_json = args["carreras_json"]
    timestamp_unico = args["timestamp_unico"]
except:
    # Si no viene timestamp_unico, obtener solo carreras_json
    args = getResolvedOptions(sys.argv, ["carreras_json"])
    carreras_json = args["carreras_json"]
    timestamp_unico = None

# ¡IMPORTANTE! Cargar la configuración
carreras_config = json.loads(carreras_json)

if not timestamp_unico:
    # Si no hay timestamp, buscarlo en la configuración
    timestamp_unico = carreras_config[0].get('timestamp_unico') if carreras_config else None
    
if not timestamp_unico:
    # Fallback: generar uno nuevo
    from datetime import datetime
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
        resultado = procesar_una_carrera(config)
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

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "=" * 80)
print("📊 RESUMEN FINAL")
print("=" * 80)
print(f"✅ Carreras exitosas: {len(resultados)}")
print(f"❌ Carreras fallidas: {len(errores)}")

if resultados:
    print("\n📋 Carpetas creadas:")
    for r in resultados:
        print(f"   - {r['carpeta_modelo']}/")
        print(f"     ├── data/ (datos Parquet)")
        print(f"     └── metadata.json")

if errores:
    print("\n⚠️ Errores encontrados:")
    for e in errores:
        print(f"   - {e['carrera']}: {e['error']}")

print("\n" + "=" * 80)
print("✅ JOB FINALIZADO")
print("=" * 80)

# ============================================================
# 🔴 SALIDA PARA STEP FUNCTIONS - VERSIÓN DEFINITIVA
# ============================================================
# ⚠️ IMPORTANTE: Esto DEBE ser lo ÚLTIMO que se imprima

import sys

# Preparar salida final (SOLO lo que necesita Step Functions)
salida_final = {
    "modelos": resultados if resultados else []
}

# Guardar en S3 para debug (SIN PRINT)
try:
    # 🟢 USAR EL TIMESTAMP DE LA PRIMERA CARPETA CREADA
    if resultados:
        # Extraer timestamp de la primera carpeta (ej: "carrera-20260303-085722")
        carpeta_ejemplo = resultados[0]['carpeta_modelo']
        # El timestamp es todo después del último guión? No, formato es "nombre-YYYYMMDD-HHMMSS"
        partes = carpeta_ejemplo.split('-')
        # El timestamp son las dos últimas partes: YYYYMMDD y HHMMSS
        timestamp_unico = partes[-2] + '-' + partes[-1]  # "20260303-085722"
        # Convertir a formato con guión bajo para el archivo
        timestamp_debug = timestamp_unico.replace('-', '_')  # "20260303_085722"
    else:
        timestamp_debug = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    s3.put_object(
        Bucket=S3_ATHENA_OUTPUT,
        Key=f"debug/salida_step_{timestamp_debug}.json",
        Body=json.dumps(salida_final)
    )
except Exception:
    pass   # Silencio absoluto - no imprimir nada

# Limpiar TODOS los buffers
sys.stdout.flush()
sys.stderr.flush()

# ⚠️ IMPRIMIR ÚNICAMENTE EL JSON (una línea, sin decoraciones)
print(json.dumps(salida_final))

# Forzar escritura inmediata
sys.stdout.flush()
sys.stderr.flush()

# ⚠️ FIN ABSOLUTO DEL SCRIPT - NO HAY NADA MÁS