import boto3
import json
import re
import time
from datetime import datetime
from collections import defaultdict

cloudwatch = boto3.client('cloudwatch')

def publicar_metrica(nombre, valor, unidad, dimensiones=None):
    metric_data = [{
        'MetricName': nombre,
        'Value': valor,
        'Unit': unidad,
        'Timestamp': datetime.utcnow()
    }]
    if dimensiones:
        metric_data[0]['Dimensions'] = dimensiones
    try:
        cloudwatch.put_metric_data(Namespace='timingsense/Entrenamiento', MetricData=metric_data)
        print(f"📊 Métrica publicada: {nombre}={valor}")
    except Exception as e:
        print(f"⚠️ Error publicando métrica: {e}")

# ============================================================
# CONFIGURACIÓN
# ============================================================
UMBRAL_SIMILITUD = 0.75           # Para clasificar Grupo A
COBERTURA_MINIMA = 0.7            # 70% mínimo en alguna dirección
VARIACION_MAXIMA = 0.3            # 30% máximo de variación
MAX_CARRERAS = 5                  # Máximo de carreras a seleccionar
MARGEN_DISTANCIA = 50             # 50 metros de margen
MAX_RETRIES = 3                   # Reintentos para operaciones S3
RETRY_BACKOFF = [1, 2, 4]         # Backoff en segundos

# Clientes AWS
s3 = boto3.client('s3')


# ============================================================
# FUNCIONES DE REINTENTO
# ============================================================

def ejecutar_con_reintento(func, operation_name, max_retries=MAX_RETRIES):
    """Ejecuta una función con reintentos y backoff exponencial"""
    for i in range(max_retries):
        try:
            return func()
        except Exception as e:
            if i == max_retries - 1:
                print(f"   ❌ {operation_name} falló después de {max_retries} intentos")
                raise
            wait_time = RETRY_BACKOFF[i] if i < len(RETRY_BACKOFF) else 2 ** i
            print(f"   ⚠️ {operation_name} falló: {e}. Reintentando en {wait_time}s...")
            time.sleep(wait_time)


# ============================================================
# FUNCIONES DE UTILIDAD
# ============================================================

def levenshtein_distance(s1, s2):
    """Calcula distancia de Levenshtein entre dos strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalizar_nombre_carrera(race_id):
    """
    Normaliza el nombre de una carrera para comparación.
    Elimina años, prefijos numerales, números.
    """
    # Eliminar año al final (ej: -2024)
    nombre = re.sub(r'-\d{4}$', '', race_id)
    
    # Eliminar prefijos numerales romanos (xxxviii, xxxix, xl, etc.)
    nombre = re.sub(r'^(x{1,4}|ix|vi{0,3}|i?x|xl|xc|cd|cm)-', '', nombre, flags=re.IGNORECASE)
    
    # Eliminar prefijos numéricos (44-, 45-, etc.)
    nombre = re.sub(r'^\d+-', '', nombre)
    
    # Normalizar variaciones comunes
    nombre = re.sub(r'[áä]', 'a', nombre)
    nombre = re.sub(r'[éë]', 'e', nombre)
    nombre = re.sub(r'[íï]', 'i', nombre)
    nombre = re.sub(r'[óö]', 'o', nombre)
    nombre = re.sub(r'[úü]', 'u', nombre)
    
    # Normalizar "marato" vs "maratón"
    nombre = re.sub(r'marato', 'maraton', nombre)
    nombre = re.sub(r'maratn', 'maraton', nombre)
    
    return nombre.lower()


def calcular_similitud_nombre(race_id_1, race_id_2):
    """Calcula similitud entre dos nombres de carrera (0-1)"""
    norm_1 = normalizar_nombre_carrera(race_id_1)
    norm_2 = normalizar_nombre_carrera(race_id_2)
    
    distancia = levenshtein_distance(norm_1, norm_2)
    max_len = max(len(norm_1), len(norm_2))
    
    if max_len == 0:
        return 1.0
    
    return 1 - (distancia / max_len)


def listar_carreras_disponibles():
    """Lista todas las carreras disponibles en el catálogo"""
    def _listar():
        bucket = 'timingsense-races-processed-wide'
        prefix = 'current/wide/race_id='
        
        carreras = []
        paginator = s3.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/'):
            for prefix_obj in page.get('CommonPrefixes', []):
                race_id = prefix_obj['Prefix'].replace(prefix, '').rstrip('/')
                carreras.append(race_id)
        
        return carreras
    
    return ejecutar_con_reintento(_listar, "Listar carreras disponibles")


# ============================================================
# FUNCIONES PARA CARGAR SPLITS EFECTIVOS (CARRERAS HISTÓRICAS)
# ============================================================

def cargar_splits_efectivos(race_id):
    """
    Carga el archivo splits_efectivos de una carrera histórica.
    """
    def _cargar():
        bucket = 'timingsense-races-processed'
        key = f"splits_efectivos/{race_id}_splits_efectivos.json"
        
        try:
            response = s3.get_object(Bucket=bucket, Key=key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except s3.exceptions.NoSuchKey:
            print(f"   ⚠️ No se encontraron splits_efectivos para {race_id}")
            return None
    
    return ejecutar_con_reintento(_cargar, f"Cargar splits_efectivos {race_id}")


def obtener_mejor_evento_para_splits(race_id, splits_objetivo_distancias):
    """
    Para una carrera histórica, carga sus splits_efectivos y encuentra
    el evento que mejor cubre los splits objetivo.
    
    Retorna:
    {
        'evento': nombre_del_evento,
        'cobertura': float,
        'puntos_cubiertos': [distancias_cubiertas],
        'splits_nombres': [nombres_de_splits_del_evento],
        'splits_distancias': [distancias_del_evento]
    }
    o None si no hay evento que cumpla
    """
    data = cargar_splits_efectivos(race_id)
    if not data:
        return None
    
    mejor_evento = None
    mejor_cobertura = 0
    mejores_puntos_cubiertos = []
    mejores_splits_nombres = []
    mejores_splits_distancias = []
    
    for evento_nombre, evento_data in data.get('eventos', {}).items():
        # Extraer splits reales de este evento (excluyendo Salida)
        splits_evento = []
        for split in evento_data.get('splits_presentes', []):
            distancia = split.get('oficial_distance')
            if distancia is not None and distancia > 0:  # Excluir Salida (0)
                splits_evento.append({
                    'nombre': split.get('oficial_name'),
                    'distancia': distancia
                })
        
        if not splits_evento:
            continue
        
        # Calcular cobertura con los splits objetivo
        distancias_evento = [s['distancia'] for s in splits_evento]
        puntos_cubiertos = []
        
        for distancia_obj in splits_objetivo_distancias:
            for dist_evento in distancias_evento:
                if abs(distancia_obj - dist_evento) <= MARGEN_DISTANCIA:
                    puntos_cubiertos.append(distancia_obj)
                    break
        
        cobertura = len(puntos_cubiertos) / len(splits_objetivo_distancias) if splits_objetivo_distancias else 0
        
        if cobertura > mejor_cobertura:
            mejor_cobertura = cobertura
            mejor_evento = evento_nombre
            mejores_puntos_cubiertos = puntos_cubiertos
            mejores_splits_nombres = [s['nombre'] for s in splits_evento]
            mejores_splits_distancias = distancias_evento
    
    if mejor_evento is None:
        return None
    
    return {
        'evento': mejor_evento,
        'cobertura': mejor_cobertura,
        'puntos_cubiertos': mejores_puntos_cubiertos,
        'splits_nombres': mejores_splits_nombres,
        'splits_distancias': mejores_splits_distancias
    }


def calcular_variacion_splits(splits_objetivo, splits_historica_distancias):
    """Calcula la variación en número de splits (0-1)"""
    len_obj = len(splits_objetivo)
    len_hist = len(splits_historica_distancias)
    
    if len_obj == 0 or len_hist == 0:
        return 1.0
    
    return abs(len_obj - len_hist) / max(len_obj, len_hist)


def seleccionar_carreras_greedy(candidatas, splits_objetivo_distancias, max_carreras=MAX_CARRERAS):
    """
    Selección greedy: prioriza Grupo A (similitud >= UMBRAL_SIMILITUD)
    y aportación de splits nuevos.
    """
    # Separar en grupos
    grupo_a = [c for c in candidatas if c['similitud'] >= UMBRAL_SIMILITUD]
    grupo_b = [c for c in candidatas if c['similitud'] < UMBRAL_SIMILITUD]
    
    # Ordenar cada grupo por cobertura (mayor a menor)
    grupo_a.sort(key=lambda x: x['cobertura'], reverse=True)
    grupo_b.sort(key=lambda x: x['cobertura'], reverse=True)
    
    seleccionadas = []
    splits_cubiertos = set()
    
    print(f"\n🎯 Selección greedy:")
    print(f"   Grupo A (similitud ≥ {UMBRAL_SIMILITUD}): {len(grupo_a)} candidatas")
    print(f"   Grupo B (similitud < {UMBRAL_SIMILITUD}): {len(grupo_b)} candidatas")
    
    # Paso 1: Grupo A
    for c in grupo_a:
        nuevos = set(c['puntos_cubiertos']) - splits_cubiertos
        if nuevos:
            seleccionadas.append(c)
            splits_cubiertos.update(c['puntos_cubiertos'])
            print(f"   ✅ Seleccionada (A): {c['race_id']} / {c['evento']} - "
                  f"sim={c['similitud']:.2f}, cob={c['cobertura']:.1%}, "
                  f"aporta {len(nuevos)} splits nuevos")
            if len(seleccionadas) >= max_carreras:
                break
        if len(splits_cubiertos) == len(splits_objetivo_distancias):
            break
    
    # Paso 2: Grupo B (si no hay cobertura total)
    if len(splits_cubiertos) < len(splits_objetivo_distancias):
        for c in grupo_b:
            nuevos = set(c['puntos_cubiertos']) - splits_cubiertos
            if nuevos:
                seleccionadas.append(c)
                splits_cubiertos.update(c['puntos_cubiertos'])
                print(f"   ✅ Seleccionada (B): {c['race_id']} / {c['evento']} - "
                      f"sim={c['similitud']:.2f}, cob={c['cobertura']:.1%}, "
                      f"aporta {len(nuevos)} splits nuevos")
                if len(seleccionadas) >= max_carreras:
                    break
            if len(splits_cubiertos) == len(splits_objetivo_distancias):
                break
    
    cobertura_total = len(splits_cubiertos) / len(splits_objetivo_distancias) if splits_objetivo_distancias else 0
    puntos_faltantes = [d for d in splits_objetivo_distancias if d not in splits_cubiertos]
    
    return seleccionadas, cobertura_total, puntos_faltantes


# ============================================================
# FUNCIÓN PARA GUARDAR INFORME EN S3
# ============================================================

def guardar_informe_seleccion(carrera_objetivo, evento_objetivo, seleccionadas, cobertura_total,
                               puntos_faltantes, splits_objetivo, timestamp_unico,
                               output_bucket='timingsense-training-data'):
    """
    Guarda un informe detallado de la selección en S3.
    """
    # Crear estructura del informe
    informe = {
        "carrera_objetivo": carrera_objetivo,
        "evento_objetivo": evento_objetivo,
        "timestamp": timestamp_unico,
        "fecha_generacion": datetime.utcnow().isoformat(),
        "parametros_usados": {
            "umbral_similitud": UMBRAL_SIMILITUD,
            "cobertura_minima": COBERTURA_MINIMA,
            "variacion_maxima": VARIACION_MAXIMA,
            "max_carreras": MAX_CARRERAS,
            "margen_distancia": MARGEN_DISTANCIA
        },
        "splits_objetivo": splits_objetivo,
        "total_splits": len(splits_objetivo),
        "cobertura_total": cobertura_total,
        "splits_cubiertos": len([s for s in splits_objetivo if s['distancia_metros'] not in puntos_faltantes]),
        "splits_faltantes": [s for s in splits_objetivo if s['distancia_metros'] in puntos_faltantes],
        "carreras_seleccionadas": [],
        "resumen": {
            "total_carreras_seleccionadas": len(seleccionadas),
            "tiene_cobertura_completa": cobertura_total >= 0.99,
            "necesita_interpolacion": len(puntos_faltantes) > 0
        }
    }
    
    # Añadir detalle de cada carrera seleccionada
    for c in seleccionadas:
        splits_cubiertos_nombres = [
            s['nombre'] for s in splits_objetivo
            if s['distancia_metros'] in c['puntos_cubiertos']
        ]
        
        informe["carreras_seleccionadas"].append({
            "race_id": c['race_id'],
            "evento_usado": c['evento'],
            "similitud_nombre": round(c['similitud'], 4),
            "cobertura": round(c['cobertura'], 4),
            "variacion_splits": round(c['variacion'], 4),
            "splits_que_aporta": splits_cubiertos_nombres,
            "total_splits_aporta": len(c['puntos_cubiertos']),
            "grupo": "A" if c['similitud'] >= UMBRAL_SIMILITUD else "B"
        })
    
    # Guardar en S3
    bucket = output_bucket
    key = f"informes/seleccion/{carrera_objetivo}/{evento_objetivo}/{timestamp_unico}/informe_seleccion.json"
    
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(informe, indent=2, ensure_ascii=False),
            ContentType='application/json'
        )
        print(f"   📄 Informe guardado en: s3://{bucket}/{key}")
        
        # También guardar versión última
        key_resumen = f"informes/seleccion/{carrera_objetivo}/{evento_objetivo}/ultimo_informe.json"
        s3.put_object(
            Bucket=bucket,
            Key=key_resumen,
            Body=json.dumps(informe, indent=2, ensure_ascii=False),
            ContentType='application/json'
        )
        
        return f"s3://{bucket}/{key}"
    except Exception as e:
        print(f"   ⚠️ Error guardando informe: {e}")
        return None


# ============================================================
# LAMBDA PRINCIPAL
# ============================================================

def lambda_handler(event, context):
    # Función interna para publicar métricas en CloudWatch
    def publicar_metrica(nombre, valor, unidad, dimensiones=None):
        import boto3
        from datetime import datetime
        cloudwatch = boto3.client('cloudwatch')
        metric_data = [{
            'MetricName': nombre,
            'Value': valor,
            'Unit': unidad,
            'Timestamp': datetime.utcnow()
        }]
        if dimensiones:
            metric_data[0]['Dimensions'] = dimensiones
        try:
            cloudwatch.put_metric_data(Namespace='timingsense/Entrenamiento', MetricData=metric_data)
            print(f"📊 Métrica publicada: {nombre}={valor}")
        except Exception as e:
            print(f"⚠️ Error publicando métrica: {e}")

    print("=" * 60)
    print("🚀 PREPARANDO CONFIGURACIÓN PARA ENTRENAMIENTO")
    print("=" * 60)
    print(f"📥 Evento de entrada:")
    print(json.dumps(event, indent=2))
    
    # ============================================================
    # LEER EVENTO DE ENTRADA
    # ============================================================
    carreras_input = event.get('carreras', [])
    # Obtener nombre de la primera carrera para métricas (si existe)
    carrera_primera = carreras_input[0].get('carrera') or carreras_input[0].get('nombre', 'desconocida') if carreras_input else 'desconocida'
    
    if not carreras_input:
        raise ValueError("No se especificaron carreras")
    
    # Parámetros opcionales desde el evento
    global UMBRAL_SIMILITUD, COBERTURA_MINIMA, VARIACION_MAXIMA, MAX_CARRERAS, MARGEN_DISTANCIA
    
    config_sel = event.get('config_seleccion', {})
    UMBRAL_SIMILITUD = config_sel.get('umbral_similitud', UMBRAL_SIMILITUD)
    COBERTURA_MINIMA = config_sel.get('cobertura_minima', COBERTURA_MINIMA)
    VARIACION_MAXIMA = config_sel.get('variacion_maxima', VARIACION_MAXIMA)
    MAX_CARRERAS = config_sel.get('max_carreras', MAX_CARRERAS)
    MARGEN_DISTANCIA = config_sel.get('margen_distancia', MARGEN_DISTANCIA)
    
    print(f"\n📋 Configuración de selección:")
    print(f"   Umbral similitud: {UMBRAL_SIMILITUD}")
    print(f"   Cobertura mínima: {COBERTURA_MINIMA:.0%}")
    print(f"   Variación máxima: {VARIACION_MAXIMA:.0%}")
    print(f"   Máx carreras: {MAX_CARRERAS}")
    print(f"   Margen distancia: {MARGEN_DISTANCIA}m")
    
    try:
        # ============================================================
        # LISTAR TODAS LAS CARRERAS HISTÓRICAS DISPONIBLES
        # ============================================================
        todas_carreras = listar_carreras_disponibles()
        print(f"\n📂 Total carreras en catálogo: {len(todas_carreras)}")
        
        # ============================================================
        # TIMESTAMP ÚNICO
        # ============================================================
        timestamp_unico = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        
        # ============================================================
        # PROCESAR CADA (CARRERA, EVENTO) DE FORMA INDEPENDIENTE
        # ============================================================
        carreras_config = []
        
        for idx, carrera_info in enumerate(carreras_input, 1):
            carrera_objetivo = carrera_info.get('carrera') or carrera_info.get('nombre')
            evento_objetivo = carrera_info.get('evento')
            splits_requeridos = carrera_info.get('splits', [])
            tipo_modelo = carrera_info.get('tipo_modelo', 'interpolacion')
            training_params = carrera_info.get('training_params', {})
            event_id_filter = carrera_info.get('event_id_filter')
            event_std_filter = carrera_info.get('event_std_filter')
            
            print(f"\n{'='*60}")
            print(f"📌 [{idx}] Procesando: {carrera_objetivo} / {evento_objetivo}")
            print(f"{'='*60}")
            
            # ============================================================
            # VALIDACIONES
            # ============================================================
            if not carrera_objetivo:
                print(f"⚠️ Carrera sin nombre, omitiendo")
                continue
            
            if not evento_objetivo:
                print(f"⚠️ Evento sin nombre, omitiendo")
                continue
            
            if not splits_requeridos:
                print(f"⚠️ Sin splits definidos, omitiendo")
                continue
            
            # ============================================================
            # PROCESAR SPLITS DEL USUARIO (convertir a distancias)
            # ============================================================
            puntos_usuario = []
            splits_normalizados = []
            
            print(f"\n📏 Procesando splits del usuario:")
            for i, split in enumerate(splits_requeridos):
                if isinstance(split, str):
                    nombre_split = split
                    dist_km = re.search(r'(\d+(?:[.,]\d+)?)', split)
                    if dist_km:
                        distancia = int(float(dist_km.group(1).replace(',', '.')) * 1000)
                    elif split.lower() in ['finish', 'meta']:
                        distancia = 42195
                    else:
                        raise ValueError(f"Split '{split}' no se puede convertir a distancia. Usa formato con número o diccionario.")
                elif isinstance(split, dict):
                    nombre_split = split.get('nombre')
                    distancia = split.get('distancia_metros')
                    if not nombre_split:
                        raise ValueError(f"El diccionario en posición {i} debe tener clave 'nombre'")
                    if distancia is None:
                        raise ValueError(f"El diccionario en posición {i} debe tener clave 'distancia_metros'")
                    distancia = int(distancia)
                else:
                    raise ValueError(f"Cada split debe ser string o diccionario")
                
                puntos_usuario.append(distancia)
                splits_normalizados.append({
                    'nombre': nombre_split,
                    'distancia_metros': distancia
                })
                print(f"   {i+1}. '{nombre_split}' → {distancia} m")
            
            # ============================================================
            # EVALUAR CADA CARRERA HISTÓRICA
            # ============================================================
            candidatas = []
            
            print(f"\n🔍 Evaluando {len(todas_carreras)} carreras históricas...")
            
            for race_id in todas_carreras:
                if race_id == carrera_objetivo:
                    continue
                
                mejor_evento_info = obtener_mejor_evento_para_splits(race_id, puntos_usuario)
                if not mejor_evento_info:
                    continue
                
                if mejor_evento_info['cobertura'] < COBERTURA_MINIMA:
                    continue
                
                variacion = calcular_variacion_splits(puntos_usuario, mejor_evento_info['splits_distancias'])
                if variacion > VARIACION_MAXIMA:
                    continue
                
                similitud = calcular_similitud_nombre(carrera_objetivo, race_id)
                
                candidatas.append({
                    'race_id': race_id,
                    'evento': mejor_evento_info['evento'],
                    'similitud': similitud,
                    'cobertura': mejor_evento_info['cobertura'],
                    'puntos_cubiertos': mejor_evento_info['puntos_cubiertos'],
                    'variacion': variacion,
                    'splits_evento': mejor_evento_info['splits_nombres'],
                    'splits_distancias': mejor_evento_info['splits_distancias']
                })
                
                print(f"   📊 {race_id} / {mejor_evento_info['evento']}: "
                      f"cob={mejor_evento_info['cobertura']:.1%}, var={variacion:.1%}, sim={similitud:.2f}")
            
            print(f"\n📊 Carreras candidatas después de filtros: {len(candidatas)}")
            
            # ============================================================
            # SELECCIÓN GREEDY
            # ============================================================
            if not candidatas:
                print(f"❌ No se encontraron carreras candidatas para {carrera_objetivo} / {evento_objetivo}")
                continue
            
            seleccionadas, cobertura_total, puntos_faltantes = seleccionar_carreras_greedy(
                candidatas, puntos_usuario, MAX_CARRERAS
            )
            
            # ============================================================
            # GUARDAR INFORME EN S3
            # ============================================================
            informe_path = guardar_informe_seleccion(
                carrera_objetivo=carrera_objetivo,
                evento_objetivo=evento_objetivo,
                seleccionadas=seleccionadas,
                cobertura_total=cobertura_total,
                puntos_faltantes=puntos_faltantes,
                splits_objetivo=splits_normalizados,
                timestamp_unico=timestamp_unico,
                output_bucket='timingsense-training-data'
            )
            
            print(f"\n📊 Resultado selección:")
            print(f"   Carreras seleccionadas: {len(seleccionadas)}")
            print(f"   Cobertura total: {cobertura_total:.1%}")
            print(f"   Splits faltantes: {puntos_faltantes}")
            if informe_path:
                print(f"   📄 Informe: {informe_path}")
            
            # ============================================================
            # PREPARAR CONFIGURACIÓN FINAL PARA EL GLUE JOB
            # ============================================================
            carreras_historicas_detalle = []
            for c in seleccionadas:
                carreras_historicas_detalle.append({
                    'race_id': c['race_id'],
                    'evento': c['evento'],
                    'cobertura': c['cobertura'],
                    'similitud': c['similitud'],
                    'variacion': c['variacion']
                })
            
            tipo_seleccion = 'misma_carrera' if any(c['similitud'] >= UMBRAL_SIMILITUD for c in seleccionadas) else 'fallback'
            
            carreras_config.append({
                'carrera_objetivo': carrera_objetivo,
                'evento_objetivo': evento_objetivo,
                'splits': splits_requeridos,
                'splits_normalizados': splits_normalizados,
                'puntos_usuario': puntos_usuario,
                'carreras_historicas_detalle': carreras_historicas_detalle,
                'tipo_seleccion': tipo_seleccion,
                'cobertura_total': cobertura_total,
                'puntos_faltantes': puntos_faltantes,
                'event_id_filter': event_id_filter,
                'event_std_filter': event_std_filter,
                'tipo_modelo': tipo_modelo,
                'training_params': training_params,
                'informe_path': informe_path,
                'timestamp_unico': timestamp_unico,      # ← NUEVO
                'generated_at': timestamp_unico          # ← NUEVO
            })
            
            print(f"\n✅ {carrera_objetivo} / {evento_objetivo} preparado correctamente")
            print(f"   Carreras usadas: {[c['race_id'] for c in carreras_historicas_detalle]}")
        
        # ============================================================
        # SALIDA FINAL
        # ============================================================
        if not carreras_config:
            raise ValueError("No se pudo preparar ninguna carrera-evento para entrenamiento")
        
        salida = {
            "carreras_config": carreras_config,
            "num_modelos": len(carreras_config),
            "generated_at": timestamp_unico,
            "timestamp_unico": timestamp_unico,
            "config_usada": {
                "umbral_similitud": UMBRAL_SIMILITUD,
                "cobertura_minima": COBERTURA_MINIMA,
                "variacion_maxima": VARIACION_MAXIMA,
                "max_carreras": MAX_CARRERAS,
                "margen_distancia": MARGEN_DISTANCIA
            }
        }
        
        print("\n" + "=" * 60)
        print("📤 SALIDA FINAL")
        print("=" * 60)
        print(json.dumps(salida, indent=2))
        
        return salida
    
    except Exception as e:
        print(f"❌ Error en lambda de preparación: {str(e)}")
        publicar_metrica('fallo_etapa', 1, 'Count', [
            {'Name': 'Etapa', 'Value': 'PrepararConfig'},
            {'Name': 'Carrera', 'Value': carrera_primera}
        ])
        raise