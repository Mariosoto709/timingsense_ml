# glue/glue_utils.py
"""
Funciones auxiliares para el job Glue (sin dependencias de AWS)
Estas funciones se pueden testear localmente sin necesidad de AWS.
"""


# ========== NUEVAS FUNCIONES PARA SELECCIÓN DE CARRERAS ==========

def cargar_catalogo_distancias(race_id, bucket='timingsense-config', prefix='splits_catalog/distancias/'):
    """Carga el catálogo de distancias de una carrera desde S3."""
    import boto3, json
    s3 = boto3.client('s3')
    key = f"{prefix}{race_id}.json"
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(response['Body'].read())
    except Exception as e:
        print(f"⚠️ No se pudo cargar catálogo para {race_id}: {e}")
        return None

def listar_carreras_disponibles(bucket='timingsense-config', prefix='splits_catalog/distancias/'):
    """Lista todos los race_id disponibles en el catálogo."""
    import boto3, re
    s3 = boto3.client('s3')
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        race_ids = []
        for obj in response.get('Contents', []):
            key = obj['Key']
            match = re.search(r'([^/]+)\.json$', key)
            if match:
                race_ids.append(match.group(1))
        return race_ids
    except Exception as e:
        print(f"⚠️ Error listando catálogos: {e}")
        return []

def calcular_cobertura_carrera(catalogo, puntos_usuario, tolerancia=10, event_id_filter=None):
    """
    Calcula qué porcentaje de los puntos del usuario cubre una carrera.
    
    Args:
        catalogo: diccionario con el catálogo de la carrera
        puntos_usuario: lista de distancias en metros
        tolerancia: margen en metros para considerar un split como coincidente (default 10m)
        event_id_filter: nombre del evento a filtrar (ej: "Marató", "Cadires", "10K")
    
    Retorna:
        dict con cobertura, puntos_cubiertos, puntos_faltantes, evento_elegido
    """
    print("🚀 VERSIÓN 2.3 - Con soporte para event_id_filter")
    print(f"   Puntos usuario: {puntos_usuario}")
    print(f"   Tolerancia: {tolerancia}m")
    print(f"   Filtro evento: {event_id_filter}")
    
    # Extraer distancias por evento desde la lista 'splits'
    distancias_por_evento = {}
    
    if 'splits' in catalogo:
        for split in catalogo['splits']:
            event_name = split.get('event')
            distance = split.get('distance')
            
            if event_name and distance is not None:
                if event_name not in distancias_por_evento:
                    distancias_por_evento[event_name] = set()
                distancias_por_evento[event_name].add(float(distance))
    
    print(f"   Eventos disponibles: {list(distancias_por_evento.keys())}")
    
    # Filtrar por event_id_filter si se especificó
    if event_id_filter:
        # Normalizar comparación (quitar acentos, etc. si es necesario)
        eventos_filtrados = {}
        for event_name, distancias in distancias_por_evento.items():
            # Comparación exacta o normalizada
            if event_name == event_id_filter or event_name.lower() == event_id_filter.lower():
                eventos_filtrados[event_name] = distancias
                print(f"   ✅ Filtrado al evento: {event_name}")
        
        if not eventos_filtrados:
            print(f"   ⚠️ No se encontró el evento '{event_id_filter}'. Eventos disponibles: {list(distancias_por_evento.keys())}")
            # Si no encuentra el filtro, usa todos los eventos (o podrías retornar 0%)
            eventos_filtrados = distancias_por_evento
    else:
        eventos_filtrados = distancias_por_evento
    
    # Si no hay eventos, usar todas las distancias del diccionario 'distancias'
    if not eventos_filtrados:
        # Fallback: usar el diccionario 'distancias'
        todas_distancias = set()
        if 'distancias' in catalogo:
            for key, dist in catalogo['distancias'].items():
                if isinstance(dist, (int, float)):
                    todas_distancias.add(float(dist))
        if todas_distancias:
            eventos_filtrados = {'todos': todas_distancias}
    
    mejor_evento = None
    mejor_puntuacion = 0
    mejor_puntos_cubiertos = []
    
    for event_name, distancias_evento in eventos_filtrados.items():
        print(f"\n📌 Evaluando evento: {event_name}")
        print(f"   Distancias: {len(distancias_evento)} splits")
        
        # Calcular qué puntos del usuario están cubiertos
        cubiertos = []
        for p in puntos_usuario:
            encontrado = False
            # Ordenar para mejor debug
            distancias_ordenadas = sorted(distancias_evento)
            for d in distancias_ordenadas:
                if abs(p - d) <= tolerancia:
                    cubiertos.append(p)
                    encontrado = True
                    print(f"   ✅ {p}m ≈ {d}m (dif: {abs(p-d)}m)")
                    break
            if not encontrado:
                if distancias_ordenadas:
                    cercana = min(distancias_ordenadas, key=lambda x: abs(x - p))
                    print(f"   ❌ {p}m -> más cercana: {cercana}m (dif: {abs(p-cercana)}m)")
                else:
                    print(f"   ❌ {p}m -> no hay distancias en este evento")
        
        puntuacion = len(cubiertos) / len(puntos_usuario) if puntos_usuario else 0
        print(f"   Cobertura: {puntuacion:.1%}")
        
        if puntuacion > mejor_puntuacion:
            mejor_puntuacion = puntuacion
            mejor_evento = event_name if event_name != 'todos' else None
            mejor_puntos_cubiertos = cubiertos
    
    print(f"\n📊 MEJOR RESULTADO: evento='{mejor_evento}', cobertura={mejor_puntuacion:.1%}")
    
    return {
        'cobertura': mejor_puntuacion,
        'puntos_cubiertos': mejor_puntos_cubiertos,
        'puntos_faltantes': [p for p in puntos_usuario if p not in mejor_puntos_cubiertos],
        'evento_elegido': mejor_evento,
        'splits_directos': []
    }

def buscar_mejor_combinacion_fallback(candidatas, puntos_usuario, max_carreras=3, umbral_minimo=0.70):
    """
    Encuentra la mejor combinación de hasta max_carreras que maximice cobertura.
    candidatas: lista de dict con keys: race_id, puntos_cubiertos, cobertura.
    """
    from itertools import combinations
    candidatas_validas = [c for c in candidatas if c['cobertura'] >= umbral_minimo]
    if not candidatas_validas:
        return {'seleccionadas': [], 'cobertura_total': 0.0, 'puntos_cubiertos': [], 'puntos_faltantes': puntos_usuario}
    mejor_cobertura = 0
    mejor_seleccion = []
    mejores_puntos = []
    for n in range(1, min(max_carreras, len(candidatas_validas)) + 1):
        for combo in combinations(candidatas_validas, n):
            puntos_cubiertos = set()
            for c in combo:
                puntos_cubiertos.update(c['puntos_cubiertos'])
            cobertura = len(puntos_cubiertos) / len(puntos_usuario)
            if cobertura > mejor_cobertura:
                mejor_cobertura = cobertura
                mejor_seleccion = [c['race_id'] for c in combo]
                mejores_puntos = list(puntos_cubiertos)
    return {
        'seleccionadas': mejor_seleccion,
        'cobertura_total': mejor_cobertura,
        'puntos_cubiertos': mejores_puntos,
        'puntos_faltantes': [p for p in puntos_usuario if p not in mejores_puntos]
    }


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


def splits_son_equivalentes(split1, split2, tolerancia=0.001):
    """
    Determina si dos splits representan el mismo punto comparando sus distancias.
    """
    dist1 = extract_split_distance(split1)
    dist2 = extract_split_distance(split2)
    
    if dist1 is None or dist2 is None:
        return False
    
    return abs(dist1 - dist2) < tolerancia


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
    # Normalizar splits objetivo (vienen con punto, convertir a guión bajo)
    splits_objetivo_norm = [s.replace('.', '_') for s in splits_objetivo]
    
    # Recopilar todos los splits disponibles en carreras históricas
    all_historical_splits = set()
    for carrera in carreras_historicas:
        all_historical_splits.update(carrera.get('splits', []))
    
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
        
        # Buscar por distancia en lugar de nombre exacto
        split_encontrado = None
        if split_dist is not None:
            for hist_split in all_historical_splits:
                if splits_son_equivalentes(split, hist_split):
                    split_encontrado = hist_split
                    break
        
        # Caso 1: Split existe directamente
        if split in all_historical_splits or split_encontrado is not None:
            nombre_real = split_encontrado if split_encontrado is not None else split
            result['splits_directos'].append(split)
            result['mapping'][split] = ('direct', nombre_real, split_dist)
            result['splits_finales'].append(split)
        
        # Caso 2: Split de distancia que no existe
        elif split_type == 'distance' and split_dist is not None:
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
            else:
                result['splits_imposibles'].append(split)
        
        # Caso 3: Split no numérico que no existe
        else:
            result['splits_imposibles'].append(split)
    
    return result


def validar_calidad_datos(df, splits_requeridos, 
                          umbral_min_registros=100, 
                          umbral_nulos_warning=0.0001, 
                          umbral_nulos_error=0.0001, 
                          umbral_outliers=0.05,
                          tiempo_maximo_segundos=21600):
    """
    Valida que los datos tengan la calidad suficiente para entrenar.
    
    Args:
        df: DataFrame con los datos
        splits_requeridos: lista de splits que se usarán en el modelo
        umbral_min_registros: número mínimo de registros requerido
        umbral_nulos_warning: porcentaje de nulos que genera warning
        umbral_nulos_error: porcentaje de nulos que genera error (detiene)
        umbral_outliers: porcentaje de outliers que genera warning
        tiempo_maximo_segundos: tiempo máximo razonable (default 6h)
    
    Returns:
        dict con:
        - valido: bool (True si se puede entrenar)
        - errores: lista de errores críticos
        - warnings: lista de advertencias
        - metricas: dict con métricas calculadas
    """
    import numpy as np
    
    resultados = {
        'valido': True,
        'errores': [],
        'warnings': [],
        'metricas': {
            'n_registros': len(df),
            'n_splits': len(splits_requeridos),
            'splits_analizados': splits_requeridos
        }
    }
    
    if len(df) < umbral_min_registros:
        resultados['valido'] = False
        resultados['errores'].append(
            f"Registros insuficientes: {len(df)} < {umbral_min_registros}"
        )
    else:
        resultados['metricas']['n_registros_ok'] = True
    

    for split in splits_requeridos:
        if split in df.columns:
            pct_nulos = df[split].isna().mean()
            resultados['metricas'][f'nulos_{split}'] = pct_nulos
            
            if pct_nulos > umbral_nulos_error:
                resultados['valido'] = False
                resultados['errores'].append(
                    f"{split}: {pct_nulos:.1%} nulos (máx {umbral_nulos_error:.0%})"
                )
            elif pct_nulos > umbral_nulos_warning:
                resultados['warnings'].append(
                    f"{split}: {pct_nulos:.1%} nulos (recomendado < {umbral_nulos_warning:.0%})"
                )
        else:
            # Split no existe en los datos
            resultados['valido'] = False
            resultados['errores'].append(
                f"Split '{split}' no encontrado en los datos"
            )
    
    
    for split in splits_requeridos:
        if split in df.columns:
            valores = df[split].dropna()
            if len(valores) > 0:
                # Tiempos negativos
                negativos = (valores < 0).sum()
                if negativos > 0:
                    resultados['warnings'].append(
                        f"{split}: {negativos} tiempos negativos (posible error de datos)"
                    )
                
                # Tiempos excesivamente grandes
                muy_grandes = (valores > tiempo_maximo_segundos).sum()
                if muy_grandes > 0:
                    resultados['warnings'].append(
                        f"{split}: {muy_grandes} tiempos > {tiempo_maximo_segundos/3600:.0f} horas"
                    )
    
    return resultados