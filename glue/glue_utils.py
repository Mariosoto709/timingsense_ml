# glue/glue_utils.py
"""
Funciones auxiliares para el job Glue (sin dependencias de AWS)
Estas funciones se pueden testear localmente sin necesidad de AWS.
"""


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
    
    # =============================================================
    # 1. Verificar número mínimo de registros
    # =============================================================
    if len(df) < umbral_min_registros:
        resultados['valido'] = False
        resultados['errores'].append(
            f"Registros insuficientes: {len(df)} < {umbral_min_registros}"
        )
    else:
        resultados['metricas']['n_registros_ok'] = True
    
    # =============================================================
    # 2. Verificar nulos por split
    # =============================================================
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
    
    # =============================================================
    # 4. Verificar rangos razonables (tiempos deben ser positivos)
    # =============================================================
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