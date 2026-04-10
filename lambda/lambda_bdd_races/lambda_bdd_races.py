import boto3
import json
from datetime import datetime
from glue_utils import extract_split_distance, listar_carreras_disponibles, cargar_catalogo_distancias, calcular_cobertura_carrera, buscar_mejor_combinacion_fallback
import re

def lambda_handler(event, context):
    print("=" * 60)
    print("🚀 PREPARANDO CONFIGURACIÓN PARA ENTRENAMIENTO")
    print("=" * 60)
    print("📥 Evento de entrada:")
    print(json.dumps(event, indent=2))

    try:
        # Recibimos la lista de carreras
        carreras_input = event.get('carreras', [])

        if not carreras_input and event.get('carrera'):
            # Caso legacy: un solo modelo
            carreras_input = [{
                'nombre': event.get('carrera'),
                'splits': event.get('splits', []),
                'event_id_filter': event.get('event_id_filter'),
                'event_std_filter': event.get('event_std_filter'),
                'tipo_modelo': event.get('tipo_modelo', 'interpolacion'),
                'training_params': event.get('training_params', {})
            }]

        if not carreras_input:
            raise ValueError("No se especificaron carreras")

        # =============================================================
        # CONFIGURACIÓN DE SELECCIÓN (umbrales, etc.)
        # =============================================================
        config_sel = event.get('config_seleccion', {})
        min_carreras_misma = config_sel.get('min_carreras_misma', 2)
        umbral_misma = config_sel.get('umbral_splits_misma', 0.80)
        umbral_fallback = config_sel.get('umbral_splits_fallback', 0.70)
        max_fallback = config_sel.get('max_fallback', 3)
        cobertura_minima = config_sel.get('cobertura_minima_total', 0.80)

        carreras_config = []

        for carrera_info in carreras_input:
            carrera_objetivo = carrera_info.get('nombre') or carrera_info.get('carrera')
            splits_requeridos = carrera_info.get('splits', [])
            event_id_filter = carrera_info.get('event_id_filter')
            event_std_filter = carrera_info.get('event_std_filter')
            tipo_modelo = carrera_info.get('tipo_modelo', 'interpolacion')
            training_params = carrera_info.get('training_params', {})

            # =============================================================
            # VALIDACIONES
            # =============================================================
            if not isinstance(splits_requeridos, list):
                raise ValueError(
                    f"splits debe ser una lista de strings, "
                    f"recibido: {type(splits_requeridos).__name__} = {splits_requeridos}"
                )
            
            for i, split in enumerate(splits_requeridos):
                if not isinstance(split, str):
                    raise ValueError(
                        f"Cada split debe ser un string, "
                        f"pero el elemento {i} es {type(split).__name__} = {split}"
                    )
            
            if not isinstance(training_params, dict):
                raise ValueError(
                    f"training_params debe ser un diccionario, "
                    f"recibido: {type(training_params).__name__} = {training_params}"
                )

            if not carrera_objetivo or not splits_requeridos:
                continue

            if tipo_modelo not in ['interpolacion', 'prediccion']:
                raise ValueError(f"tipo_modelo debe ser 'interpolacion' o 'prediccion', recibido: {tipo_modelo}")

            # =============================================================
            # CONVERTIR SPLITS A DISTANCIAS (metros)
            # =============================================================
            puntos_usuario = []
            for split in splits_requeridos:
                dist_km = extract_split_distance(split)
                if dist_km is not None:
                    puntos_usuario.append(int(dist_km * 1000))  # km -> metros
                elif split.lower() == 'start':
                    puntos_usuario.append(0)
                elif split.lower() == 'finish':
                    puntos_usuario.append(42195)
                else:
                    raise ValueError(f"Split '{split}' no se puede convertir a distancia")

            print(f"📏 Puntos de control del usuario (metros): {puntos_usuario}")

            # =============================================================
            # EXTRAER NOMBRE_BASE Y AÑO DE LA CARRERA FUTURA
            # =============================================================
            match = re.match(r"(.*)-(\d{4})$", carrera_objetivo)
            if not match:
                raise ValueError(f"Formato inválido para carrera objetivo: {carrera_objetivo}")
            nombre_base, año_str = match.groups()
            año_objetivo = int(año_str)
            print(f"📌 Carrera objetivo: {carrera_objetivo}")
            print(f"📌 Nombre base: {nombre_base}, Año: {año_objetivo}")

            # =============================================================
            # OBTENER TODAS LAS CARRERAS DISPONIBLES DEL CATÁLOGO
            # =============================================================
            todas_carreras = listar_carreras_disponibles()
            print(f"📂 Total carreras en catálogo: {len(todas_carreras)}")

            # =============================================================
            # FASE 2: BUSCAR EDICIONES ANTERIORES (MISMA CARRERA)
            # =============================================================
            carreras_misma = []
            for race_id in todas_carreras:
                if not race_id.startswith(nombre_base):
                    continue
                m = re.search(r'-(\d{4})$', race_id)
                if not m:
                    continue
                año = int(m.group(1))
                if año >= año_objetivo:
                    continue
                catalogo = cargar_catalogo_distancias(race_id)
                if catalogo:
                    info = calcular_cobertura_carrera(catalogo, puntos_usuario)
                    # Solo añadir si supera el umbral de misma carrera
                    if info['cobertura'] >= umbral_misma:
                        carreras_misma.append({
                            'race_id': race_id,
                            'año': año,
                            'cobertura': info['cobertura'],
                            'puntos_cubiertos': info['puntos_cubiertos'],
                            'puntos_faltantes': info['puntos_faltantes'],
                            'evento_elegido': info['evento_elegido'],
                            'splits_directos': info.get('splits_directos', []),
                            'catalogo': catalogo
                        })
                        print(f"   - {race_id}: {info['cobertura']:.1%} cobertura ✅")
                    else:
                        print(f"   - {race_id}: {info['cobertura']:.1%} cobertura ❌ (no alcanza {umbral_misma:.0%})")

            # Ordenar por año descendente
            carreras_misma.sort(key=lambda x: x['año'], reverse=True)

            # Evaluar cobertura combinada de las mismas carreras
            puntos_cubiertos_misma = set()
            for c in carreras_misma:
                puntos_cubiertos_misma.update(c['puntos_cubiertos'])
            cobertura_misma = len(puntos_cubiertos_misma) / len(puntos_usuario) if puntos_usuario else 0

            print(f"\n📊 Análisis de mismas carreras:")
            print(f"   Carreras que superan umbral ({umbral_misma:.0%}): {len(carreras_misma)}")
            print(f"   Cobertura combinada: {cobertura_misma:.1%}")
            print(f"   Mínimo carreras requerido: {min_carreras_misma}")
            print(f"   Mínimo cobertura requerida: {cobertura_minima:.0%}")

            # =============================================================
            # DECISIÓN: ¿USAR MISMAS CARRERAS O BUSCAR FALLBACK?
            # =============================================================
            # Usar mismas carreras si cumplen AMBAS condiciones
            if len(carreras_misma) >= min_carreras_misma and cobertura_misma >= cobertura_minima:
                print(f"✅ Usando mismas carreras ({len(carreras_misma)} carreras, cobertura {cobertura_misma:.1%})")
                carreras_seleccionadas = carreras_misma
                tipo_seleccion = 'misma_carrera'
                cobertura_total = cobertura_misma
                puntos_faltantes = [p for p in puntos_usuario if p not in puntos_cubiertos_misma]
            
            else:
                # Explicar por qué no se usan
                if len(carreras_misma) < min_carreras_misma:
                    print(f"⚠️ Insuficientes carreras mismas: {len(carreras_misma)} < {min_carreras_misma}")
                if cobertura_misma < cobertura_minima:
                    print(f"⚠️ Cobertura insuficiente: {cobertura_misma:.1%} < {cobertura_minima:.0%}")
                
                print("🔍 Buscando otras carreras (fallback)...")

                # =============================================================
                # FASE 4: BUSCAR OTRAS CARRERAS (FALLBACK)
                # =============================================================
                candidatas_fallback = []
                for race_id in todas_carreras:
                    # Excluir las que ya son de la misma ciudad
                    if race_id.startswith(nombre_base):
                        continue
                    catalogo = cargar_catalogo_distancias(race_id)
                    if catalogo:
                        info = calcular_cobertura_carrera(catalogo, puntos_usuario)
                        if info['cobertura'] >= umbral_fallback:
                            candidatas_fallback.append({
                                'race_id': race_id,
                                'cobertura': info['cobertura'],
                                'puntos_cubiertos': info['puntos_cubiertos'],
                                'puntos_faltantes': info['puntos_faltantes'],
                                'evento_elegido': info['evento_elegido'],
                                'splits_directos': info.get('splits_directos', []),
                                'catalogo': catalogo
                            })
                            print(f"   - {race_id}: {info['cobertura']:.1%} cobertura")

                # Buscar mejor combinación
                mejor_combo = buscar_mejor_combinacion_fallback(
                    candidatas_fallback, 
                    puntos_usuario, 
                    max_carreras=max_fallback,
                    umbral_minimo=umbral_fallback
                )

                cobertura_total = mejor_combo['cobertura_total']
                puntos_faltantes = mejor_combo['puntos_faltantes']

                if cobertura_total >= cobertura_minima:
                    print(f"✅ Fallback alcanza cobertura {cobertura_total:.1%} ≥ {cobertura_minima:.0%}")
                    # Filtrar las carreras seleccionadas en la combinación
                    carreras_seleccionadas = [c for c in candidatas_fallback if c['race_id'] in mejor_combo['seleccionadas']]
                    tipo_seleccion = 'fallback'
                else:
                    # ERROR: no se alcanza cobertura mínima
                    raise ValueError(
                        f"No se puede entrenar modelo para {carrera_objetivo}\n"
                        f"Cobertura máxima alcanzable: {cobertura_total:.1%}\n"
                        f"Mínimo requerido: {cobertura_minima:.0%}\n"
                        f"Puntos no cubiertos: {puntos_faltantes}"
                    )

            # =============================================================
            # PREPARAR LISTA DE CARRERAS HISTÓRICAS CON SUS SPLITS
            # =============================================================
            carreras_historicas_detalle = []
            for c in carreras_seleccionadas:
                # Extraer la lista de splits normalizados del evento elegido
                evento_elegido = c['evento_elegido']
                splits_evento = []
                for event in c['catalogo'].get('events', []):
                    if event.get('name') == evento_elegido:
                        for split in event.get('splits', []):
                            nombre_norm = split.get('normalized_name')
                            if nombre_norm:
                                splits_evento.append(nombre_norm)
                        break
                
                carreras_historicas_detalle.append({
                    'race_id': c['race_id'],
                    'evento': evento_elegido,
                    'splits': splits_evento,
                    'cobertura': c['cobertura']
                })

            print(f"\n📋 Carreras seleccionadas ({tipo_seleccion}):")
            for c in carreras_historicas_detalle:
                print(f"   - {c['race_id']} (evento: {c['evento']}, cobertura: {c['cobertura']:.1%})")
            print(f"   Cobertura total: {cobertura_total:.1%}")
            print(f"   Puntos faltantes: {puntos_faltantes}")

            # =============================================================
            # CONSTRUIR CONFIGURACIÓN FINAL
            # =============================================================
            carreras_config.append({
                'carrera_objetivo': carrera_objetivo,
                'splits': splits_requeridos,
                'puntos_usuario': puntos_usuario,
                'carreras_historicas_detalle': carreras_historicas_detalle,
                'tipo_seleccion': tipo_seleccion,
                'cobertura_total': cobertura_total,
                'puntos_faltantes': puntos_faltantes,
                'event_id_filter': event_id_filter,
                'event_std_filter': event_std_filter,
                'tipo_modelo': tipo_modelo,
                'training_params': training_params,
                'umbrales_usados': {
                    'min_carreras_misma': min_carreras_misma,
                    'umbral_splits_misma': umbral_misma,
                    'umbral_splits_fallback': umbral_fallback,
                    'cobertura_minima_total': cobertura_minima,
                    'max_fallback': max_fallback
                }
            })

        if not carreras_config:
            raise ValueError("No se pudo preparar ninguna carrera para entrenamiento")

        print("✅ Configuración preparada correctamente")

        timestamp_unico = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

        salida = {
            "carreras_config": carreras_config,
            "num_modelos": len(carreras_config),
            "generated_at": timestamp_unico,
            "timestamp_unico": timestamp_unico
        }

        print("📤 Salida de la Lambda:")
        print(json.dumps(salida, indent=2))

        return salida

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        raise e