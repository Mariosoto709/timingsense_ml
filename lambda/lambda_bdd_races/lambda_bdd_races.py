import boto3
import json
from datetime import datetime

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

        carreras_config = []

        for carrera_info in carreras_input:
            carrera_objetivo = carrera_info.get('nombre') or carrera_info.get('carrera')
            splits_requeridos = carrera_info.get('splits', [])
            event_id_filter = carrera_info.get('event_id_filter')
            event_std_filter = carrera_info.get('event_std_filter')
            tipo_modelo = carrera_info.get('tipo_modelo', 'interpolacion')
            training_params = carrera_info.get('training_params', {})

            # =============================================================
            # 🆕 NUEVA VALIDACIÓN 1: Verificar que splits sea una LISTA
            # =============================================================
            if not isinstance(splits_requeridos, list):
                raise ValueError(
                    f"splits debe ser una lista de strings, "
                    f"recibido: {type(splits_requeridos).__name__} = {splits_requeridos}"
                )
            
            # =============================================================
            # 🆕 NUEVA VALIDACIÓN 2: Verificar que cada split sea STRING
            # =============================================================
            for i, split in enumerate(splits_requeridos):
                if not isinstance(split, str):
                    raise ValueError(
                        f"Cada split debe ser un string, "
                        f"pero el elemento {i} es {type(split).__name__} = {split}"
                    )
            
            # =============================================================
            # 🆕 NUEVA VALIDACIÓN 3: Verificar que training_params sea DICCIONARIO
            # =============================================================
            if not isinstance(training_params, dict):
                raise ValueError(
                    f"training_params debe ser un diccionario, "
                    f"recibido: {type(training_params).__name__} = {training_params}"
                )

            # =============================================================
            # VALIDACIONES EXISTENTES
            # =============================================================
            if not carrera_objetivo or not splits_requeridos:
                continue

            # Validar tipo_modelo
            if tipo_modelo not in ['interpolacion', 'prediccion']:
                raise ValueError(f"tipo_modelo debe ser 'interpolacion' o 'prediccion', recibido: {tipo_modelo}")

            carreras_config.append({
                'carrera_objetivo': carrera_objetivo,
                'splits': splits_requeridos,
                'event_id_filter': event_id_filter,
                'event_std_filter': event_std_filter,
                'tipo_modelo': tipo_modelo,
                'training_params': training_params
            })

        if not carreras_config:
            raise ValueError("No se pudo preparar ninguna carrera para entrenamiento")

        print("✅ Configuración preparada correctamente")

        # 🆕 (OPCIONAL) Corregir el warning de deprecación
        # timestamp_unico = datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
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