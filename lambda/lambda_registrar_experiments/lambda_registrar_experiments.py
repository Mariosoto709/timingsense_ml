import boto3
import json
from datetime import datetime

s3_client = boto3.client('s3')
sm_client = boto3.client('sagemaker')
sns_client = boto3.client('sns')
cloudwatch = boto3.client('cloudwatch')
BUCKET = "timingsense-athena-output-2026"

# ============================================================
# FUNCIÓN DE VALIDACIÓN POR TIPO DE MODELO
# ============================================================

def validar_modelo(metadata, carrera, evento, tipo_modelo):
    """
    Valida si el modelo es lo suficientemente bueno para registry.
    Umbrales diferentes para interpolación y predicción.
    """
    validaciones = metadata.get('validaciones', {})
    
    # Extraer métricas reales
    calidad_promedio = validaciones.get('puntuacion_promedio', 0)
    cv_promedio = validaciones.get('cv_promedio', 1.0)
    tasa_aprobacion = validaciones.get('tasa_aprobacion', 0)
    n_modelos = metadata.get('modelos_guardados', 0)
    mejora_promedio = validaciones.get('mejora_promedio_sobre_naive', 0)
    
    # Umbrales según tipo de modelo
    if tipo_modelo == 'interpolacion':
        umbrales = {
            'calidad_minima': 70.0,      # Calidad ≥ 70
            'cv_maximo': 0.5,            # CV ≤ 0.5
            'tasa_aprobacion_min': 0.6,  # 60% de modelos aprobados
            'n_modelos_min': 2,          # Mínimo 2 modelos útiles
            'mejora_minima': 0.1         # Mejora mínima del 10% sobre naïve
        }
    else:  # prediccion
        umbrales = {
            'calidad_minima': 80.0,      # Calidad ≥ 80 (más exigente)
            'cv_maximo': 0.4,            # CV ≤ 0.4 (más estricto)
            'tasa_aprobacion_min': 0.7,  # 70% de modelos aprobados
            'n_modelos_min': 1,          # Al menos 1 modelo útil
            'mejora_minima': 0.15        # Mejora mínima del 15% sobre naïve
        }
    
    print(f"🔍 Validando {carrera}/{evento} ({tipo_modelo}):")
    print(f"   Calidad promedio: {calidad_promedio:.1f} (mín {umbrales['calidad_minima']})")
    print(f"   CV promedio: {cv_promedio:.3f} (máx {umbrales['cv_maximo']})")
    print(f"   Tasa aprobación: {tasa_aprobacion:.1%} (mín {umbrales['tasa_aprobacion_min']:.0%})")
    print(f"   Modelos guardados: {n_modelos} (mín {umbrales['n_modelos_min']})")
    print(f"   Mejora promedio: {mejora_promedio:.1%} (mín {umbrales['mejora_minima']:.0%})")
    
    # Verificar cada condición
    calidad_ok = calidad_promedio >= umbrales['calidad_minima']
    cv_ok = cv_promedio <= umbrales['cv_maximo']
    tasa_ok = tasa_aprobacion >= umbrales['tasa_aprobacion_min']
    modelos_ok = n_modelos >= umbrales['n_modelos_min']
    mejora_ok = mejora_promedio >= umbrales['mejora_minima']
    
    aprobado = calidad_ok and cv_ok and tasa_ok and modelos_ok and mejora_ok
    
    print(f"\n   Resultados:")
    print(f"   - Calidad: {'✅' if calidad_ok else '❌'} ({calidad_promedio:.1f} ≥ {umbrales['calidad_minima']})")
    print(f"   - CV: {'✅' if cv_ok else '❌'} ({cv_promedio:.3f} ≤ {umbrales['cv_maximo']})")
    print(f"   - Tasa aprobación: {'✅' if tasa_ok else '❌'} ({tasa_aprobacion:.1%} ≥ {umbrales['tasa_aprobacion_min']:.0%})")
    print(f"   - Modelos: {'✅' if modelos_ok else '❌'} ({n_modelos} ≥ {umbrales['n_modelos_min']})")
    print(f"   - Mejora: {'✅' if mejora_ok else '❌'} ({mejora_promedio:.1%} ≥ {umbrales['mejora_minima']:.0%})")
    
    if not aprobado:
        # Enviar alerta SNS
        sns_client.publish(
            TopicArn="arn:aws:sns:eu-north-1:515358862381:timingsense-errores",
            Subject=f"🚨 MODELOS RECHAZADOS - {carrera}/{evento} ({tipo_modelo})",
            Message=f"""❌ Entrenamiento RECHAZADO

Carrera: {carrera}
Evento: {evento}
Tipo: {tipo_modelo}

Métricas obtenidas vs requeridas:
- Calidad promedio: {calidad_promedio:.1f}/100 (mín {umbrales['calidad_minima']})
- CV promedio: {cv_promedio:.3f} (máx {umbrales['cv_maximo']})
- Tasa aprobación: {tasa_aprobacion:.1%} (mín {umbrales['tasa_aprobacion_min']:.0%})
- Modelos guardados: {n_modelos} (mín {umbrales['n_modelos_min']})
- Mejora sobre naïve: {mejora_promedio:.1%} (mín {umbrales['mejora_minima']:.0%})

NO registrado en Experiments.
"""
        )
        return False
    
    print(f"✅ {carrera}/{evento} APROBADO para Experiments")
    return True


# ============================================================
# FUNCIÓN PARA PUBLICAR MÉTRICAS A CLOUDWATCH
# ============================================================

def publicar_metricas_cloudwatch(carrera, evento, tipo_modelo, metadata, aprobado):
    """Publica métricas a CloudWatch"""
    try:
        validaciones = metadata.get('validaciones', {})
        
        metricas = [
            {
                'MetricName': 'ModelosGenerados',
                'Value': 1 if metadata.get('modelos_guardados', 0) > 0 else 0,
                'Unit': 'Count',
                'Timestamp': datetime.now(),
                'Dimensions': [
                    {'Name': 'Carrera', 'Value': carrera},
                    {'Name': 'Evento', 'Value': evento},
                    {'Name': 'TipoModelo', 'Value': tipo_modelo}
                ]
            },
            {
                'MetricName': 'CalidadPromedio',
                'Value': validaciones.get('puntuacion_promedio', 0),
                'Unit': 'None',
                'Timestamp': datetime.now(),
                'Dimensions': [
                    {'Name': 'Carrera', 'Value': carrera},
                    {'Name': 'Evento', 'Value': evento},
                    {'Name': 'TipoModelo', 'Value': tipo_modelo}
                ]
            },
            {
                'MetricName': 'TasaAprobacion',
                'Value': validaciones.get('tasa_aprobacion', 0) * 100,
                'Unit': 'Percent',
                'Timestamp': datetime.now(),
                'Dimensions': [
                    {'Name': 'Carrera', 'Value': carrera},
                    {'Name': 'Evento', 'Value': evento},
                    {'Name': 'TipoModelo', 'Value': tipo_modelo}
                ]
            },
            {
                'MetricName': 'CVPromedio',
                'Value': validaciones.get('cv_promedio', 1.0),
                'Unit': 'None',
                'Timestamp': datetime.now(),
                'Dimensions': [
                    {'Name': 'Carrera', 'Value': carrera},
                    {'Name': 'Evento', 'Value': evento},
                    {'Name': 'TipoModelo', 'Value': tipo_modelo}
                ]
            },
            {
                'MetricName': 'ModelosAprobados',
                'Value': metadata.get('modelos_guardados', 0),
                'Unit': 'Count',
                'Timestamp': datetime.now(),
                'Dimensions': [
                    {'Name': 'Carrera', 'Value': carrera},
                    {'Name': 'Evento', 'Value': evento},
                    {'Name': 'TipoModelo', 'Value': tipo_modelo}
                ]
            }
        ]
        
        # Si el modelo fue aprobado, publicar métrica adicional
        if aprobado:
            metricas.append({
                'MetricName': 'ModeloAprobado',
                'Value': 1,
                'Unit': 'Count',
                'Timestamp': datetime.now(),
                'Dimensions': [
                    {'Name': 'Carrera', 'Value': carrera},
                    {'Name': 'Evento', 'Value': evento},
                    {'Name': 'TipoModelo', 'Value': tipo_modelo}
                ]
            })
        
        cloudwatch.put_metric_data(
            Namespace='timingsense/ML',
            MetricData=metricas
        )
        print(f"📊 Métricas CloudWatch publicadas para {carrera}/{evento}")
        
    except Exception as e:
        print(f"⚠️ Error publicando métricas a CloudWatch: {e}")


# ============================================================
# HANDLER PRINCIPAL
# ============================================================

def lambda_handler(event, context):
    """
    Registra las métricas del entrenamiento en SageMaker Experiments
    SOLO si pasa validación automática.
    """
    print("="*60)
    print("📊 REGISTRANDO EN SAGEMAKER EXPERIMENTS")
    print("="*60)
    print(f"Evento recibido: {json.dumps(event, indent=2)}")
    
    # Obtener datos del evento
    carrera = event.get('carrera')
    evento_objetivo = event.get('evento_objetivo')
    tipo_modelo = event.get('tipo_modelo', 'interpolacion')
    timestamp = event.get('timestamp_unico')
    model_path = event.get('model_path')
    
    if not carrera or not evento_objetivo or not timestamp or not model_path:
        print("❌ Faltan datos para registrar experimento")
        return {
            "status": "error", 
            "message": "Missing data: carrera, evento_objetivo, timestamp_unico or model_path"
        }
    
    try:
        # Leer metadata.json de S3
        parts = model_path.replace('s3://', '').split('/')
        bucket = parts[0]
        key = '/'.join(parts[1:])
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        metadata = json.loads(response['Body'].read())
        
        print(f"✅ Metadata cargada de: s3://{bucket}/{key}")
        print(f"   Modelos guardados: {metadata.get('modelos_guardados', 0)}")
        print(f"   Calidad promedio: {metadata.get('validaciones', {}).get('puntuacion_promedio', 0)}")
        
        # =============================================================
        # VALIDACIÓN ANTES DE EXPERIMENTS
        # =============================================================
        aprobado = validar_modelo(metadata, carrera, evento_objetivo, tipo_modelo)
        
        # Publicar métricas a CloudWatch (siempre, para monitorización)
        publicar_metricas_cloudwatch(carrera, evento_objetivo, tipo_modelo, metadata, aprobado)
        
        if not aprobado:
            return {
                "status": "rejected",
                "message": f"Entrenamiento {carrera}/{evento_objetivo} ({tipo_modelo}) rechazado por calidad insuficiente",
                "metricas": {
                    "calidad_promedio": metadata.get('validaciones', {}).get('puntuacion_promedio', 0),
                    "cv_promedio": metadata.get('validaciones', {}).get('cv_promedio', 0),
                    "tasa_aprobacion": metadata.get('validaciones', {}).get('tasa_aprobacion', 0),
                    "modelos_guardados": metadata.get('modelos_guardados', 0),
                    "tipo_modelo": tipo_modelo
                }
            }
        
        # =============================================================
        # REGISTRAR EN SAGEMAKER EXPERIMENTS
        # =============================================================
        
        # Nombre del experimento incluye tipo_modelo y evento
        experiment_name = f"timingsense-{tipo_modelo}-{carrera.replace(' ', '_').replace('-', '_')}_{evento_objetivo.replace(' ', '_')}"
        run_name = f"{carrera.replace(' ', '_')}_{evento_objetivo.replace(' ', '_')}-{timestamp}"
        
        # Crear experimento
        try:
            sm_client.create_experiment(
                ExperimentName=experiment_name,
                Description=f"Experiment for {carrera} - {evento_objetivo} ({tipo_modelo})"
            )
            print(f"✅ Experimento creado: {experiment_name}")
        except sm_client.exceptions.ResourceAlreadyExistsException:
            print(f"ℹ️ Experimento ya existe: {experiment_name}")
        
        # Crear trial
        try:
            trial_name = run_name
            sm_client.create_trial(
                TrialName=trial_name,
                ExperimentName=experiment_name,
                TrialComponentName=f"{carrera}-{evento_objetivo}-training"
            )
            print(f"✅ Trial creado: {trial_name}")
        except sm_client.exceptions.ResourceAlreadyExistsException:
            print(f"ℹ️ Trial ya existe: {trial_name}")
        
        # Registrar métricas
        sm_client.create_trial_component(
            TrialComponentName=f"{carrera}-{evento_objetivo}-training",
            DisplayName=f"Training {carrera} {evento_objetivo} {timestamp}",
            InputArtifacts={
                "hyperparameters": {
                    "Value": json.dumps(metadata.get('hiperparametros', {}))
                },
                "tipo_modelo": {
                    "Value": tipo_modelo
                }
            },
            OutputArtifacts={
                "metadata": {
                    "Value": f"s3://{bucket}/{key}"
                }
            },
            Parameters={
                "carrera": {"StringValue": carrera},
                "evento": {"StringValue": evento_objetivo},
                "tipo_modelo": {"StringValue": tipo_modelo},
                "timestamp": {"StringValue": timestamp},
                "n_splits": {"NumberValue": len(metadata.get('splits', []))},
                "n_modelos_guardados": {"NumberValue": metadata.get('modelos_guardados', 0)}
            },
            Metrics=[
                {
                    "MetricName": "tasa_aprobacion",
                    "Value": metadata.get('validaciones', {}).get('tasa_aprobacion', 0),
                    "Timestamp": datetime.now()
                },
                {
                    "MetricName": "calidad_promedio",
                    "Value": metadata.get('validaciones', {}).get('puntuacion_promedio', 0),
                    "Timestamp": datetime.now()
                },
                {
                    "MetricName": "cv_promedio",
                    "Value": metadata.get('validaciones', {}).get('cv_promedio', 0),
                    "Timestamp": datetime.now()
                },
                {
                    "MetricName": "mejora_promedio",
                    "Value": metadata.get('validaciones', {}).get('mejora_promedio_sobre_naive', 0),
                    "Timestamp": datetime.now()
                },
                {
                    "MetricName": "modelos_aprobados",
                    "Value": metadata.get('validaciones', {}).get('aprobados', 0),
                    "Timestamp": datetime.now()
                },
                {
                    "MetricName": "modelos_rechazados",
                    "Value": metadata.get('validaciones', {}).get('rechazados', 0),
                    "Timestamp": datetime.now()
                }
            ]
        )
        
        print(f"✅ Trial component registrado para {carrera}/{evento_objetivo}")
        
        # Enviar notificación de éxito
        sns_client.publish(
            TopicArn="arn:aws:sns:eu-north-1:515358862381:timingsense-exito",
            Subject=f"✅ MODELOS APROBADOS - {carrera}/{evento_objetivo} ({tipo_modelo})",
            Message=f"""✅ Entrenamiento APROBADO y registrado

Carrera: {carrera}
Evento: {evento_objetivo}
Tipo: {tipo_modelo}
Timestamp: {timestamp}

Métricas:
- Calidad promedio: {metadata.get('validaciones', {}).get('puntuacion_promedio', 0):.1f}/100
- Modelos guardados: {metadata.get('modelos_guardados', 0)}
- Tasa aprobación: {metadata.get('validaciones', {}).get('tasa_aprobacion', 0):.1%}

Registrado en Experiments: {experiment_name}
"""
        )
        
        return {
            "status": "success",
            "experiment_name": experiment_name,
            "trial_name": trial_name,
            "validated": True,
            "tipo_modelo": tipo_modelo,
            "carrera": carrera,
            "evento": evento_objetivo
        }
        
    except Exception as e:
        print(f"❌ Error registrando experimento: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Enviar alerta de error
        try:
            sns_client.publish(
                TopicArn="arn:aws:sns:eu-north-1:515358862381:timingsense-errores",
                Subject=f"❌ ERROR REGISTRANDO EXPERIMENTO - {carrera}/{evento_objetivo}",
                Message=f"""Error registrando experimento en SageMaker Experiments

Carrera: {carrera}
Evento: {evento_objetivo}
Tipo: {tipo_modelo}
Error: {str(e)}

Revisar logs de Lambda para más detalles.
"""
            )
        except:
            pass
        
        return {"status": "error", "message": str(e)}