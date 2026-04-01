import boto3
import json
from datetime import datetime

s3_client = boto3.client('s3')
sm_client = boto3.client('sagemaker')
sns_client = boto3.client('sns')  # 🆕 SNS para alertas
BUCKET = "timingsense-athena-output-2026"

# 🆕 NUEVA FUNCIÓN: Gatekeeper de modelos
def validar_modelo(metadata, carrera):
    """
    Valida SI el modelo es lo suficientemente bueno para registry
    """
    validaciones = metadata.get('validaciones', {})
    
    # 🆕 Extraer métricas REALES de tu metadata.json
    mae_promedio = validaciones.get('puntuacion_promedio', 999)
    cv_promedio = validaciones.get('cv_promedio', 0) 
    tasa_aprobacion = validaciones.get('tasa_aprobacion', 0)
    n_modelos = metadata.get('modelos_guardados', 0)
    
    umbrales = {
        'mae_maximo': 30.0,        # segundos
        'cv_minimo': 0.85,         # calidad mínima
        'tasa_aprobacion_min': 0.8, # 80% modelos OK
        'n_modelos_min': 3         # mínimo modelos útiles
    }
    
    print(f"🔍 Validando {carrera}: MAE={mae_promedio}, CV={cv_promedio}, Aprob={tasa_aprobacion}")
    
    # 🆕 TUS MÉTRICAS REALES vs umbrales
    mae_ok = mae_promedio <= umbrales['mae_maximo']
    cv_ok = cv_promedio >= umbrales['cv_minimo']
    tasa_ok = tasa_aprobacion >= umbrales['tasa_aprobacion_min']
    modelos_ok = n_modelos >= umbrales['n_modelos_min']
    
    aprobado = mae_ok and cv_ok and tasa_ok and modelos_ok
    
    print(f"  MAE: {mae_ok} ({mae_promedio}≤30)")
    print(f"  CV:  {cv_ok} ({cv_promedio}≥0.85)")
    print(f"  Tasa: {tasa_ok} ({tasa_aprobacion}≥0.8)")
    print(f"  Modelos: {modelos_ok} ({n_modelos}≥3)")
    
    if not aprobado:
        # 🆕 SNS ALERTA INMEDIATA
        sns_client.publish(
            TopicArn="arn:aws:sns:eu-north-1:515358862381:timingsense-errores",
            Subject=f"🚨 MODELOS RECHAZADOS - {carrera}",
            Message=f"""❌ Entrenamiento RECHAZADO

Carrera: {carrera}
MAE promedio: {mae_promedio}s (máx 30s)
Calidad CV: {cv_promedio} (mín 0.85)
Tasa aprobación: {tasa_aprobacion} (mín 0.8)
Modelos guardados: {n_modelos} (mín 3)

NO registrado en Experiments."""
        )
        return False
    
    print(f"✅ {carrera} APROBADO para Experiments")
    return True

def lambda_handler(event, context):
    """
    Registra las métricas del entrenamiento en SageMaker Experiments
    🆕 SOLO si pasa validación automática
    """
    print("📊 Registrando en SageMaker Experiments")
    print(f"Evento recibido: {json.dumps(event, indent=2)}")
    
    # Obtener datos del evento
    carrera = event.get('carrera')
    timestamp = event.get('timestamp_unico')
    model_path = event.get('model_path')  # ruta S3 del metadata.json
    
    if not carrera or not timestamp or not model_path:
        print("❌ Faltan datos para registrar experimento")
        return {"status": "error", "message": "Missing data"}
    
    try:
        # Leer metadata.json de S3
        parts = model_path.replace('s3://', '').split('/')
        bucket = parts[0]
        key = '/'.join(parts[1:])
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        metadata = json.loads(response['Body'].read())
        
        print(f"✅ Metadata cargada: {json.dumps(metadata, indent=2)[:500]}...")
        
        # 🆕 CAMBIO #1: VALIDACIÓN ANTES de Experiments
        if not validar_modelo(metadata, carrera):
            return {
                "status": "rejected",
                "message": f"Entrenamiento {carrera} rechazado por calidad insuficiente",
                "metricas": {
                    "mae_promedio": metadata.get('validaciones', {}).get('puntuacion_promedio'),
                    "cv_promedio": metadata.get('validaciones', {}).get('cv_promedio'),
                    "tasa_aprobacion": metadata.get('validaciones', {}).get('tasa_aprobacion')
                }
            }
        
        # 🆕 DESPUÉS de validación → Tu código ORIGINAL sin cambios
        experiment_name = f"timingsense-{carrera.replace(' ', '_').replace('-', '_')}"
        run_name = f"{carrera.replace(' ', '_')}-{timestamp}"
        
        # Crear experimento (tu código original)
        try:
            sm_client.create_experiment(
                ExperimentName=experiment_name,
                Description=f"Experiment for {carrera}"
            )
            print(f"✅ Experimento creado: {experiment_name}")
        except sm_client.exceptions.ResourceAlreadyExistsException:
            print(f"ℹ️ Experimento ya existe: {experiment_name}")
        
        # Crear trial (tu código original)
        try:
            trial_name = run_name
            sm_client.create_trial(
                TrialName=trial_name,
                ExperimentName=experiment_name,
                TrialComponentName=f"{carrera}-training"
            )
            print(f"✅ Trial creado: {trial_name}")
        except sm_client.exceptions.ResourceAlreadyExistsException:
            print(f"ℹ️ Trial ya existe: {trial_name}")
        
        # Registrar métricas (tu código original)
        sm_client.create_trial_component(
            TrialComponentName=f"{carrera}-training",
            DisplayName=f"Training {carrera} {timestamp}",
            InputArtifacts={
                "hyperparameters": {
                    "Value": json.dumps(metadata.get('hiperparametros', {}))
                }
            },
            OutputArtifacts={
                "metadata": {
                    "Value": f"s3://{bucket}/{key}"
                }
            },
            Parameters={
                "carrera": {"StringValue": carrera},
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
                }
            ]
        )
        
        print(f"✅ Trial component registrado")
        
        return {
            "status": "success",
            "experiment_name": experiment_name,
            "trial_name": trial_name,
            "validated": True  # 🆕 Confirmación de validación
        }
        
    except Exception as e:
        print(f"❌ Error registrando experimento: {str(e)}")
        return {"status": "error", "message": str(e)}