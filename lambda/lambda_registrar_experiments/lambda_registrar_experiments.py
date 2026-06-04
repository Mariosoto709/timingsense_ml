import boto3
import json
from datetime import datetime
from urllib.parse import urlparse

s3_client = boto3.client('s3')
sm_client = boto3.client('sagemaker')
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

def lambda_handler(event, context):
    print("📊 Registrando en SageMaker Experiments")
    
    print(f"Evento recibido: {json.dumps(event, indent=2)}")
    
    carrera = event.get('carrera')
    timestamp = event.get('timestamp_unico')
    dataset_metadata_path = event.get('dataset_metadata_path')  # opcional
    training_job_name = event.get('training_job_name')
    
    # Solo exigimos campos obligatorios
    if not all([carrera, timestamp, training_job_name]):
        print("❌ Faltan datos obligatorios para registrar experimento")
        publicar_metrica('fallo_etapa', 1, 'Count', [
            {'Name': 'Carrera', 'Value': carrera},
            {'Name': 'Etapa', 'Value': 'RegistrarExperiments'}
        ])
        return {"status": "error", "message": "Missing required fields"}
    
    try:
        # Cargar metadata del dataset solo si se proporcionó y es válida
        dataset_metadata = None
        if dataset_metadata_path:
            try:
                parsed = urlparse(dataset_metadata_path)
                bucket = parsed.netloc
                key = parsed.path.lstrip('/')
                response = s3_client.get_object(Bucket=bucket, Key=key)
                dataset_metadata = json.loads(response['Body'].read())
                print(f"✅ Dataset metadata cargada (hash: {dataset_metadata.get('hash_dataset', 'unknown')[:8]})")
            except Exception as e:
                print(f"⚠️ No se pudo cargar metadata del dataset desde {dataset_metadata_path}: {e}")
                dataset_metadata = None
        else:
            print("ℹ️ No se proporcionó dataset_metadata_path, se registrará solo información del entrenamiento")
        
        # Obtener información del training job de SageMaker
        training_info = sm_client.describe_training_job(TrainingJobName=training_job_name)
        hyperparams = training_info.get('HyperParameters', {})
        
        # Extraer métricas finales (RMSE, MAE, etc.)
        metrics = {}
        for metric in training_info.get('FinalMetricDataList', []):
            metric_name = metric.get('MetricName')
            metric_value = metric.get('Value')
            if metric_name and metric_value is not None:
                metrics[metric_name] = metric_value
        
        print(f"✅ Métricas del entrenamiento: {metrics}")
        
        # Crear o recuperar experimento en SageMaker
        experiment_name = f"timingsense-{carrera.replace(' ', '_').replace('-', '_')}"
        run_name = f"{carrera.replace(' ', '_')}-{timestamp}"
        
        try:
            sm_client.create_experiment(
                ExperimentName=experiment_name,
                Description=f"Experiment for {carrera}"
            )
            print(f"✅ Experimento creado: {experiment_name}")
        except sm_client.exceptions.ResourceAlreadyExistsException:
            print(f"ℹ️ Experimento ya existe: {experiment_name}")
        
        # Crear trial (una ejecución específica)
        trial_name = run_name
        try:
            sm_client.create_trial(
                TrialName=trial_name,
                ExperimentName=experiment_name,
                TrialComponentName=f"{carrera}-training"
            )
            print(f"✅ Trial creado: {trial_name}")
        except sm_client.exceptions.ResourceAlreadyExistsException:
            print(f"ℹ️ Trial ya existe: {trial_name}")
        
        # Construir artefactos de entrada (dataset_metadata solo si está disponible)
        input_artifacts = {
            "hyperparameters": {"Value": json.dumps(hyperparams)}
        }
        if dataset_metadata:
            input_artifacts["dataset_metadata"] = {"Value": dataset_metadata_path}
        
        # Parámetros fijos + opcionales
        parameters = {
            "carrera": {"StringValue": carrera},
            "timestamp": {"StringValue": timestamp},
            "instance_type": {"StringValue": training_info.get('ResourceConfig', {}).get('InstanceType', '')}
        }
        if dataset_metadata:
            parameters["hash_dataset"] = {"StringValue": dataset_metadata.get('hash_dataset', '')}
            parameters["n_splits"] = {"NumberValue": dataset_metadata.get('splits_originales', 0)}
            parameters["carreras_usadas"] = {"NumberValue": dataset_metadata.get('carreras_usadas', 0)}
            parameters["cobertura_total"] = {"NumberValue": dataset_metadata.get('cobertura_total', 0)}
            parameters["tipo_seleccion"] = {"StringValue": dataset_metadata.get('tipo_seleccion', '')}
        
        # Registrar trial component con todas las métricas y artefactos
        sm_client.create_trial_component(
            TrialComponentName=f"{carrera}-training",
            DisplayName=f"Training {carrera} {timestamp}",
            InputArtifacts=input_artifacts,
            OutputArtifacts={
                "model_artifact": {"Value": training_info.get('ModelArtifacts', {}).get('S3ModelArtifacts', '')}
            },
            Parameters=parameters,
            Metrics=[
                {
                    "MetricName": name,
                    "Value": value,
                    "Timestamp": datetime.now()
                } for name, value in metrics.items()
            ]
        )
        
        print(f"✅ Trial component registrado con {len(metrics)} métricas")
        
        # Publicar métrica de éxito
        publicar_metrica('entrenamiento_exitoso', 1, 'Count', [{'Name': 'Carrera', 'Value': carrera}])
        
        return {
            "status": "success",
            "experiment_name": experiment_name,
            "trial_name": trial_name,
            "metrics": metrics
        }
        
    except Exception as e:
        print(f"❌ Error registrando experimento: {str(e)}")
        publicar_metrica('fallo_etapa', 1, 'Count', [
            {'Name': 'Carrera', 'Value': carrera},
            {'Name': 'Etapa', 'Value': 'RegistrarExperiments'}
        ])
        return {"status": "error", "message": str(e)}