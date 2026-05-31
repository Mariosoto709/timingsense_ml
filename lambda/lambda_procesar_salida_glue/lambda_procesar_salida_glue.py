import boto3
import json
import os
from datetime import datetime

s3 = boto3.client('s3')
BUCKET = os.environ.get('S3_OUTPUT_BUCKET', 'timingsense-athena-output-2026')  # ← MEJORA 1


def publicar_metrica_metadata_empty(carrera):
    """Publica métrica cuando no se encuentra metadata.json"""
    try:
        cloudwatch = boto3.client('cloudwatch', region_name='eu-north-1')
        sns = boto3.client('sns', region_name='eu-north-1')
        
        cloudwatch.put_metric_data(
            Namespace='timingsense/S3',
            MetricData=[{
                'MetricName': 'MetadataEmpty',
                'Value': 1,
                'Unit': 'Count',
                'Timestamp': datetime.now(),
                'Dimensions': [{'Name': 'Carrera', 'Value': carrera}]
            }]
        )
        print(f"📊 Métrica publicada: S3/MetadataEmpty=1 para {carrera}")
        
        sns.publish(
            TopicArn="arn:aws:sns:eu-north-1:515358862381:timingsense-errores",
            Subject=f"🚨 CRÍTICO: Metadata.json no encontrado - {carrera}",
            Message=f"""No se encontró metadata.json para {carrera}

Impacto: Step Function no puede continuar.
Posibles causas:
- Glue job falló silenciosamente
- S3 path incorrecto
- Permisos S3 insuficientes

Acción inmediata: Revisar logs de Glue y S3.
"""
        )
        print(f"📧 Alerta SNS enviada")
    except Exception as e:
        print(f"⚠️ Error publicando métrica: {e}")


def publicar_metrica_datos_insuficientes(carrera, total_filas):
    """Publica métrica de datos insuficientes a CloudWatch"""
    try:
        cloudwatch = boto3.client('cloudwatch', region_name='eu-north-1')
        sns = boto3.client('sns', region_name='eu-north-1')
        
        datos_insuficientes = 1 if total_filas < 100 else 0
        
        cloudwatch.put_metric_data(
            Namespace='timingsense/Glue',
            MetricData=[{
                'MetricName': 'DatosInsuficientes',
                'Value': datos_insuficientes,
                'Unit': 'Count',
                'Timestamp': datetime.now(),
                'Dimensions': [{'Name': 'Carrera', 'Value': carrera}]
            }]
        )
        print(f"📊 Métrica CloudWatch: DatosInsuficientes={datos_insuficientes} ({total_filas} filas)")
        
        if datos_insuficientes == 1:
            sns.publish(
                TopicArn="arn:aws:sns:eu-north-1:515358862381:timingsense-errores",
                Subject=f"⚠️ DATOS INSUFICIENTES - {carrera}",
                Message=f"Volumen mínimo no alcanzado para {carrera}\n\nTotal filas: {total_filas}\nMínimo requerido: 100\n\nImpacto: Modelos XGBoost sobreajustados."
            )
            print(f"📧 Alerta SNS enviada")
    except Exception as e:
        print(f"⚠️ Error publicando métrica: {e}")


# =============================================================
# HANDLER PRINCIPAL
# =============================================================
def lambda_handler(event, context):
    print("🔍 Procesando salida de Glue")
    print(f"Evento recibido: {json.dumps(event, indent=2)}")
    
    glue_job_run_id = event.get('glue_job_run_id')
    carrera = event.get('carrera_objetivo', 'desconocida')
    evento_objetivo = event.get('evento_objetivo')  # ← MEJORA 2
    generated_at = event.get('generated_at')
    
    if not generated_at:
        print("❌ No hay generated_at en el evento")
        publicar_metrica_metadata_empty(carrera)
        return {"modelos": []}
    
    # Construir la carpeta correcta
    timestamp_archivo = generated_at.replace('-', '_')
    key = f"debug/salida_step_{timestamp_archivo}.json"
    
    print(f"📄 Buscando archivo: {key}")
    
    try:
        response = s3.get_object(Bucket=BUCKET, Key=key)
        contenido = json.loads(response['Body'].read())
        print(f"✅ Archivo encontrado")
        
        modelos = contenido.get("modelos", [])
        for modelo in modelos:
            carrera_modelo = modelo.get("carrera", carrera)
            carpeta_correcta = f"{carrera_modelo}-{generated_at}"
            modelo["carpeta_modelo"] = carpeta_correcta
            modelo["data_s3_path"] = f"s3://{BUCKET}/modelos/{carpeta_correcta}/data/"
            modelo["tabla_generada"] = f"modelo_{carrera_modelo.replace('-', '_')}_{generated_at.replace('-', '_')}"
            modelo["evento_objetivo"] = evento_objetivo  # ← MEJORA 2
        
        # Publicar métrica de datos insuficientes
        total_filas = contenido.get('total_filas', 0)
        if total_filas == 0:
            total_filas = contenido.get('num_registros', 0)
        publicar_metrica_datos_insuficientes(carrera, total_filas)
        
        return {"modelos": modelos}
        
    except Exception as e:
        print(f"❌ Error leyendo archivo específico: {str(e)}")
        print("⚠️ Buscando el archivo más reciente como fallback...")
        
        try:
            response = s3.list_objects_v2(
                Bucket=BUCKET,
                Prefix="debug/salida_step_",
                MaxKeys=10
            )
            
            if 'Contents' not in response:
                print("❌ No se encontraron archivos")
                publicar_metrica_metadata_empty(carrera)
                publicar_metrica_datos_insuficientes(carrera, 0)
                
                return {
                    "modelos": [
                        {
                            "carrera": carrera,
                            "evento_objetivo": evento_objetivo,  # ← MEJORA 2
                            "carpeta_modelo": f"{carrera}-{generated_at}",
                            "data_s3_path": f"s3://{BUCKET}/modelos/{carrera}-{generated_at}/data/",
                            "tabla_generada": f"modelo_{carrera.replace('-', '_')}_{generated_at.replace('-', '_')}",
                            "splits": 12,
                            "carreras_usadas": 1
                        }
                    ]
                }
            
            archivos = sorted(response['Contents'], 
                             key=lambda x: x['LastModified'], 
                             reverse=True)
            
            archivo_reciente = archivos[0]
            key = archivo_reciente['Key']
            
            print(f"📄 Usando archivo más reciente: {key}")
            
            response = s3.get_object(Bucket=BUCKET, Key=key)
            contenido = json.loads(response['Body'].read())
            
            modelos = contenido.get("modelos", [])
            for modelo in modelos:
                carpeta_correcta = f"{carrera}-{generated_at}"
                modelo["carpeta_modelo"] = carpeta_correcta
                modelo["data_s3_path"] = f"s3://{BUCKET}/modelos/{carpeta_correcta}/data/"
                modelo["tabla_generada"] = f"modelo_{carrera.replace('-', '_')}_{generated_at.replace('-', '_')}"
                modelo["evento_objetivo"] = evento_objetivo  
            
            total_filas = contenido.get('total_filas', 0)
            publicar_metrica_datos_insuficientes(carrera, total_filas)
            
            return {"modelos": modelos}
            
        except Exception as e2:
            print(f"❌ Error en fallback: {str(e2)}")
            publicar_metrica_metadata_empty(carrera)
            publicar_metrica_datos_insuficientes(carrera, 0)
            
            return {
                "modelos": [
                    {
                        "carrera": carrera,
                        "evento_objetivo": evento_objetivo,  
                        "carpeta_modelo": f"{carrera}-{generated_at}",
                        "data_s3_path": f"s3://{BUCKET}/modelos/{carrera}-{generated_at}/data/",
                        "tabla_generada": f"modelo_{carrera.replace('-', '_')}_{generated_at.replace('-', '_')}",
                        "splits": 12,
                        "carreras_usadas": 1
                    }
                ]
            }