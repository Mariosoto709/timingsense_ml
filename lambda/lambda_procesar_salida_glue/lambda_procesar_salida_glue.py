import boto3
import json
import os
import re
from datetime import datetime

s3 = boto3.client('s3')
BUCKET = os.environ.get('S3_OUTPUT_BUCKET', 'timingsense-athena-output-2026')

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
# FUNCIONES AUXILIARES
# =============================================================

def extraer_timestamp_de_key(key):
    """Extrae el timestamp del nombre del archivo debug/salida_step_YYYYMMDD_HHMMSS.json"""
    match = re.search(r'salida_step_(\d{8}_\d{6})\.json', key)
    if match:
        timestamp = match.group(1).replace('_', '-')
        return timestamp
    return None

def extraer_carpeta_de_ruta(ruta_s3):
    """Extrae el nombre de la carpeta modelo de una ruta S3"""
    match = re.search(r'/modelos/([^/]+)/data/', ruta_s3)
    if match:
        return match.group(1)
    return None

# =============================================================
# HANDLER PRINCIPAL
# =============================================================

def lambda_handler(event, context):
    print("=" * 60)
    print("🔍 PROCESANDO SALIDA DE GLUE")
    print("=" * 60)
    print(f"Evento recibido: {json.dumps(event, indent=2)}")
    
    carrera = event.get('carrera_objetivo', 'desconocida')
    evento_objetivo = event.get('evento_objetivo')
    generated_at = event.get('generated_at')
    
    if not generated_at:
        print("❌ No hay generated_at en el evento")
        publicar_metrica_metadata_empty(carrera)
        publicar_metrica('fallo_etapa', 1, 'Count', [{'Name': 'Carrera', 'Value': carrera}, {'Name': 'Etapa', 'Value': 'ProcesarSalidaGlue'}])
        return {"modelos": []}
    
    print(f"📌 Carrera: {carrera}")
    print(f"📌 Evento: {evento_objetivo}")
    print(f"📌 generated_at recibido: {generated_at}")
    
    timestamp_archivo = generated_at.replace('-', '_')
    key = f"debug/salida_step_{timestamp_archivo}.json"
    
    print(f"\n📄 Buscando archivo específico: {key}")
    
    try:
        response = s3.get_object(Bucket=BUCKET, Key=key)
        contenido = json.loads(response['Body'].read())
        print(f"✅ Archivo específico encontrado")
        
        modelos = contenido.get("modelos", [])
        for modelo in modelos:
            # Rutas S3
            if "data_s3_path" in modelo and modelo["data_s3_path"]:
                carpeta_existente = extraer_carpeta_de_ruta(modelo["data_s3_path"])
                if carpeta_existente:
                    modelo["carpeta_modelo"] = carpeta_existente
                    print(f"   📁 Usando carpeta existente: {carpeta_existente}")
                else:
                    modelo["carpeta_modelo"] = f"{carrera}-{generated_at}"
                    modelo["data_s3_path"] = f"s3://{BUCKET}/modelos/{modelo['carpeta_modelo']}/data/"
            else:
                modelo["carpeta_modelo"] = f"{carrera}-{generated_at}"
                modelo["data_s3_path"] = f"s3://{BUCKET}/modelos/{modelo['carpeta_modelo']}/data/"
            
            # Respetar tabla_generada si ya viene del Glue (cache hit)
            if "tabla_generada" in modelo and modelo["tabla_generada"]:
                print(f"   ♻️ Reutilizando tabla existente: {modelo['tabla_generada']}")
            else:
                modelo["tabla_generada"] = f"modelo_{carrera.replace('-', '_')}_{generated_at.replace('-', '_')}"
                print(f"   🆕 Generando nueva tabla: {modelo['tabla_generada']}")
            
            # Añadir dataset_metadata_path
            if "hash_dataset" in modelo and modelo["hash_dataset"]:
                modelo["dataset_metadata_path"] = f"s3://{BUCKET}/datasets/{modelo['hash_dataset']}/metadata.json"
                print(f"   📄 Dataset metadata path: {modelo['dataset_metadata_path']}")
            else:
                modelo["dataset_metadata_path"] = None
                print(f"   ⚠️ No se encontró hash_dataset para este modelo")
            
            modelo["evento_objetivo"] = evento_objetivo
        
        total_filas = contenido.get('total_filas', 0) or contenido.get('num_registros', 0)
        publicar_metrica_datos_insuficientes(carrera, total_filas)
        
        # Publicar cobertura_total si hay modelos
        if modelos:
            cobertura = modelos[0].get('cobertura_total', 0)
            publicar_metrica('cobertura_total', cobertura, 'None', [{'Name': 'Carrera', 'Value': carrera}])
        
        print(f"\n✅ Salida generada con {len(modelos)} modelo(s)")
        for m in modelos:
            print(f"   - {m.get('carrera')}: {m.get('data_s3_path')} -> tabla {m.get('tabla_generada')}")
        
        return {"modelos": modelos}
    
    except Exception as e:
        print(f"⚠️ Error leyendo archivo específico: {str(e)}")
        publicar_metrica('fallo_etapa', 1, 'Count', [{'Name': 'Carrera', 'Value': carrera}, {'Name': 'Etapa', 'Value': 'ProcesarSalidaGlue'}])
        
        print("⚠️ Buscando el archivo más reciente como fallback...")
        
        try:
            response = s3.list_objects_v2(
                Bucket=BUCKET,
                Prefix="debug/salida_step_",
                MaxKeys=10
            )
            
            if 'Contents' not in response:
                print("❌ No se encontraron archivos en fallback")
                publicar_metrica_metadata_empty(carrera)
                publicar_metrica_datos_insuficientes(carrera, 0)
                return {
                    "modelos": [{
                        "carrera": carrera,
                        "evento_objetivo": evento_objetivo,
                        "carpeta_modelo": f"{carrera}-{generated_at}",
                        "data_s3_path": f"s3://{BUCKET}/modelos/{carrera}-{generated_at}/data/",
                        "tabla_generada": f"modelo_{carrera.replace('-', '_')}_{generated_at.replace('-', '_')}",
                        "dataset_metadata_path": None,
                        "splits": 12,
                        "carreras_usadas": 1
                    }]
                }
            
            archivos = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
            archivo_reciente = archivos[0]
            key = archivo_reciente['Key']
            
            print(f"📄 Usando archivo más reciente: {key}")
            print(f"   Última modificación: {archivo_reciente['LastModified']}")
            
            timestamp_real = extraer_timestamp_de_key(key) or generated_at
            print(f"✅ Timestamp real del archivo: {timestamp_real}")
            
            response = s3.get_object(Bucket=BUCKET, Key=key)
            contenido = json.loads(response['Body'].read())
            
            modelos = contenido.get("modelos", [])
            for modelo in modelos:
                # Reconstruir ruta
                if "data_s3_path" in modelo and modelo["data_s3_path"]:
                    carpeta_existente = extraer_carpeta_de_ruta(modelo["data_s3_path"])
                    if carpeta_existente:
                        modelo["carpeta_modelo"] = carpeta_existente
                    else:
                        modelo["carpeta_modelo"] = f"{carrera}-{timestamp_real}"
                        modelo["data_s3_path"] = f"s3://{BUCKET}/modelos/{modelo['carpeta_modelo']}/data/"
                else:
                    modelo["carpeta_modelo"] = f"{carrera}-{timestamp_real}"
                    modelo["data_s3_path"] = f"s3://{BUCKET}/modelos/{modelo['carpeta_modelo']}/data/"
                
                # Respetar tabla_generada si existe
                if "tabla_generada" in modelo and modelo["tabla_generada"]:
                    print(f"   ♻️ Reutilizando tabla existente (fallback): {modelo['tabla_generada']}")
                else:
                    modelo["tabla_generada"] = f"modelo_{carrera.replace('-', '_')}_{timestamp_real.replace('-', '_')}"
                    print(f"   🆕 Generando nueva tabla (fallback): {modelo['tabla_generada']}")
                
                # Añadir dataset_metadata_path en fallback
                if "hash_dataset" in modelo and modelo["hash_dataset"]:
                    modelo["dataset_metadata_path"] = f"s3://{BUCKET}/datasets/{modelo['hash_dataset']}/metadata.json"
                    print(f"   📄 Dataset metadata path (fallback): {modelo['dataset_metadata_path']}")
                else:
                    modelo["dataset_metadata_path"] = None
                    print(f"   ⚠️ No se encontró hash_dataset para este modelo en fallback")
                
                modelo["evento_objetivo"] = evento_objetivo
            
            total_filas = contenido.get('total_filas', 0) or contenido.get('num_registros', 0)
            publicar_metrica_datos_insuficientes(carrera, total_filas)
            
            # Publicar cobertura_total si hay modelos
            if modelos:
                cobertura = modelos[0].get('cobertura_total', 0)
                publicar_metrica('cobertura_total', cobertura, 'None', [{'Name': 'Carrera', 'Value': carrera}])
            
            print(f"\n✅ Fallback exitoso:")
            for m in modelos:
                print(f"   - {m.get('carrera')}: {m.get('data_s3_path')} -> tabla {m.get('tabla_generada')}")
            
            return {"modelos": modelos}
        
        except Exception as e2:
            print(f"❌ Error en fallback: {str(e2)}")
            publicar_metrica_metadata_empty(carrera)
            publicar_metrica_datos_insuficientes(carrera, 0)
            # Ya publicamos fallo_etapa al inicio del except principal, no repetir
            return {
                "modelos": [{
                    "carrera": carrera,
                    "evento_objetivo": evento_objetivo,
                    "carpeta_modelo": f"{carrera}-{generated_at}",
                    "data_s3_path": f"s3://{BUCKET}/modelos/{carrera}-{generated_at}/data/",
                    "tabla_generada": f"modelo_{carrera.replace('-', '_')}_{generated_at.replace('-', '_')}",
                    "dataset_metadata_path": None,
                    "splits": 12,
                    "carreras_usadas": 1
                }]
            }