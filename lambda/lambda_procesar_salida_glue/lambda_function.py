import boto3
import json
from datetime import datetime

s3 = boto3.client('s3')
BUCKET = "timingsense-athena-output-2026"

def lambda_handler(event, context):
    print("🔍 Procesando salida de Glue")
    print(f"Evento recibido: {json.dumps(event, indent=2)}")
    
    glue_job_run_id = event.get('glue_job_run_id')
    carrera = event.get('carrera_objetivo', 'desconocida')
    generated_at = event.get('generated_at')
    
    if not generated_at:
        print("❌ No hay generated_at en el evento")
        return {"modelos": []}
    
    # 🟢🟢🟢 CONSTRUIR LA CARPETA CORRECTA CON GENERATED_AT 🟢🟢🟢
    timestamp_archivo = generated_at.replace('-', '_')
    key = f"debug/salida_step_{timestamp_archivo}.json"
    
    print(f"📄 Buscando archivo: {key}")
    
    try:
        # Intentar leer el archivo específico
        response = s3.get_object(Bucket=BUCKET, Key=key)
        contenido = json.loads(response['Body'].read())
        print(f"✅ Archivo encontrado: {contenido}")
        
        # 🟢 MODIFICAR LA CARPETA CON EL GENERATED_AT CORRECTO 🟢
        modelos = contenido.get("modelos", [])
        for modelo in modelos:
            # Reemplazar la carpeta con la correcta
            carpeta_correcta = f"{carrera}-{generated_at}"
            modelo["carpeta_modelo"] = carpeta_correcta
            modelo["data_s3_path"] = f"s3://{BUCKET}/modelos/{carpeta_correcta}/data/"
            modelo["tabla_generada"] = f"modelo_{carrera.replace('-', '_')}_{generated_at.replace('-', '_')}"
        
        # Devolver en el formato que espera Step Functions
        return {
            "modelos": modelos
        }
        
    except Exception as e:
        print(f"❌ Error leyendo archivo específico: {str(e)}")
        print("⚠️ Buscando el archivo más reciente como fallback...")
        
        # Fallback: listar archivos recientes
        try:
            response = s3.list_objects_v2(
                Bucket=BUCKET,
                Prefix="debug/salida_step_",
                MaxKeys=10
            )
            
            if 'Contents' not in response:
                print("❌ No se encontraron archivos")
                # 🟢 Fallback: construir modelo con generated_at
                return {
                    "modelos": [
                        {
                            "carrera": carrera,
                            "carpeta_modelo": f"{carrera}-{generated_at}",
                            "data_s3_path": f"s3://{BUCKET}/modelos/{carrera}-{generated_at}/data/",
                            "tabla_generada": f"modelo_{carrera.replace('-', '_')}_{generated_at.replace('-', '_')}",
                            "splits": 12,
                            "carreras_usadas": 1
                        }
                    ]
                }
            
            # Mostrar archivos disponibles
            print("📂 Archivos disponibles:")
            for obj in response['Contents']:
                print(f"   - {obj['Key']} ({obj['LastModified']})")
            
            # Tomar el más reciente
            archivos = sorted(response['Contents'], 
                             key=lambda x: x['LastModified'], 
                             reverse=True)
            
            archivo_reciente = archivos[0]
            key = archivo_reciente['Key']
            
            print(f"📄 Usando archivo más reciente: {key}")
            
            response = s3.get_object(Bucket=BUCKET, Key=key)
            contenido = json.loads(response['Body'].read())
            
            # 🟢 MODIFICAR LA CARPETA CON EL GENERATED_AT CORRECTO 🟢
            modelos = contenido.get("modelos", [])
            for modelo in modelos:
                carpeta_correcta = f"{carrera}-{generated_at}"
                modelo["carpeta_modelo"] = carpeta_correcta
                modelo["data_s3_path"] = f"s3://{BUCKET}/modelos/{carpeta_correcta}/data/"
                modelo["tabla_generada"] = f"modelo_{carrera.replace('-', '_')}_{generated_at.replace('-', '_')}"
            
            return {
                "modelos": modelos
            }
            
        except Exception as e2:
            print(f"❌ Error en fallback: {str(e2)}")
            # 🟢 Fallback final: construir modelo con generated_at
            return {
                "modelos": [
                    {
                        "carrera": carrera,
                        "carpeta_modelo": f"{carrera}-{generated_at}",
                        "data_s3_path": f"s3://{BUCKET}/modelos/{carrera}-{generated_at}/data/",
                        "tabla_generada": f"modelo_{carrera.replace('-', '_')}_{generated_at.replace('-', '_')}",
                        "splits": 12,
                        "carreras_usadas": 1
                    }
                ]
            }