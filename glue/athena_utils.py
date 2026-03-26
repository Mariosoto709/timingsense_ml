# glue/athena_utils.py
"""
Funciones de Athena y Glue para el job (dependen de AWS)
"""

import time
import boto3
import re
from awsglue.utils import getResolvedOptions

# ============================================================
# CONFIG
# ============================================================

REGION = "eu-north-1"
DATABASE = "timingsense_db"
WORKGROUP = "timingsense-ath"
S3_OUTPUT = "s3://timingsense-athena-output-2026/query-results/"
S3_PROCESSED_BUCKET = "timingsense-races-processed-wide"
S3_ATHENA_OUTPUT = "timingsense-athena-output-2026"

athena = boto3.client("athena", region_name=REGION)
s3 = boto3.client("s3")
glue = boto3.client("glue", region_name=REGION)


# ============================================================
# UTILIDADES ATHENA
# ============================================================

def wait_for_query(execution_id):
    while True:
        response = athena.get_query_execution(QueryExecutionId=execution_id)
        state = response["QueryExecution"]["Status"]["State"]

        if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            if state != "SUCCEEDED":
                reason = response["QueryExecution"]["Status"].get("StateChangeReason", "Unknown")
                raise Exception(f"Query falló: {reason}")
            return
        time.sleep(2)


def execute_athena_query(query):
    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": DATABASE},
        ResultConfiguration={"OutputLocation": S3_OUTPUT},
        WorkGroup=WORKGROUP
    )
    execution_id = response["QueryExecutionId"]
    wait_for_query(execution_id)
    result = athena.get_query_execution(QueryExecutionId=execution_id)
    return result["QueryExecution"]["ResultConfiguration"]["OutputLocation"]


# ============================================================
# FUNCIÓN PARA VERIFICAR SI UNA TABLA EXISTE
# ============================================================

def tabla_existe(nombre_tabla):
    try:
        glue.get_table(DatabaseName=DATABASE, Name=nombre_tabla)
        return True
    except glue.exceptions.EntityNotFoundException:
        return False
    except Exception as e:
        print(f"⚠️ Error verificando tabla {nombre_tabla}: {e}")
        return False


# ============================================================
# CREAR TABLA EN GLUE DIRECTAMENTE
# ============================================================

def create_glue_table(table_name, split_columns, partitioned=True):
    """
    Crea una tabla en el Catálogo de Datos de Glue apuntando a los datos Parquet.
    """
    # Procesar columnas de splits
    columns = []
    for col_def in split_columns:
        match = re.match(r'"([^"]+)"\s+(\w+)', col_def)
        if match:
            col_name, col_type = match.groups()
        else:
            parts = col_def.split()
            col_name = parts[0].strip('"')
            col_type = parts[1] if len(parts) > 1 else 'string'
        columns.append({'Name': col_name, 'Type': col_type})

    # Columnas estándar
    standard_columns = [
        {'Name': 'athlete_id', 'Type': 'string'},
        {'Name': 'event_std', 'Type': 'string'},
        {'Name': 'gender', 'Type': 'string'},
        {'Name': 'age', 'Type': 'int'}
    ]

    all_columns = standard_columns + columns

    storage_descriptor = {
        'Columns': all_columns,
        'Location': f's3://{S3_PROCESSED_BUCKET}/wide/',
        'InputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat',
        'OutputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat',
        'SerdeInfo': {
            'SerializationLibrary': 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe',
            'Parameters': {'serialization.format': '1'}
        },
        'Parameters': {'classification': 'parquet'}
    }

    table_input = {
        'Name': table_name,
        'StorageDescriptor': storage_descriptor,
        'TableType': 'EXTERNAL_TABLE',
        'Parameters': {'EXTERNAL': 'TRUE'}
    }

    if partitioned:
        table_input['PartitionKeys'] = [
            {'Name': 'race_id', 'Type': 'string'},
            {'Name': 'event_id', 'Type': 'string'}
        ]

    try:
        glue.create_table(DatabaseName=DATABASE, TableInput=table_input)
        print(f"✅ Tabla {table_name} creada en Glue")
    except glue.exceptions.AlreadyExistsException:
        print(f"ℹ️ Tabla {table_name} ya existe")
    except Exception as e:
        print(f"❌ Error creando tabla en Glue: {e}")
        raise