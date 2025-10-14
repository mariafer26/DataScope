import os
import pandas as pd
from sqlalchemy import create_engine, inspect
from django.conf import settings

from .models import UploadedFile, DataSource


# 🔹 Construye la cadena de conexión SQLAlchemy según el motor
def build_connection_string(data_source: DataSource):
    engine = data_source.engine.lower()

    if engine == "sqlite":
        if not data_source.sqlite_path:
            raise ValueError("No SQLite path specified")
        path = data_source.sqlite_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"SQLite file not found: {path}")
        return f"sqlite:///{path}"

    elif engine == "postgresql":
        return (
            f"postgresql://{data_source.username}:{data_source.password}"
            f"@{data_source.host}:{data_source.port}/{data_source.db_name}"
        )

    elif engine == "mysql":
        return (
            f"mysql+mysqldb://{data_source.username}:{data_source.password}"
            f"@{data_source.host}:{data_source.port}/{data_source.db_name}"
        )

    else:
        raise ValueError(f"Unsupported engine: {engine}")


def get_table_names_from_source(data_source: DataSource):
    conn_string = build_connection_string(data_source)
    engine = create_engine(conn_string)
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    engine.dispose()
    return tables


def get_table_data_from_source(data_source: DataSource, table_name: str, limit: int = 1000):
    conn_string = build_connection_string(data_source)
    engine = create_engine(conn_string)
    query = f"SELECT * FROM {table_name} LIMIT {limit}"
    df = pd.read_sql_query(query, engine)
    engine.dispose()
    return df


def get_table_names_from_file(uploaded_file: UploadedFile):
    file_path = uploaded_file.file.path
    if file_path.endswith(".csv"):
        return ["(Archivo CSV único)"]
    elif file_path.endswith(".xlsx"):
        excel = pd.ExcelFile(file_path)
        return excel.sheet_names
    else:
        raise ValueError("Formato de archivo no soportado (solo CSV o XLSX)")


def get_table_data_from_file(uploaded_file: UploadedFile, sheet_name=None):
    file_path = uploaded_file.file.path
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        raise ValueError("Formato de archivo no soportado")
    return df
