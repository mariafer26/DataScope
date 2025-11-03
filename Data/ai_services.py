import google.generativeai as genai
import os
import re
from django.db import connection
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# Configure the API key
# Make sure to set the GOOGLE_API_KEY environment variable
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


def get_sql_from_question(question: str, table_name: str) -> str:
    """
    Generates an SQL query from a natural language question using Google's Generative AI.
    Only uses tables within the same temporary database created for the uploaded file.
    """
    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro")

        # Detectar preguntas sin sentido: si no contiene ninguna palabra clave, forzar error
        keywords = [
            # English
            "sum", "avg", "average", "mean",
            "max", "maximum", "min", "minimum",
            "total", "count", "filter", "where",
            "group", "order", "sort", "top",
            "list", "show", "display", "select",
            # Spanish support
            "suma", "promedio", "media", "máximo", "mínimo",
            "total", "contar", "filtrar", "ordenar", "mostrar"
        ]
        
             

        if not any(k in question.lower() for k in keywords):
            return "__NLP_ERROR__"

        
        # Obtener todas las tablas dentro de la misma base (no las globales del sistema)
        with connection.cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]

        # Construir esquema de todas las tablas dentro de esa base
        schemas = []
        with connection.cursor() as cursor:
            for t in tables:
                cursor.execute(f"PRAGMA table_info('{t}');")
                cols = cursor.fetchall()
                if cols:
                    col_list = ", ".join([f"{c[1]} ({c[2]})" for c in cols])
                    schemas.append(f"Table {t}: {col_list}")

        schema_text = "\n".join(schemas)

        prompt = f"""
        You are an expert in SQLite.
        The uploaded file has been imported as a temporary SQLite database.
        The database contains several tables:

        {schema_text}

        The user question is about: "{question}"

        Generate a valid SQLite SQL query to answer the question.
        Prefer using the table "{table_name}" if it is relevant,
        but you can use other tables if the question explicitly mentions them.

        Return only the SQL query, no explanation.

        Remember, it must be a valid SQL query.

        If a summary is requested, you must access the data and generate the response in natural language.
        """

        response = model.generate_content(prompt)
        sql_query = response.text.strip()
        if not sql_query:
            raise ValueError("No SQL returned")

        # Si hay formato markdown, extraer el bloque SQL
        if "```sql" in sql_query.lower():
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()

        # Si el modelo escribió texto antes del SELECT, limpiarlo
        sql_query = re.sub(
            r"^[\w]*(SELECT|WITH|INSERT|UPDATE|DELETE)",
            r"\1",
            sql_query,
            flags=re.IGNORECASE,
        )

        # Asegurar que termina con punto y coma
        if not sql_query.strip().endswith(";"):
            sql_query += ";"


        sql_lower = sql_query.lower().strip()

        invalid_patterns = [
            "", "null", "none",
            "table", "column", "schema",
            "select;", "with;", "pragma;"
        ]

        if any(sql_lower == p for p in invalid_patterns):
            return "__NLP_ERROR__"

        if not sql_lower.startswith(("select", "with", "pragma")):
            return "__NLP_ERROR__"
        # -----------------------------

        return sql_query

    except Exception as e:
        return "__NLP_ERROR__"


def get_response_from_external_db(question, data_source):
    try:
        """
        Ejecuta una consulta SQL en una base de datos externa según el tipo de motor.
        """

        if data_source.engine == "postgresql":
            conn_str = f"postgresql://{data_source.username}:{data_source.password}@{data_source.host}:{data_source.port}/{data_source.db_name}"
        elif data_source.engine == "mysql":
            conn_str = f"mysql+pymysql://{data_source.username}:{data_source.password}@{data_source.host}:{data_source.port}/{data_source.db_name}"
        elif data_source.engine == "sqlite":
            conn_str = f"sqlite:///{data_source.sqlite_path}"
        else:
            raise ValueError("Tipo de base de datos no soportado")

        engine = create_engine(conn_str)

        # Puedes usar el mismo modelo LLM
        sql_query = get_sql_from_question(question, None)

        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            columns = result.keys()
            rows = result.fetchall()

        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        return "__NLP_ERROR__"


def get_summary_from_data(table_name: str) -> str:
    """
    Generates a summary of the data in the specified table.

    Args:
        table_name: The name of the table.

    Returns:
        The generated summary.
    """
    try:
        
        model = genai.GenerativeModel("models/gemini-2.5-pro")

        # Get the table schema
        with connection.cursor() as cursor:
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            columns_info = cursor.fetchall()

        schema = ", ".join([f"{col[1]} ({col[2]})" for col in columns_info])

        # Get the first 20 rows of data
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 20;")
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()

        header = " | ".join(columns)
        data = "\n".join([" | ".join(map(str, row)) for row in rows])

        prompt = f"""
        You are a data analyst. Given the following table schema and data, provide a summary of the data.

        Table: {table_name}
        Schema: {schema}

        Data:
        {header}
        {data}

        Summary:
        """

        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return "__NLP_ERROR__"
