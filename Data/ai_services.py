import google.generativeai as genai
import os
import re
import requests
import json
from django.db import connection
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# Configure the API key
# Make sure to set the GOOGLE_API_KEY environment variable
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# API keys for other LLMs
HUGGING_API_KEY = os.environ.get("HUGGING_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL_ID = os.environ.get("OPENROUTER_MODEL_ID", "deepseek/deepseek-v3-0324:free")


def get_sql_from_question(question: str, table_name: str) -> str:
    """
    Generates an SQL query from a natural language question using Google's Generative AI.
    Only uses tables within the same temporary database created for the uploaded file.
    """
    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro")

        # Validación básica: la pregunta debe tener al menos 3 palabras
        if len(question.split()) < 3:
            print(f"❌ Pregunta muy corta: {question}")
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
        print(f"❌ Error en get_sql_from_question (Gemini): {str(e)}")
        import traceback
        traceback.print_exc()
        return "__NLP_ERROR__"


def get_response_from_external_db(question, data_source, model="gemini"):
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

        # Usar el modelo LLM seleccionado
        sql_query = get_sql_from_question_unified(question, None, model)

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


# ==================== HUGGING FACE FUNCTIONS ====================

def get_sql_from_question_hf(question: str, table_name: str) -> str:
    """
    Generates an SQL query from a natural language question using Hugging Face API.
    """
    try:
        # Validación básica: la pregunta debe tener al menos 3 palabras
        if len(question.split()) < 3:
            print(f"❌ Pregunta muy corta: {question}")
            return "__NLP_ERROR__"

        # Obtener esquema de tablas
        with connection.cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]

        schemas = []
        with connection.cursor() as cursor:
            for t in tables:
                cursor.execute(f"PRAGMA table_info('{t}');")
                cols = cursor.fetchall()
                if cols:
                    col_list = ", ".join([f"{c[1]} ({c[2]})" for c in cols])
                    schemas.append(f"Table {t}: {col_list}")

        schema_text = "\n".join(schemas)

        prompt = f"""You are an expert in SQLite.
The database contains these tables:

{schema_text}

User question: "{question}"

Generate a valid SQLite SQL query to answer this question.
Prefer using table "{table_name}" if relevant.
Return ONLY the SQL query, no explanation or extra text.
"""

        # Llamada a Hugging Face API - usando un modelo más confiable
        api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        headers = {"Authorization": f"Bearer {HUGGING_API_KEY}"}
        
        # Formato específico para modelos de chat
        payload = {
            "inputs": f"<s>[INST] {prompt} [/INST]",
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.1,
                "return_full_text": False
            }
        }

        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            return "__NLP_ERROR__"

        result = response.json()
        
        # Extraer texto de la respuesta
        if isinstance(result, list) and len(result) > 0:
            sql_query = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            sql_query = result.get("generated_text", "")
        else:
            return "__NLP_ERROR__"

        sql_query = sql_query.strip()
        if not sql_query:
            return "__NLP_ERROR__"

        # Limpiar formato markdown si existe
        if "```sql" in sql_query.lower():
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].split("```")[0].strip()

        # Limpiar texto previo al SELECT
        sql_query = re.sub(
            r"^.*?(SELECT|WITH|INSERT|UPDATE|DELETE)",
            r"\1",
            sql_query,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Agregar punto y coma si no existe
        if not sql_query.strip().endswith(";"):
            sql_query += ";"

        sql_lower = sql_query.lower().strip()

        # Validar que es una consulta válida
        if not sql_lower.startswith(("select", "with", "pragma")):
            return "__NLP_ERROR__"

        return sql_query

    except Exception as e:
        print(f"Error en Hugging Face: {str(e)}")
        return "__NLP_ERROR__"


def get_summary_from_data_hf(table_name: str) -> str:
    """
    Generates a summary of the data using Hugging Face API.
    """
    try:
        # Get table schema
        with connection.cursor() as cursor:
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            columns_info = cursor.fetchall()

        schema = ", ".join([f"{col[1]} ({col[2]})" for col in columns_info])

        # Get sample data
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 20;")
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()

        header = " | ".join(columns)
        data = "\n".join([" | ".join(map(str, row)) for row in rows])

        prompt = f"""You are a data analyst. Analyze this data and provide a summary.

Table: {table_name}
Schema: {schema}

Data:
{header}
{data}

Provide a concise summary of the data:"""

        # Llamada a Hugging Face API - usando Mistral para mejor compatibilidad
        api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        headers = {"Authorization": f"Bearer {HUGGING_API_KEY}"}
        
        payload = {
            "inputs": f"<s>[INST] {prompt[:1000]} [/INST]",  # Limitar tamaño
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.3,
                "return_full_text": False
            }
        }

        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            return "__NLP_ERROR__"

        result = response.json()
        
        # Extraer texto de la respuesta
        if isinstance(result, list) and len(result) > 0:
            summary = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            summary = result.get("generated_text", "") or result.get("summary_text", "")
        else:
            return "__NLP_ERROR__"

        return summary.strip() if summary else "__NLP_ERROR__"

    except Exception as e:
        print(f"Error en Hugging Face summary: {str(e)}")
        return "__NLP_ERROR__"


# ==================== OPENROUTER FUNCTIONS ====================

def get_sql_from_question_openrouter(question: str, table_name: str) -> str:
    """
    Generates an SQL query from a natural language question using OpenRouter API.
    """
    try:
        # Validación básica: la pregunta debe tener al menos 3 palabras
        if len(question.split()) < 3:
            print(f"❌ Pregunta muy corta: {question}")
            return "__NLP_ERROR__"

        # Obtener esquema de tablas
        with connection.cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]

        schemas = []
        with connection.cursor() as cursor:
            for t in tables:
                cursor.execute(f"PRAGMA table_info('{t}');")
                cols = cursor.fetchall()
                if cols:
                    col_list = ", ".join([f"{c[1]} ({c[2]})" for c in cols])
                    schemas.append(f"Table {t}: {col_list}")

        schema_text = "\n".join(schemas)

        prompt = f"""You are an expert in SQLite.
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

        # Llamada a OpenRouter API
        api_url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": OPENROUTER_MODEL_ID,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            print(f"OpenRouter error: {response.status_code} - {response.text}")
            return "__NLP_ERROR__"

        result = response.json()
        sql_query = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if not sql_query:
            return "__NLP_ERROR__"

        # Limpiar formato markdown
        if "```sql" in sql_query.lower():
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].split("```")[0].strip()

        # Limpiar texto previo al SELECT
        sql_query = re.sub(
            r"^.*?(SELECT|WITH|INSERT|UPDATE|DELETE)",
            r"\1",
            sql_query,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Agregar punto y coma
        if not sql_query.strip().endswith(";"):
            sql_query += ";"

        sql_lower = sql_query.lower().strip()

        # Validar consulta
        if not sql_lower.startswith(("select", "with", "pragma")):
            return "__NLP_ERROR__"

        return sql_query

    except Exception as e:
        print(f"Error en OpenRouter: {str(e)}")
        return "__NLP_ERROR__"


def get_summary_from_data_openrouter(table_name: str) -> str:
    """
    Generates a summary of the data using OpenRouter API.
    """
    try:
        # Get table schema
        with connection.cursor() as cursor:
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            columns_info = cursor.fetchall()

        schema = ", ".join([f"{col[1]} ({col[2]})" for col in columns_info])

        # Get sample data
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 20;")
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()

        header = " | ".join(columns)
        data = "\n".join([" | ".join(map(str, row)) for row in rows])

        prompt = f"""You are a data analyst. Given the following table schema and data, provide a summary of the data.

Table: {table_name}
Schema: {schema}

Data:
{header}
{data}

Summary:
"""

        # Llamada a OpenRouter API
        api_url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": OPENROUTER_MODEL_ID,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            return "__NLP_ERROR__"

        result = response.json()
        summary = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        return summary if summary else "__NLP_ERROR__"

    except Exception as e:
        print(f"Error en OpenRouter summary: {str(e)}")
        return "__NLP_ERROR__"


# ==================== UNIFIED WRAPPER FUNCTIONS ====================

def get_sql_from_question_unified(question: str, table_name: str, model: str = "gemini") -> str:
    """
    Unified function to generate SQL query using the selected LLM model.
    
    Args:
        question: Natural language question
        table_name: Name of the table to query
        model: LLM model to use ('gemini', 'huggingface', 'openrouter')
    
    Returns:
        SQL query string or "__NLP_ERROR__" if failed
    """
    print(f"🔍 DEBUG: get_sql_from_question_unified llamada con modelo: {model}")
    print(f"🔍 DEBUG: Pregunta: {question}")
    print(f"🔍 DEBUG: Tabla: {table_name}")
    
    if model == "huggingface":
        result = get_sql_from_question_hf(question, table_name)
    elif model == "openrouter":
        result = get_sql_from_question_openrouter(question, table_name)
    else:  # default to gemini
        result = get_sql_from_question(question, table_name)
    
    print(f"🔍 DEBUG: Resultado de {model}: {result[:100] if result != '__NLP_ERROR__' else result}")
    return result


def get_summary_from_data_unified(table_name: str, model: str = "gemini") -> str:
    """
    Unified function to generate data summary using the selected LLM model.
    
    Args:
        table_name: Name of the table to summarize
        model: LLM model to use ('gemini', 'huggingface', 'openrouter')
    
    Returns:
        Summary text or "__NLP_ERROR__" if failed
    """
    if model == "huggingface":
        return get_summary_from_data_hf(table_name)
    elif model == "openrouter":
        return get_summary_from_data_openrouter(table_name)
    else:  # default to gemini
        return get_summary_from_data(table_name)
