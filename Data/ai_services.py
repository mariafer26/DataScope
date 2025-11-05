import google.generativeai as genai
import os
import re
import requests
import json
from django.db import connection
from django.conf import settings
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


# ==================== DATABASE COMPATIBILITY HELPERS ====================

def get_db_engine():
    """Detecta si estamos usando SQLite o PostgreSQL"""
    engine = settings.DATABASES['default']['ENGINE']
    if 'sqlite' in engine:
        return 'sqlite'
    elif 'postgres' in engine:
        return 'postgresql'
    elif 'mysql' in engine:
        return 'mysql'
    return 'sqlite'  # default


def get_table_schema(table_name, cursor=None):
    """
    Obtiene el esquema de una tabla de forma compatible con SQLite y PostgreSQL.
    Retorna lista de tuplas: (nombre_columna, tipo_dato)
    """
    close_cursor = False
    if cursor is None:
        cursor = connection.cursor()
        close_cursor = True
    
    try:
        db_engine = get_db_engine()
        
        if db_engine == 'postgresql':
            # PostgreSQL: usar information_schema
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = %s 
                ORDER BY ordinal_position
            """, [table_name])
            columns = [(row[0], row[1]) for row in cursor.fetchall()]
        else:
            # SQLite: usar PRAGMA
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            # PRAGMA retorna: (cid, name, type, notnull, dflt_value, pk)
            columns = [(row[1], row[2]) for row in cursor.fetchall()]
        
        return columns
    finally:
        if close_cursor:
            cursor.close()


def get_all_tables(cursor=None):
    """
    Obtiene lista de todas las tablas de forma compatible.
    """
    close_cursor = False
    if cursor is None:
        cursor = connection.cursor()
        close_cursor = True
    
    try:
        db_engine = get_db_engine()
        
        if db_engine == 'postgresql':
            # PostgreSQL: usar information_schema
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
            """)
            tables = [row[0] for row in cursor.fetchall()]
        else:
            # SQLite: usar sqlite_master
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
        
        return tables
    finally:
        if close_cursor:
            cursor.close()


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
        
        # Obtener todas las tablas dentro de la misma base (compatible con SQLite y PostgreSQL)
        with connection.cursor() as cursor:
            tables = get_all_tables(cursor)

        # Construir esquema de todas las tablas dentro de esa base
        schemas = []
        with connection.cursor() as cursor:
            for t in tables:
                columns_info = get_table_schema(t, cursor)
                if columns_info:
                    col_list = ", ".join([f"{name} ({dtype})" for name, dtype in columns_info])
                    schemas.append(f"Table {t}: {col_list}")

        schema_text = "\n".join(schemas)
        
        # Detectar el motor de BD para el prompt
        db_engine = get_db_engine()
        db_type = "PostgreSQL" if db_engine == "postgresql" else "SQLite"

        prompt = f"""
        You are an expert in {db_type}.
        The uploaded file has been imported as a temporary {db_type} database.
        The database contains several tables:

        {schema_text}

        The user question is about: "{question}"

        Generate a valid {db_type} SQL query to answer the question.
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
        error_msg = str(e)
        
        # Detectar error de quota de Gemini
        if "429" in error_msg or "quota" in error_msg.lower():
            print(f"⚠️ Gemini quota excedida - considera usar OpenRouter o esperar")
        else:
            print(f"❌ Error en get_sql_from_question (Gemini): {error_msg}")
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

        # Get the table schema (compatible con SQLite y PostgreSQL)
        with connection.cursor() as cursor:
            columns_info = get_table_schema(table_name, cursor)

        schema = ", ".join([f"{name} ({dtype})" for name, dtype in columns_info])

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

        # Obtener todas las tablas dentro de la misma base (no las globales del sistema)
        with connection.cursor() as cursor:
            tables = get_all_tables(cursor)

        schemas = []
        with connection.cursor() as cursor:
            for t in tables:
                columns_info = get_table_schema(t, cursor)
                if columns_info:
                    col_list = ", ".join([f"{name} ({dtype})" for name, dtype in columns_info])
                    schemas.append(f"Table {t}: {col_list}")

        schema_text = "\n".join(schemas)
        
        # Detectar el motor de BD para el prompt
        db_engine = get_db_engine()
        db_type = "PostgreSQL" if db_engine == "postgresql" else "SQLite"

        prompt = f"""You are an expert in {db_type}.
The database contains these tables:

{schema_text}

User question: "{question}"

Generate a valid {db_type} SQL query to answer this question.
Prefer using table "{table_name}" if relevant.
Return ONLY the SQL query, no explanation or extra text.
"""

        # Usar Mistral a través de OpenRouter (modelo gratuito y confiable)
        # Esto soluciona el problema del endpoint deprecado de HuggingFace
        api_url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Usar Mistral 7B gratuito en lugar de modelos de HuggingFace deprecados
        payload = {
            "model": "mistralai/mistral-7b-instruct:free",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 200
        }

        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        print(f"🔍 HF (Mistral via OpenRouter) SQL Response Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ HF SQL Error: {response.text}")
            return "__NLP_ERROR__"

        result = response.json()
        print(f"🔍 HF SQL Response: {result}")
        
        # Extraer respuesta de OpenRouter (formato de chat completion)
        try:
            sql_query = result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            print(f"❌ HF SQL formato inesperado: {e}")
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
        # Get table schema (compatible con SQLite y PostgreSQL)
        with connection.cursor() as cursor:
            columns_info = get_table_schema(table_name, cursor)

        schema = ", ".join([f"{name} ({dtype})" for name, dtype in columns_info])

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

        # Usar Mistral a través de OpenRouter (modelo gratuito y confiable)
        api_url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mistralai/mistral-7b-instruct:free",
            "messages": [
                {"role": "user", "content": prompt[:1500]}  # Limitar tamaño
            ],
            "temperature": 0.3,
            "max_tokens": 300
        }

        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        print(f"🔍 HF (Mistral via OpenRouter) Summary Response Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ HF Summary Error: {response.text}")
            return "__NLP_ERROR__"

        result = response.json()
        print(f"🔍 HF Summary Response: {result}")
        
        # Extraer respuesta de OpenRouter (formato de chat completion)
        try:
            summary = result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            print(f"❌ HF Summary formato inesperado: {e}")
            return "__NLP_ERROR__"

        return summary if summary else "__NLP_ERROR__"

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

        # Obtener esquema de tablas (compatible con SQLite y PostgreSQL)
        with connection.cursor() as cursor:
            tables = get_all_tables(cursor)

        schemas = []
        with connection.cursor() as cursor:
            for t in tables:
                columns_info = get_table_schema(t, cursor)
                if columns_info:
                    col_list = ", ".join([f"{name} ({dtype})" for name, dtype in columns_info])
                    schemas.append(f"Table {t}: {col_list}")

        schema_text = "\n".join(schemas)
        
        # Detectar el motor de BD para el prompt
        db_engine = get_db_engine()
        db_type = "PostgreSQL" if db_engine == "postgresql" else "SQLite"

        prompt = f"""You are an expert in {db_type}.
The uploaded file has been imported as a temporary {db_type} database.
The database contains several tables:

{schema_text}

The user question is about: "{question}"

Generate a valid {db_type} SQL query to answer the question.
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
        # Get table schema (compatible con SQLite y PostgreSQL)
        with connection.cursor() as cursor:
            columns_info = get_table_schema(table_name, cursor)

        schema = ", ".join([f"{name} ({dtype})" for name, dtype in columns_info])

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
    Con fallback automático a Gemini si el modelo seleccionado falla.
    
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
    
    # Intentar con el modelo seleccionado
    if model == "huggingface":
        result = get_sql_from_question_hf(question, table_name)
        # Fallback a OpenRouter si HuggingFace falla (mejor que Gemini por límites de quota)
        if result == "__NLP_ERROR__":
            print(f"⚠️ HuggingFace falló, usando OpenRouter como fallback")
            result = get_sql_from_question_openrouter(question, table_name)
    elif model == "openrouter":
        result = get_sql_from_question_openrouter(question, table_name)
        # Fallback a Gemini si OpenRouter falla
        if result == "__NLP_ERROR__":
            print(f"⚠️ OpenRouter falló, usando Gemini como fallback")
            result = get_sql_from_question(question, table_name)
    else:  # default to gemini
        result = get_sql_from_question(question, table_name)
    
    print(f"🔍 DEBUG: Resultado de {model}: {result[:100] if result != '__NLP_ERROR__' else result}")
    return result


def get_summary_from_data_unified(table_name: str, model: str = "gemini") -> str:
    """
    Unified function to generate data summary using the selected LLM model.
    Con fallback automático a Gemini si el modelo seleccionado falla.
    
    Args:
        table_name: Name of the table to summarize
        model: LLM model to use ('gemini', 'huggingface', 'openrouter')
    
    Returns:
        Summary text or "__NLP_ERROR__" if failed
    """
    # Intentar con el modelo seleccionado
    if model == "huggingface":
        result = get_summary_from_data_hf(table_name)
        # Fallback a OpenRouter si HuggingFace falla (mejor que Gemini por límites de quota)
        if result == "__NLP_ERROR__":
            print(f"⚠️ HuggingFace summary falló, usando OpenRouter como fallback")
            result = get_summary_from_data_openrouter(table_name)
        return result
    elif model == "openrouter":
        result = get_summary_from_data_openrouter(table_name)
        # Fallback a Gemini si OpenRouter falla
        if result == "__NLP_ERROR__":
            print(f"⚠️ OpenRouter summary falló, usando Gemini como fallback")
            result = get_summary_from_data(table_name)
        return result
    else:  # default to gemini
        return get_summary_from_data(table_name)
