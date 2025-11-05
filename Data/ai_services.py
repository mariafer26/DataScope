import google.generativeai as genai
import os
import re
import requests
from django.db import connection
from dotenv import load_dotenv

load_dotenv()

# Configure the API key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# API keys for other LLMs
HUGGING_API_KEY = os.environ.get("HUGGING_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL_ID = os.environ.get("OPENROUTER_MODEL_ID", "deepseek/deepseek-v3-0324:free")


# ==================== HELPER FUNCTIONS ====================

def get_tables_and_columns(engine_type="sqlite"):
    """
    Obtiene todas las tablas y sus columnas dependiendo del motor de base de datos.
    """
    tables = []
    schemas = []

    with connection.cursor() as cursor:
        if engine_type == "sqlite":
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            for t in tables:
                cursor.execute(f"PRAGMA table_info('{t}');")
                cols = cursor.fetchall()
                if cols:
                    col_list = ", ".join([f"{c[1]} ({c[2]})" for c in cols])
                    schemas.append(f"Table {t}: {col_list}")
        elif engine_type == "postgresql":
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
            """)
            tables = [row[0] for row in cursor.fetchall()]
            for t in tables:
                cursor.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = %s
                    ORDER BY ordinal_position;
                """, [t])
                cols = cursor.fetchall()
                if cols:
                    col_list = ", ".join([f"{c[0]} ({c[1]})" for c in cols])
                    schemas.append(f"Table {t}: {col_list}")
    return tables, schemas


def get_table_schema(table_name: str, engine_type="sqlite"):
    """
    Devuelve el esquema de columnas de una tabla.
    """
    with connection.cursor() as cursor:
        if engine_type == "sqlite":
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            cols = cursor.fetchall()
            schema = ", ".join([f"{c[1]} ({c[2]})" for c in cols])
        elif engine_type == "postgresql":
            cursor.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position;
            """, [table_name])
            cols = cursor.fetchall()
            schema = ", ".join([f"{c[0]} ({c[1]})" for c in cols])
    return schema


# ==================== GEMINI FUNCTIONS ====================

def get_sql_from_question(question: str, table_name: str, engine_type="sqlite") -> str:
    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro")

        if len(question.split()) < 3:
            return "__NLP_ERROR__"

        tables, schemas = get_tables_and_columns(engine_type)
        schema_text = "\n".join(schemas)

        prompt = f"""
        You are an expert in {engine_type}.
        The database contains several tables:

        {schema_text}

        The user question is about: "{question}"

        Generate a valid SQL query to answer the question.
        Prefer using the table "{table_name}" if it is relevant.

        Return only the SQL query or the explanation if required.
        """

        response = model.generate_content(prompt)
        sql_query = response.text.strip()
        if not sql_query:
            return "__NLP_ERROR__"

        if "```sql" in sql_query.lower():
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()

        sql_query = re.sub(
            r"^[\w]*(SELECT|WITH|INSERT|UPDATE|DELETE)",
            r"\1",
            sql_query,
            flags=re.IGNORECASE,
        )

        if not sql_query.strip().endswith(";"):
            sql_query += ";"

        sql_lower = sql_query.lower().strip()
        if not sql_lower.startswith(("select", "with")):
            return "__NLP_ERROR__"

        return sql_query

    except Exception as e:
        print(f"❌ Error en Gemini SQL: {e}")
        return "__NLP_ERROR__"


def get_summary_from_data(table_name: str, engine_type="sqlite") -> str:
    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro")

        schema = get_table_schema(table_name, engine_type)

        with connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 20;")
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()

        header = " | ".join(columns)
        data = "\n".join([" | ".join(map(str, row)) for row in rows])

        prompt = f"""
        You are a data analyst. Given the following table schema and data, provide a summary.

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
        print(f"❌ Error en Gemini summary: {e}")
        return "__NLP_ERROR__"


# ==================== HUGGINGFACE / OPENROUTER FUNCTIONS ====================

def get_sql_from_question_hf(question: str, table_name: str, engine_type="postgresql") -> str:
    try:
        if len(question.split()) < 3:
            return "__NLP_ERROR__"

        tables, schemas = get_tables_and_columns(engine_type)
        schema_text = "\n".join(schemas)

        prompt = f"""You are an expert in {engine_type}.
The database contains these tables:

{schema_text}

User question: "{question}"

Generate a valid SQL query to answer this question.
Prefer using table "{table_name}" if relevant.
Return ONLY the SQL query, no explanation.
"""

        api_url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": "mistralai/mistral-7b-instruct:free", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": 200}

        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            return "__NLP_ERROR__"

        result = response.json()
        sql_query = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if "```sql" in sql_query.lower():
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].split("```")[0].strip()

        sql_query = re.sub(r"^.*?(SELECT|WITH|INSERT|UPDATE|DELETE)", r"\1", sql_query, flags=re.IGNORECASE | re.DOTALL)

        if not sql_query.strip().endswith(";"):
            sql_query += ";"

        sql_lower = sql_query.lower().strip()
        if not sql_lower.startswith(("select", "with")):
            return "__NLP_ERROR__"

        return sql_query

    except Exception as e:
        print(f"Error en HuggingFace SQL: {e}")
        return "__NLP_ERROR__"


def get_summary_from_data_hf(table_name: str, engine_type="postgresql") -> str:
    try:
        schema = get_table_schema(table_name, engine_type)

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

        api_url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": "mistralai/mistral-7b-instruct:free", "messages": [{"role": "user", "content": prompt[:1500]}], "temperature": 0.3, "max_tokens": 300}

        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            return "__NLP_ERROR__"

        result = response.json()
        summary = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return summary if summary else "__NLP_ERROR__"

    except Exception as e:
        print(f"Error en HuggingFace summary: {e}")
        return "__NLP_ERROR__"


# ==================== UNIFIED WRAPPER FUNCTIONS ====================

def get_sql_from_question_unified(question: str, table_name: str, model: str = "gemini", engine_type="postgresql") -> str:
    if model == "huggingface":
        result = get_sql_from_question_hf(question, table_name, engine_type)
        if result == "__NLP_ERROR__":
            result = get_sql_from_question(question, table_name, engine_type)
    elif model == "openrouter":
        result = get_sql_from_question_hf(question, table_name, engine_type)
        if result == "__NLP_ERROR__":
            result = get_sql_from_question(question, table_name, engine_type)
    else:
        result = get_sql_from_question(question, table_name, engine_type)
    return result


def get_summary_from_data_unified(table_name: str, model: str = "gemini", engine_type="postgresql") -> str:
    if model == "huggingface":
        result = get_summary_from_data_hf(table_name, engine_type)
        if result == "__NLP_ERROR__":
            result = get_summary_from_data(table_name, engine_type)
    elif model == "openrouter":
        result = get_summary_from_data_hf(table_name, engine_type)
        if result == "__NLP_ERROR__":
            result = get_summary_from_data(table_name, engine_type)
    else:
        result = get_summary_from_data(table_name, engine_type)
    return result
