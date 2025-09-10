import google.generativeai as genai
import os
from django.db import connection

# Configure the API key
# Make sure to set the GOOGLE_API_KEY environment variable
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


def get_sql_from_question(question: str, table_name: str) -> str:
    """
    Generates an SQL query from a natural language question using Google's Generative AI.

    Args:
        question: The natural language question.
        table_name: The name of the table to query.

    Returns:
        The generated SQL query.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Get the table schema
    with connection.cursor() as cursor:
        cursor.execute(f"PRAGMA table_info('{table_name}');")
        columns_info = cursor.fetchall()

    schema = ", ".join([f"{col[1]} ({col[2]})" for col in columns_info])

    prompt = f"""
    You are a SQL expert. Given the following table schema and a natural language question,
    generate a valid SQL query to answer the question.

    Table: {table_name}
    Schema: {schema}

    Question: {question}

    SQL Query:
    """

    response = model.generate_content(prompt)
    sql_query = response.text.strip()

    # Clean the response to get only the SQL query
    if "```sql" in sql_query:
        sql_query = sql_query.split("```sql")[1].split("```")[0].strip()

    return sql_query


def get_summary_from_data(table_name: str) -> str:
    """
    Generates a summary of the data in the specified table.

    Args:
        table_name: The name of the table.

    Returns:
        The generated summary.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")

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
