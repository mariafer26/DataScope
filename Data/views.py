from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from .forms import UploadFileForm
from .models import UploadedFile
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from .user_forms import CustomUserCreationForm
import os
import pandas as pd
import re
from django.db import connection
from . import ai_services
from sqlalchemy import create_engine, text
from django.conf import settings

def home(request):
    return render(request, 'base.html')


def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"¡Bienvenido, {username}!")
                return redirect('/upload/')
 

            else:
                messages.error(request, "Usuario o contraseña inválidos.")
        else:
            messages.error(request, "Usuario o contraseña inválidos.")
    else:
        form = AuthenticationForm()

    return render(request, 'login.html', {'form': form})


def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = CustomUserCreationForm()
    return render(request, 'register.html', {'form': form})

def logout_view(request):
    logout(request)  
    return redirect("home") 

def _sanitize_table_name(name):
    # Remove the extension
    name = os.path.splitext(name)[0]
    # Replace invalid characters with underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Ensure it doesn't start with a number
    if name[0].isdigit():
        name = '_' + name
    return name

def upload_file_view(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save(commit=False)
            uploaded_file.user = request.user
            uploaded_file.name = uploaded_file.file.name
            uploaded_file.save()

            messages.success(request, "File uploaded successfully!")
            return redirect("analyze_file", file_id=uploaded_file.id)  
        else:
            messages.error(request, "There was a problem with the file.")
    else:
        form = UploadFileForm()

    return render(request, "upload.html", {"form": form})


def analyze_file_view(request, file_id):
    uploaded_file = UploadedFile.objects.get(id=file_id)
    file_path = uploaded_file.file.path
    ext = os.path.splitext(file_path)[1].lower()

    table_html = None
    stats = {}
    answer = None
    error = ""
    stats_checked = False

    try:
        if ext == '.csv':
            try:
                df = pd.read_csv(file_path, encoding='utf-8-sig', sep=None, engine='python')
            except Exception:
                df = pd.read_csv(file_path, encoding='latin-1', sep=None, engine='python')
        else:
            df = pd.read_excel(file_path, engine='openpyxl')

        table_html = df.head(20).to_html(index=False, classes="data-table", border=0)

        numeric_df = df.select_dtypes(include="number")
        stats_checked = True

        if numeric_df.empty:
            df_coerced = df.copy()
            def _coerce_to_numeric(series: pd.Series) -> pd.Series:
                t = series.astype(str).str.replace(r'\s|\u00A0', '', regex=True)
                num1 = pd.to_numeric(t, errors="coerce")
                t_alt = t.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                num2 = pd.to_numeric(t_alt, errors="coerce")
                return num2 if num2.notna().sum() > num1.notna().sum() else num1

            for col in df_coerced.columns:
                if df_coerced[col].dtype == "object":
                    coerced = _coerce_to_numeric(df_coerced[col])
                    if coerced.notna().any():
                        df_coerced[col] = coerced
            numeric_df = df_coerced.select_dtypes(include="number")

        if not numeric_df.empty:
            for col in numeric_df.columns:
                s = numeric_df[col].dropna()
                if s.empty:
                    continue

                def r(x):
                    try:
                        return round(float(x), 2)
                    except Exception:
                        return x

                stats[col] = {
                    "mean": r(s.mean()),
                    "median": r(s.median()),
                    "min": r(s.min()),
                    "max": r(s.max()),
                    "count": int(s.count()),
                }
        else:
            messages.info(request, "No numeric columns were detected in the file.")

        answer = [
            {
                "Column": str(c),
                "Type": str(df[c].dtype),
                "BlankSpaces": int(df[c].isna().sum()),
            }
            for c in df.columns
        ]

    except Exception as e:
        messages.error(request, f"Error processing file: {str(e)}")
        error = str(e)

    return render(
        request,
        "analyze.html", 
        {
            "uploaded_file": uploaded_file,
            "table_html": table_html,
            "stats": stats,
            "stats_checked": stats_checked,
            "answer": answer,
            "result": None,   
            "loading": False,
            "error": error,
        },
    )


def ask_question_view(request, file_id):
    uploaded_file = get_object_or_404(UploadedFile, id=file_id, user=request.user)

    question = ""
    answer = None
    error = ""
    loading = False

    if request.method == "POST":
        question = request.POST.get("question", "")
        table_name = _sanitize_table_name(uploaded_file.name)
        
        if question and table_name:
            try:
                loading = True
                
                # Create a database engine
                db_path = settings.DATABASES['default']['NAME']
                engine = create_engine(f'sqlite:///{db_path}')

                # Load data into a DataFrame
                file_path = uploaded_file.file.path
                ext = os.path.splitext(file_path)[1].lower()
                if ext == '.csv':
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8-sig', sep=None, engine='python')
                    except Exception:
                        df = pd.read_csv(file_path, encoding='latin-1', sep=None, engine='python')
                else:
                    df = pd.read_excel(file_path, engine='openpyxl')

                # Save DataFrame to a temporary table
                df.to_sql(table_name, engine, if_exists='replace', index=False)

                if "summary" in question.lower() or "analyze" in question.lower():
                    answer = ai_services.get_summary_from_data(table_name)
                else:
                    sql_query = ai_services.get_sql_from_question(question, table_name)
                    with connection.cursor() as cursor:
                        cursor.execute(sql_query)
                        columns = [col[0] for col in cursor.description]
                        answer = [dict(zip(columns, row)) for row in cursor.fetchall()]

            except Exception as e:
                error = f"Error executing query: {str(e)}"
                messages.error(request, error)
            finally:
                loading = False
                # Drop the temporary table
                if 'engine' in locals() and table_name:
                    with engine.connect() as conn:
                        conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}"'))
                        conn.commit()
        else:
            error = "Please provide a valid question."
            messages.error(request, error)

    return render(
        request,
        "answer.html",
        {
            "uploaded_file": uploaded_file,
            "question": question,
            "answer": answer,
            "error": error,
            "loading": loading,
        },
    )