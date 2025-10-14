from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from .forms import UploadFileForm, DBConnectionForm
from .models import UploadedFile, DataSource
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from .user_forms import CustomUserCreationForm
import os
import pandas as pd
import re
from django.db import connection
from . import ai_services
from django.conf import settings
from django.utils import translation
from django.contrib.auth.decorators import login_required
from .decorators import admin_required
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine, text, inspect
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.http import HttpResponse, HttpResponseBadRequest
from django.template.loader import get_template
from xhtml2pdf import pisa
import io
import base64
from django.contrib.staticfiles import finders
from .tables import (
    get_table_names_from_source,
    get_table_data_from_source,
    get_table_names_from_file,
    get_table_data_from_file
)



def home(request):
    return render(request, "base.html")


def login_view(request):
    with translation.override('en'):
        if request.method == 'POST':
            form = AuthenticationForm(request, data=request.POST)
            if form.is_valid():
                username = form.cleaned_data.get('username')
                password = form.cleaned_data.get('password')
                user = authenticate(username=username, password=password)
                if user is not None:
                    login(request, user)
                    messages.success(request, f"¡Bienvenido, {username}!")
                    return redirect('home')

                else:
                    messages.error(request, "Usuario o contraseña inválidos.")
            else:
                messages.error(request, "Usuario o contraseña inválidos.")
        else:
            form = AuthenticationForm()

        return render(request, 'login.html', {'form': form})


def register_view(request):
    with translation.override('en'):
        if request.method == 'POST':
            form = CustomUserCreationForm(request.POST)
            if form.is_valid():
                user = form.save()
                login(request, user)
                return redirect('home')
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
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Ensure it doesn't start with a number
    if name[0].isdigit():
        name = "_" + name
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
    
    is_csv = ext == ".csv"
    table_html = None
    stats = {}
    answer = None
    error = ""
    stats_checked = False

    try:
        if ext == ".csv":
            try:
                df = pd.read_csv(
                    file_path,
                    encoding='utf-8-sig',
                    sep=None,
                    engine='python'
                )
            except Exception:
                df = pd.read_csv(
                    file_path,
                    encoding='latin-1',
                    sep=None,
                    engine='python'
                )
        else:
            df = pd.read_excel(file_path, engine="openpyxl")

        table_html = df.head(20).to_html(
            index=False,
            classes="data-table",
            border=0
        )

        numeric_df = df.select_dtypes(include="number")
        stats_checked = True

        if numeric_df.empty:
            df_coerced = df.copy()

            def _coerce_to_numeric(series: pd.Series) -> pd.Series:
                t = series.astype(str).str.replace(r"\s|\u00A0", "", regex=True)
                num1 = pd.to_numeric(t, errors="coerce")
                t_alt = t.str.replace(".", "", regex=False).str.replace(
                    ",", ".", regex=False
                )
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
    
        
    try:
        last_ctx = {
            "file_name": uploaded_file.name or os.path.basename(file_path),
            "generated_at": timezone.now().strftime("%Y-%m-%d %H:%M"),
            "table_html": table_html or "",
            "stats": stats or {},
        }

        
        if isinstance(answer, (list, dict)):
            last_ctx["answer"] = answer
        elif isinstance(answer, str):
            last_ctx["answer_text"] = answer

        request.session["last_upload_context"] = last_ctx
    except Exception:
        
        request.session["last_upload_context"] = {}
     

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
            "is_csv": is_csv,
            "is_database": False,
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
                db_path = settings.DATABASES["default"]["NAME"]
                engine = create_engine(f"sqlite:///{db_path}")

                # Load data into a DataFrame
                file_path = uploaded_file.file.path
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".csv":
                    try:
                        df = pd.read_csv(
                            file_path, encoding="utf-8-sig", sep=None, engine="python"
                        )
                    except Exception:
                        df = pd.read_csv(
                            file_path, encoding="latin-1", sep=None, engine="python"
                        )
                else:
                    df = pd.read_excel(file_path, engine="openpyxl")

                # Save DataFrame to a temporary table
                df.to_sql(table_name, engine, if_exists="replace", index=False)

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
                if "engine" in locals() and table_name:
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


@login_required
def dashboard_view(request):
    files = UploadedFile.objects.filter(user=request.user)
    return render(request, "dashboard.html", {"files": files})

@login_required
@admin_required
def admin_dashboard_view(request):
    """Panel exclusivo para administradores"""
    files = UploadedFile.objects.all().order_by("-uploaded_at")
    return render(request, "admin_dashboard.html", {"files": files})

def _build_sqlalchemy_url(ds: DataSource) -> str:
    """
    Construye el URL de conexión SQLAlchemy según el motor elegido.
    """
    if ds.engine == "postgresql":
        # driver psycopg2
        return str(URL.create(
            drivername="postgresql+psycopg2",
            username=ds.username,
            password=ds.password,
            host=ds.host,
            port=int(ds.port) if ds.port else None,
            database=ds.db_name,
        ))
    elif ds.engine == "mysql":
        # driver PyMySQL
        return str(URL.create(
            drivername="mysql+pymysql",
            username=ds.username,
            password=ds.password,
            host=ds.host,
            port=int(ds.port) if ds.port else None,
            database=ds.db_name,
        ))
    elif ds.engine == "sqlite":
        # Para sqlite usamos ruta absoluta (o relativa) al archivo .db / .sqlite
        path = ds.sqlite_path or ""
        return f"sqlite:///{path}"
    else:
        raise ValueError("Unsupported engine")


@login_required
@require_http_methods(["GET", "POST"])
def connect_db_view(request):
    """
    Muestra el formulario, valida la conexión con SELECT 1,
    y guarda la configuración si todo OK.
    """
    if request.method == "POST":
        form = DBConnectionForm(request.POST)
        if form.is_valid():
            ds: DataSource = form.save(commit=False)
            ds.user = request.user

            # Si se marca como activa, desactiva las demás del usuario
            if ds.is_active:
                DataSource.objects.filter(user=request.user, is_active=True).update(is_active=False)

            # Probar conexión
            try:
                # Para probar, usamos un DataSource "temporal" no guardado
                test_url = _build_sqlalchemy_url(ds)
                engine = create_engine(test_url)
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                # Si llegamos aquí, la conexión funciona
                ds.save()
                messages.success(request, "Connection successful and configuration saved.")
                return redirect("analyze_db", db_id=ds.id)
            except Exception as e:
                messages.error(request, f"Connection failed: {e}")
        else:
            messages.error(request, "Please correct the errors in the form.")
    else:
        form = DBConnectionForm()

    return render(request, "connect_db.html", {"form": form})


@login_required
def connections_list_view(request):

    """
    Lista conexiones guardadas del usuario y permite activar una.
    """
    if request.method == "POST":
        # Activar una conexión
        ds_id = request.POST.get("activate_id")
        if ds_id:
            ds = get_object_or_404(DataSource, id=ds_id, user=request.user)
            DataSource.objects.filter(user=request.user, is_active=True).update(is_active=False)
            ds.is_active = True
            ds.save()
            messages.success(request, f"'{ds.name}' is now the active connection.")
            return redirect("connections_list")

    items = DataSource.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "connections_list.html", {"items": items})


def sanitize_dataframe(df, max_text_length=100):
    """
    Cleans a DataFrame before showing it in the frontend:
      - Removes sensitive columns (passwords, tokens, etc.)
      - Removes completely empty columns
      - Truncates very long text values for display safety
    """
    df = df.dropna(axis=1, how='all')

    sensitive_patterns = [
        r"password", r"pass", r"pwd",
        r"token", r"secret", r"auth",
        r"apikey", r"api_key",
        r"ssn", r"credit", r"card",
        r"email", r"user_?id", r"login"
    ]
    cols_to_drop = [
        col for col in df.columns
        if any(re.search(pattern, col.lower()) for pattern in sensitive_patterns)
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors="ignore")

    # --- 3. Truncate long text values ---
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).apply(
            lambda x: (x[:max_text_length] + "…") if len(x) > max_text_length else x
        )

    return df, cols_to_drop


def analyze_db_view(request, db_id):
    ds = get_object_or_404(DataSource, id=db_id, user=request.user)
    try:
        # Crear engine SQLAlchemy
        db_url = _build_sqlalchemy_url(ds)
        engine = create_engine(db_url)

        # Obtener las tablas disponibles
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        if not tables:
            messages.warning(request, "The database has no visible tables.")
            return redirect("connections_list")

        # Tomar la primera tabla por defecto
        table_name = tables[0]
        df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 50", engine)

        df, hidden_cols = sanitize_dataframe(df)

        # Si se ocultaron columnas, notificar sin revelar nombres
        if hidden_cols:
            messages.warning(
                request,
                f"{len(hidden_cols)} sensitive column(s) were hidden for security reasons."
            )

        # Mostrar solo las primeras 4 columnas para evitar desbordes
        df_preview = df.iloc[:, :4]

        # Convertir a HTML para mostrar
        table_html = df_preview.head(20).to_html(
            index=False,
            classes="data-table",
            border=0
        )

        # Calcular estadísticas
        numeric_df = df.select_dtypes(include="number")
        stats = {}
        for col in numeric_df.columns:
            s = numeric_df[col].dropna()
            if not s.empty:
                stats[col] = {
                    "mean": round(float(s.mean()), 2),
                    "median": round(float(s.median()), 2),
                    "min": round(float(s.min()), 2),
                    "max": round(float(s.max()), 2),
                    "count": int(s.count()),
                }

        # Guardar contexto en sesión (opcional)
        request.session["last_upload_context"] = {
            "file_name": ds.name,
            "generated_at": timezone.now().strftime("%Y-%m-%d %H:%M"),
            "table_html": table_html,
            "stats": stats,
        }

        # Renderizar plantilla
        return render(
            request,
            "analyze.html",
            {
                "uploaded_file": ds,
                "table_html": table_html,
                "stats": stats,
                "stats_checked": True,
                "answer": None,
                "result": None,
                "loading": False,
                "error": "",
                "is_csv": False,
                "is_database": True,
                "table_name": table_name,
                "tables": tables,
            },
        )

    except Exception as e:
        messages.error(request, f"Error analyzing database: {str(e)}")
        return redirect("connections_list")


def export_pdf_view(request):
    """
    Genera un PDF con Data preview, Basic metrics y Respuesta
    usando el contexto guardado en sesión por analyze_file_view.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("Invalid method")

    data = request.session.get("last_upload_context") or {}
    if not data or (
        not data.get("table_html")
        and not data.get("stats")
        and not data.get("answer")
        and not data.get("answer_text")
    ):
        return HttpResponseBadRequest("No hay resultados para exportar")

    # Renderizar HTML imprimible
    template = get_template("exports/report.pdf.html")

    # ---- Logo en base64 (si existe en static/img/logo.png) ----
    logo_b64 = ""
    try:
        logo_path = finders.find("img/logo.png")
        if logo_path and os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        logo_b64 = ""
    

    html = template.render({
        "export": data,
        "logo_b64": logo_b64,
    })

    # Convertir a PDF en memoria
    result = io.BytesIO()
    pdf = pisa.CreatePDF(io.BytesIO(html.encode("utf-8")), dest=result, encoding="utf-8")
    if pdf.err:
        return HttpResponseBadRequest("Error generando el PDF")

    # Descargar
    filename = (data.get("file_name") or "DataScope_Report").replace('"', '')
    response = HttpResponse(result.getvalue(), content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{filename}.pdf"'
    return response


def select_table_view(request, source_type, source_id):
    """
    Permite listar tablas según el tipo de origen: archivo o base de datos.
    """
    if source_type == "db":
        data_source = DataSource.objects.get(pk=source_id)
        tables = get_table_names_from_source(data_source)
    elif source_type == "file":
        uploaded = UploadedFile.objects.get(pk=source_id)
        tables = get_table_names_from_file(uploaded)
    else:
        return render(request, "error.html", {"msg": "Tipo de fuente inválido"})

    return render(request, "select_table.html", {"tables": tables, "source_id": source_id, "source_type": source_type})


def show_table_view(request):
    if request.method == "POST":
        source_type = request.POST.get("source_type")
        source_id = request.POST.get("source_id")
        table = request.POST.get("table")

        if source_type == "db":
            data_source = DataSource.objects.get(pk=source_id)
            df = get_table_data_from_source(data_source, table)
        else:
            uploaded = UploadedFile.objects.get(pk=source_id)
            sheet_name = None if table == "(Archivo CSV único)" else table
            df = get_table_data_from_file(uploaded, sheet_name)

        html = df.head(100).to_html(classes="table table-striped", index=False)
        return render(request, "show_table.html", {"table_html": html, "table_name": table})