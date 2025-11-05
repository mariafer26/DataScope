from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from .forms import UploadFileForm, DBConnectionForm, FavoriteQuestionForm
from .models import UploadedFile, DataSource, FavoriteQuestion, QueryHistory, LLMMetrics
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from .user_forms import CustomUserCreationForm
import os
import time
import pandas as pd
import json
import re
from django.db import connection
from . import ai_services
from django.conf import settings
from django.utils import translation
from django.contrib.auth.decorators import login_required
from .decorators import admin_required
from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine, text, inspect
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
from django.template.loader import get_template
from django.core.paginator import Paginator
from xhtml2pdf import pisa
import io
import base64
import json
import numpy as np
from datetime import datetime
from django.contrib.staticfiles import finders
from .tables import (
    get_table_names_from_source,
    get_table_data_from_source,
    get_table_names_from_file,
    get_table_data_from_file,
)

#set as default source
def home(request):
    return render(request, "base.html")

def check_session_expired(request):
    if request.session.get('session_expired'):
        # Limpiar el flag
        del request.session['session_expired']
        messages.warning(
            request,
            "Your session has expired due to inactivity. Please refresh the page and log in again."
        )
        return True
    return False



def login_view(request):
    with translation.override("en"):

        session_expired = request.COOKIES.get('session_expired') == 'true'

        if request.method == "POST":
            form = AuthenticationForm(request, data=request.POST)
            if form.is_valid():
                username = form.cleaned_data.get("username")
                password = form.cleaned_data.get("password")
                user = authenticate(username=username, password=password)
                if user is not None:
                    login(request, user)

                    from django.utils import timezone
                    request.session['last_activity'] = timezone.now().isoformat()
                    request.session['login_time'] = timezone.now().isoformat()
                    request.session.set_expiry(1800)  # 30 minutos en segundos

                    messages.success(request, f"¡Bienvenido, {username}!")

                    # Crear respuesta y limpiar cookie de expiración
                    response = redirect("home")
                    response.delete_cookie('session_expired')
                    return response
                else:
                    messages.error(request, "Usuario o contraseña inválidos.")
            else:
                messages.error(request, "Usuario o contraseña inválidos.")
        else:
            form = AuthenticationForm()

            if session_expired:
                messages.warning(
                    request,
                    "Your session has expired due to inactivity. Please log in again."
                )

        response = render(request, "login.html", {"form": form})

        if session_expired:
            response.delete_cookie('session_expired')

        return response


def register_view(request):
    with translation.override("en"):
        if request.method == "POST":
            form = CustomUserCreationForm(request.POST)
            if form.is_valid():
                user = form.save()
                login(request, user)
                return redirect("home")
        else:
            form = CustomUserCreationForm()
        return render(request, "register.html", {"form": form})


def logout_view(request):
    if 'last_activity' in request.session:
        del request.session['last_activity']
    if 'login_time' in request.session:
        del request.session['login_time']


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


def build_chart_config_from_df(df: pd.DataFrame) -> dict | None:
    """
    Construye una config de Chart.js 'inteligente' según el DataFrame:
      - Si hay fechas: elige la mejor columna datetime, la mejor numérica (por varianza)
        y agrega por frecuencia adaptativa (D, W o M) según el rango total.
      - Si hay categórica + numérica: agrega y muestra TOP_N; el resto como 'Others'.
        Si categorías <= 6 => pie, si no => bar.
      - Si solo hay categórica: conteo de ocurrencias (TOP_N + 'Others').
    Devuelve None si no hay nada dibujable.
    """
    if df is None or df.empty:
        return None

    dfx = df.copy()

    # --- 1) Parseo agresivo de posibles fechas en columnas de tipo object ---
    for col in dfx.columns:
        if dfx[col].dtype == object:
            try:
                parsed = pd.to_datetime(dfx[col], errors="raise", infer_datetime_format=True, utc=False)
                dfx[col] = pd.to_datetime(dfx[col], errors="coerce")
            except Exception:
                pass

    # --- 2) Identificar tipos ---
    num_cols = [c for c in dfx.columns if pd.api.types.is_numeric_dtype(dfx[c])]
    dt_cols  = [c for c in dfx.columns if pd.api.types.is_datetime64_any_dtype(dfx[c])]

    # Categóricas candidatas: cardinalidad moderada (<= 50) y no num/dt
    cat_cols = []
    for c in dfx.columns:
        if c in num_cols or c in dt_cols:
            continue
        nunique = dfx[c].nunique(dropna=True)
        if 0 < nunique <= 50:
            cat_cols.append((c, nunique))
    # ordenar por cardinalidad ascendente (prioriza más compactas)
    cat_cols = [c for c, _ in sorted(cat_cols, key=lambda x: x[1])]

    # Helper: elegir mejor columna numérica por varianza (informativa)
    def best_numeric(cols):
        if not cols:
            return None
        if len(cols) == 1:
            return cols[0]
        variances = []
        for c in cols:
            s = dfx[c].dropna().astype(float)
            variances.append((c, float(s.var()) if len(s) else -1.0))
        variances.sort(key=lambda x: x[1], reverse=True)
        return variances[0][c := 0]  # devuelve nombre con mayor varianza

    # Helper: elegir mejor datetime por número de valores no nulos
    def best_datetime(cols):
        if not cols:
            return None
        if len(cols) == 1:
            return cols[0]
        scores = []
        for c in cols:
            s = dfx[c].dropna()
            scores.append((c, int(s.shape[0])))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    # --- 3) Caso serie temporal ---
    if dt_cols and num_cols:
        tcol = best_datetime(dt_cols)
        ncol = best_numeric(num_cols)
        if tcol and ncol:
            tmp = dfx[[tcol, ncol]].dropna()
            if not tmp.empty:
                # Rango para decidir frecuencia
                tmin, tmax = tmp[tcol].min(), tmp[tcol].max()
                if isinstance(tmin, pd.Timestamp) and isinstance(tmax, pd.Timestamp):
                    total_days = (tmax - tmin).days if pd.notna(tmax) and pd.notna(tmin) else None
                else:
                    total_days = None

                if total_days is None:
                    freq = "D"
                elif total_days <= 45:
                    freq = "D"
                elif total_days <= 360:
                    freq = "W"
                else:
                    freq = "M"

                agg = (
                    tmp.set_index(tcol)
                       .groupby(pd.Grouper(freq=freq))[ncol]
                       .sum()
                       .reset_index()
                       .dropna()
                )
                if agg.empty:
                    return None

                # Etiquetas legibles según freq
                if freq == "D":
                    labels = agg[tcol].dt.strftime("%Y-%m-%d").tolist()
                elif freq == "W":
                    labels = agg[tcol].dt.strftime("W%U %Y").tolist()
                else:  # "M"
                    labels = agg[tcol].dt.strftime("%Y-%m").tolist()

                data = agg[ncol].astype(float).round(2).tolist()

                return {
                    "type": "line",
                    "data": {
                        "labels": labels,
                        "datasets": [{
                            "label": f"{ncol} over time",
                            "data": data,
                            "tension": 0.25,
                            "pointRadius": 2,
                            "borderWidth": 2,
                            # sin colores fijos, Chart.js pondrá los suyos
                        }]
                    },
                    "options": {
                        "responsive": True,
                        "maintainAspectRatio": False,  # usamos nuestro alto de CSS
                        "plugins": {
                            "legend": {"display": True},
                            "tooltip": {"mode": "index", "intersect": False}
                        },
                        "scales": {
                            "x": {"title": {"display": True, "text": str(tcol)}},
                            "y": {"title": {"display": True, "text": str(ncol)}, "beginAtZero": True}
                        }
                    }
                }

    # --- 4) Categórica + numérica ---
    if cat_cols and num_cols:
        ccol = cat_cols[0]
        ncol = best_numeric(num_cols) or num_cols[0]
        tmp = dfx[[ccol, ncol]].dropna()
        if not tmp.empty:
            agg = tmp.groupby(ccol, dropna=True)[ncol].sum().reset_index()
            if agg.empty:
                return None
            agg = agg.sort_values(ncol, ascending=False)

            TOP_N = 10
            MAX_PIE = 6

            top = agg.head(TOP_N).copy()
            if len(agg) > TOP_N:
                others_val = float(agg.iloc[TOP_N:][ncol].sum())
                top = pd.concat([top, pd.DataFrame({ccol: ["Others"], ncol: [others_val]})], ignore_index=True)

            labels = top[ccol].astype(str).tolist()
            data = top[ncol].astype(float).round(2).tolist()

            if len(labels) <= MAX_PIE:
                chart_type = "pie"
                options = {"responsive": True, "plugins": {"legend": {"display": True}}}
            else:
                chart_type = "bar"
                options = {
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "plugins": {"legend": {"display": True}},
                    "scales": {
                        "x": {"title": {"display": True, "text": str(ccol)}},
                        "y": {"title": {"display": True, "text": str(ncol)}, "beginAtZero": True}
                    }
                }

            return {
                "type": chart_type,
                "data": {"labels": labels, "datasets": [{"label": f"{ncol} by {ccol}", "data": data}]},
                "options": options
            }

    # --- 5) Solo categórica => conteo ---
    if cat_cols and not num_cols:
        ccol = cat_cols[0]
        counts = dfx[ccol].dropna().astype(str).value_counts()
        if counts.empty:
            return None

        TOP_N = 10
        MAX_PIE = 6

        labels = counts.index.tolist()
        values = counts.values.tolist()

        if len(labels) > TOP_N:
            top_labels = labels[:TOP_N]
            top_vals = values[:TOP_N]
            others = sum(values[TOP_N:])
            top_labels.append("Others")
            top_vals.append(others)
            labels, values = top_labels, top_vals

        chart_type = "pie" if len(labels) <= MAX_PIE else "bar"
        options = {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"legend": {"display": True}},
        }
        if chart_type == "bar":
            options["scales"] = {"x": {"title": {"display": True, "text": str(ccol)}},
                                 "y": {"beginAtZero": True}}

        return {
            "type": chart_type,
            "data": {"labels": labels, "datasets": [{"label": f"Count by {ccol}", "data": values}]},
            "options": options
        }

    return None

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
    # Obtener el archivo solicitado
    uploaded_file = UploadedFile.objects.get(id=file_id)

    # Validar acceso: solo el propietario o un admin pueden analizar el archivo
    if not request.user.is_authenticated:
        messages.error(request, "You must be logged in to access this page.")
        return redirect("login")

    if not request.user.is_admin() and uploaded_file.user != request.user:
        messages.error(request, "You are not allowed to access this file.")
        return redirect("dashboard")

    # Si pasa la validación, continuar con el procesamiento
    file_path = uploaded_file.file.path
    ext = os.path.splitext(file_path)[1].lower()

    table_html = None
    stats = {}
    answer = None
    error = ""
    stats_checked = False
    # ---  valores por defecto para la gráfica ---
    chart_json = None 

    try:
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

        table_html = df.head(20).to_html(index=False, classes="data-table", border=0)

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

         # --- NUEVO: construir configuración para Chart.js ---
        chart_cfg = build_chart_config_from_df(df)
        chart_json = json.dumps(chart_cfg) if chart_cfg else None

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
            "chart_config": chart_json,          #
            "is_csv": (ext == ".csv"),           # 
            "is_database": False,                #
        },
    )


def read_excel_tables(file_path):
    xls = pd.ExcelFile(file_path)
    tablas = {}
    for sheet in xls.sheet_names:
        tablas[sheet] = pd.read_excel(xls, sheet_name=sheet)
    return tablas


def ask_chat_view(request, source_type, source_id):
    """
    Chat persistente que funciona tanto con archivos cargados como con conexiones externas.
    source_type: 'file' o 'db'
    source_id: id del UploadedFile o DataSource
    """
    initial_question = request.GET.get("question", "")
    
    # Obtener el modelo LLM seleccionado de la sesión (default: gemini)
    selected_llm = request.session.get("selected_llm", "gemini")

    # --- Determinar fuente ---
    if source_type == "file":
        source = get_object_or_404(UploadedFile, id=source_id, user=request.user)
        source_label = f"File: {source.name}"
        source_kind = "file"
    elif source_type == "db":
        source = get_object_or_404(DataSource, id=source_id, user=request.user)
        source_label = f"Base de datos: {source.name}"
        source_kind = "db"
    else:
        return HttpResponseBadRequest("Tipo de fuente inválido")

    # --- Historial de chat por sesión ---
    session_key = f"chat_history_{source_kind}_{source_id}"
    chat_history = request.session.get(session_key, [])

    if not chat_history:
        chat_history.append({
            "sender": "bot",
            "text": (
                f"Hello! You are connected to file: {source_label}. "
                "You can ask questions about the data, generate summaries, or run analysis."
            ),
            "is_json": False,
            "data": []
        })
        request.session[session_key] = chat_history

    # --- Cambio de modelo LLM ---
    if request.method == "POST" and "change_llm" in request.POST:
        new_llm = request.POST.get("selected_llm", "gemini")
        request.session["selected_llm"] = new_llm
        
        messages.success(request, f"LLM model changed to: {new_llm}")
        return redirect("ask_chat", source_type=source_kind, source_id=source_id)

    # --- Procesar pregunta del usuario ---
    if request.method == "POST" and "question" in request.POST:
        print("==== POST RECIBIDO ====")
        
        # IMPORTANTE: Obtener el modelo LLM del POST si viene (para requests AJAX)
        # Si no viene, usar el de la sesión
        if "current_llm" in request.POST:
            selected_llm = request.POST.get("current_llm", "gemini")
            print(f"🔍 Modelo LLM obtenido del POST: {selected_llm}")
        else:
            print(f"🔍 Modelo LLM obtenido de sesión: {selected_llm}")
        
        question = request.POST.get("question", "").strip()
        if question:
            chat_history.append({"sender": "user", "text": question})
            print("Pregunta recibida:", question)
            try:
                if source_kind == "file":
                    file_path = source.file.path
                    ext = os.path.splitext(file_path)[1].lower()
                    db_path = settings.DATABASES["default"]["NAME"]
                    engine = create_engine(f"sqlite:///{db_path}")
                    tablas_cargadas = []

                    if ext == ".csv":
                        df = pd.read_csv(file_path, encoding="utf-8-sig", sep=None, engine="python")
                        table_name = _sanitize_table_name(source.name)
                        df.to_sql(table_name, engine, if_exists="replace", index=False)
                        tablas_cargadas = [table_name]
                    elif ext in [".xls", ".xlsx"]:
                        hojas = read_excel_tables(file_path)
                        for hoja, df in hojas.items():
                            table_name = _sanitize_table_name(hoja)
                            df.to_sql(table_name, engine, if_exists="replace", index=False)
                            tablas_cargadas.append(table_name)
                    else:
                        raise ValueError("Formato de archivo no soportado.")

                    tablas_info = ", ".join(tablas_cargadas)
                    context_prompt = f"Tablas disponibles: {tablas_info}."

                    if any(k in question.lower() for k in ["resumen", "summary", "analiza", "analyze"]):
                        summaries = []
                        for tabla in tablas_cargadas:
                            s = ai_services.get_summary_from_data_unified(tabla, selected_llm)
                            summaries.append(f"**{tabla}**:\n{s}")
                        answer = "\n\n".join(summaries)
                        sql_query = None  # No hay SQL en resúmenes
                    else:
                        # Registrar tiempo de inicio
                        start_time = time.time()
                        
                        # Detectar qué tabla se menciona en la pregunta
                        target_table = tablas_cargadas[0]  # default: primera tabla
                        question_lower = question.lower()
                        for tabla in tablas_cargadas:
                            if tabla.lower() in question_lower:
                                target_table = tabla
                                break
                        
                        print(f"🎯 Tabla detectada: {target_table}")
                        
                        sql_query = ai_services.get_sql_from_question_unified(
                            f"{context_prompt}\n{question}", target_table, selected_llm
                        )
                        
                        # Calcular tiempo de respuesta del LLM
                        llm_response_time = time.time() - start_time
                        
                        if sql_query == "__NLP_ERROR__":
                            # Registrar métrica de fallo
                            LLMMetrics.objects.create(
                                user=request.user,
                                llm_model=selected_llm,
                                question=question,
                                success=False,
                                sql_generated=None,
                                response_time=llm_response_time,
                                result_count=0,
                                error_message="LLM no pudo generar SQL válido"
                            )
                            
                            bot_msg = {
                                "sender": "bot",
                                "text": (
                                    "🤖 No pude entender tu pregunta.\n\n"
                                    "💡 Intenta reformularla. Ejemplos:\n"
                                    "- Total de ventas por mes\n"
                                    "- Promedio de edad de empleados\n"
                                    "- Filtrar clientes por país\n"
                                ),
                                "is_json": False,
                                "data": []
                            }
                            chat_history.append(bot_msg)
                            request.session[session_key] = chat_history
                            request.session.modified = True
                            return JsonResponse(bot_msg, safe=False)
                    
                        with connection.cursor() as cursor:
                            cursor.execute(sql_query)
                            columns = [col[0] for col in cursor.description]
                            result = [
                                dict(zip(columns, row)) for row in cursor.fetchall()
                            ]
                        
                        # Registrar métrica de éxito
                        total_time = time.time() - start_time
                        LLMMetrics.objects.create(
                            user=request.user,
                            llm_model=selected_llm,
                            question=question,
                            success=True,
                            sql_generated=sql_query,
                            response_time=total_time,
                            result_count=len(result),
                            error_message=None
                        )
                        
                        answer = result if result else "No se encontraron resultados."

                else:
                    answer = ai_services.get_response_from_external_db(question, source, selected_llm)

                if isinstance(answer, list) and len(answer) > 0 and isinstance(answer[0], dict):
                    bot_msg = {
                        "sender": "bot",
                        "text": None,
                        "is_json": True,
                        "data": answer
                    }
                else:
                    bot_msg = {
                        "sender": "bot",
                        "text": str(answer),
                        "is_json": False,
                        "data": []
                    }

                chat_history.append(bot_msg)
                from .views import log_query
                log_query(request.user, question, bot_msg["text"] or bot_msg["data"], selected_llm)

            except Exception as e:
                # Registrar métrica de error
                if 'start_time' in locals():
                    error_time = time.time() - start_time
                else:
                    error_time = 0
                
                LLMMetrics.objects.create(
                    user=request.user,
                    llm_model=selected_llm,
                    question=question,
                    success=False,
                    sql_generated=sql_query if 'sql_query' in locals() else None,
                    response_time=error_time,
                    result_count=0,
                    error_message=str(e)
                )
                
                bot_msg = {
                    "sender": "bot",
                    "text": f"❌ Ocurrió un error al procesar tu solicitud: {str(e)}",
                    "is_json": False,
                    "data": []
                }
                         
                chat_history.append(bot_msg)
                from .views import log_query
                log_query(request.user, question, bot_msg["text"] or bot_msg["data"], selected_llm)

            finally:
                if source_kind == "file" and "engine" in locals() and "tablas_cargadas" in locals():
                    with engine.connect() as conn:
                        for t in tablas_cargadas:
                            conn.execute(text(f'DROP TABLE IF EXISTS "{t}"'))
                        conn.commit()

            request.session[session_key] = chat_history
            request.session.modified = True

            print("==== DEBUG RESPUESTA ====")
            print(bot_msg)
            print("==== FIN DEBUG ====")
            # --- NUEVO: Si es una petición AJAX, devolvemos solo el último mensaje ---
            if request.headers.get("x-requested-with", "").lower() == "xmlhttprequest":
                return JsonResponse(bot_msg, safe=False)
            # ... dentro del bloque POST, justo antes del return ...
            # Si no es AJAX
            return redirect("ask_chat", source_type=source_kind, source_id=source_id)

    # --- Renderizado inicial ---
    return render(request, "chat.html", {
        "source": source,
        "source_type": source_kind,
        "chat_history": chat_history,
        "initial_question": initial_question,
        "selected_llm": selected_llm,
        "llm_choices": [
            ("gemini", "Google Gemini"),
            ("huggingface", "Hugging Face"),
            ("openrouter", "OpenRouter"),
        ],
    })


@login_required
def dashboard_view(request):
    files = UploadedFile.objects.filter(user=request.user)
    return render(request, "dashboard.html", {"files": files})


@login_required
def llm_comparison_view(request):
    """
    Vista para comparar la precisión y rendimiento de los diferentes modelos LLM
    """
    from django.db.models import Count, Avg, Q
    
    # Obtener métricas del usuario actual
    user_metrics = LLMMetrics.objects.filter(user=request.user)
    
    # Estadísticas por modelo
    stats_by_model = {}
    for model_code, model_name in LLMMetrics.LLM_CHOICES:
        model_metrics = user_metrics.filter(llm_model=model_code)
        total_queries = model_metrics.count()
        
        if total_queries > 0:
            successful = model_metrics.filter(success=True).count()
            failed = model_metrics.filter(success=False).count()
            success_rate = (successful / total_queries) * 100
            avg_response_time = model_metrics.aggregate(Avg('response_time'))['response_time__avg'] or 0
            avg_results = model_metrics.filter(success=True).aggregate(Avg('result_count'))['result_count__avg'] or 0
            
            stats_by_model[model_code] = {
                'name': model_name,
                'total': total_queries,
                'successful': successful,
                'failed': failed,
                'success_rate': round(success_rate, 2),
                'avg_response_time': round(avg_response_time, 3),
                'avg_results': round(avg_results, 1)
            }
    
    # Últimas consultas con detalles
    recent_queries = user_metrics.order_by('-timestamp')[:20]
    
    # Errores comunes por modelo
    common_errors = {}
    for model_code, model_name in LLMMetrics.LLM_CHOICES:
        errors = user_metrics.filter(
            llm_model=model_code, 
            success=False, 
            error_message__isnull=False
        ).values('error_message').annotate(count=Count('error_message')).order_by('-count')[:5]
        common_errors[model_code] = errors
    
    context = {
        'stats_by_model': stats_by_model,
        'recent_queries': recent_queries,
        'common_errors': common_errors,
    }
    
    return render(request, "llm_comparison.html", context)


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
        return str(
            URL.create(
                drivername="postgresql+psycopg2",
                username=ds.username,
                password=ds.password,
                host=ds.host,
                port=int(ds.port) if ds.port else None,
                database=ds.db_name,
            )
        )
    elif ds.engine == "mysql":
        # driver PyMySQL
        return str(
            URL.create(
                drivername="mysql+pymysql",
                username=ds.username,
                password=ds.password,
                host=ds.host,
                port=int(ds.port) if ds.port else None,
                database=ds.db_name,
            )
        )
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
                DataSource.objects.filter(user=request.user, is_active=True).update(
                    is_active=False
                )

            # Probar conexión
            try:
                # Para probar, usamos un DataSource "temporal" no guardado
                test_url = _build_sqlalchemy_url(ds)
                engine = create_engine(test_url)
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                # Si llegamos aquí, la conexión funciona
                ds.save()
                messages.success(
                    request, "Connection successful and configuration saved."
                )
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
            DataSource.objects.filter(user=request.user, is_active=True).update(
                is_active=False
            )
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
    df = df.dropna(axis=1, how="all")

    sensitive_patterns = [
        r"password",
        r"pass",
        r"pwd",
        r"token",
        r"secret",
        r"auth",
        r"apikey",
        r"api_key",
        r"ssn",
        r"credit",
        r"card",
        r"email",
        r"user_?id",
        r"login",
    ]
    cols_to_drop = [
        col
        for col in df.columns
        if any(re.search(pattern, col.lower()) for pattern in sensitive_patterns)
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors="ignore")

    # --- 3. Truncate long text values ---
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = (
            df[col]
            .astype(str)
            .apply(
                lambda x: (x[:max_text_length] + "…") if len(x) > max_text_length else x
            )
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
                f"{len(hidden_cols)} sensitive column(s) were hidden for security reasons.",
            )

        # Mostrar solo las primeras 4 columnas para evitar desbordes
        df_preview = df.iloc[:, :4]

        # Convertir a HTML para mostrar
        table_html = df_preview.head(20).to_html(
            index=False, classes="data-table", border=0
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
            "table_html": table_html or "",
            "stats": stats or {},
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

    html = template.render(
        {
            "export": data,
            "logo_b64": logo_b64,
        }
    )

    # Convertir a PDF en memoria
    result = io.BytesIO()
    pdf = pisa.CreatePDF(
        io.BytesIO(html.encode("utf-8")), dest=result, encoding="utf-8"
    )
    if pdf.err:
        return HttpResponseBadRequest("Error generando el PDF")

    # Descargar
    filename = (data.get("file_name") or "DataScope_Report").replace('"', "")
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

    return render(
        request,
        "select_table.html",
        {"tables": tables, "source_id": source_id, "source_type": source_type},
    )


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
        return render(
            request, "show_table.html", {"table_html": html, "table_name": table}
        )



@login_required
def favorite_questions_view(request):
    if request.method == "POST":
        form = FavoriteQuestionForm(request.POST)
        if form.is_valid():
            favorite = form.save(commit=False)
            favorite.user = request.user
            favorite.save()
            messages.success(request, "Favorite question saved!")
            return redirect("favorite_questions")
    else:
        form = FavoriteQuestionForm()

    favorites = FavoriteQuestion.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "favorite_questions.html", {"favorites": favorites, "form": form})


@login_required
@require_http_methods(["POST"])
def delete_favorite_question_view(request, question_id):
    question = get_object_or_404(FavoriteQuestion, id=question_id, user=request.user)
    question.delete()
    messages.success(request, "Favorite question deleted.")
    return redirect("favorite_questions")


@login_required
def use_favorite_question_view(request, question_id):
    question = get_object_or_404(FavoriteQuestion, id=question_id, user=request.user)
    
    # Find the last source the user interacted with to redirect them back
    # This is a simple approach; a more robust solution might store the
    # last active source in the session.
    last_source = None
    if 'last_upload_context' in request.session:
        # This is a custom session key I see in the code
        # It seems to be related to the analyze_db_view
        # I will assume it has the source info.
        # A better approach would be to have a consistent way of storing the last source.
        pass # I will leave this for now and redirect to dashboard if no source is found

    # For now, let's just redirect to the dashboard, and the user can select the source again.
    # A better implementation would be to redirect to the last used source.
    
    # The user will be redirected to the dashboard, and from there they can select a source
    # and the question will be in the chat.
    # I will modify the ask_chat_view to handle the question from the URL.

    # I will construct the redirect URL to the chat page of the last active source
    # For now, I will just redirect to the dashboard.
    # I will need to find a way to get the last active source.

    # Looking at the code, there is a `is_active` field in the `DataSource` model.
    # I will use that to find the active source.
    active_source = DataSource.objects.filter(user=request.user, is_active=True).first()

    if active_source:
        return redirect(f"/ask/db/{active_source.id}/?question={question.question_text}")
    else:
        # If no active DB source, try to find the last uploaded file
        last_file = UploadedFile.objects.filter(user=request.user).order_by("-uploaded_at").first()
        if last_file:
            return redirect(f"/ask/file/{last_file.id}/?question={question.question_text}")

    messages.info(request, "Please select a data source to use this favorite question.")
    return redirect("dashboard")


def log_query(user, question, result, llm_model="gemini"):
    """Guarda en BD cada pregunta y su respuesta resumida."""
    if user.is_authenticated:
        from .models import Query
        Query.objects.create(
            user=user,
            query_text=question,
            ai_response=str(result)[:400],
            llm_model=llm_model
        )
        # También guardar en QueryHistory para mantener compatibilidad
        QueryHistory.objects.create(
            user=user,
            question=question,
            result_preview=str(result)[:400]
        )


@login_required
def history_view(request):
    """Vista para ver el historial de preguntas y respuestas del usuario."""
    history = QueryHistory.objects.filter(user=request.user)
    paginator = Paginator(history, 8)  # 8 registros por página
    page = request.GET.get("page")
    page_obj = paginator.get_page(page)
    return render(request, "history.html", {"page_obj": page_obj})


@login_required
def data_sources_view(request):
    """
    Vista principal para seleccionar y cambiar entre fuentes de datos.
    Muestra tanto archivos subidos como conexiones de bases de datos.
    """
    # Obtener todas las fuentes del usuario
    uploaded_files = UploadedFile.objects.filter(user=request.user).order_by('-uploaded_at')
    db_connections = DataSource.objects.filter(user=request.user).order_by('-created_at')

    # Obtener la fuente activa actual (si existe)
    active_db = db_connections.filter(is_active=True).first()

    # Contar totales
    total_sources = uploaded_files.count() + db_connections.count()

    # Información de la fuente seleccionada (si hay una)
    selected_source = None
    selected_type = None
    source_info = None

    # Verificar si se seleccionó una fuente mediante GET
    source_id = request.GET.get('source_id')
    source_type = request.GET.get('source_type')  # 'file' o 'db'

    if source_id and source_type:
        if source_type == 'file':
            try:
                selected_source = UploadedFile.objects.get(id=source_id, user=request.user)
                selected_type = 'file'
                # Obtener información básica del archivo
                source_info = get_file_info(selected_source)
            except UploadedFile.DoesNotExist:
                messages.error(request, "File not found.")

        elif source_type == 'db':
            try:
                selected_source = DataSource.objects.get(id=source_id, user=request.user)
                selected_type = 'db'
                # Obtener información básica de la base de datos
                source_info = get_db_info(selected_source)
            except DataSource.DoesNotExist:
                messages.error(request, "Database connection not found.")

    context = {
        'uploaded_files': uploaded_files,
        'db_connections': db_connections,
        'active_db': active_db,
        'total_sources': total_sources,
        'selected_source': selected_source,
        'selected_type': selected_type,
        'source_info': source_info,
    }

    return render(request, 'data_sources.html', context)


def get_file_info(uploaded_file):
    """
    Obtiene información básica de un archivo sin cargarlo completamente.
    """
    info = {
        'name': uploaded_file.name,
        'type': 'CSV File' if uploaded_file.name.lower().endswith('.csv') else 'Excel File',
        'uploaded_at': uploaded_file.uploaded_at,
        'status': 'Active',
        'row_count': None,
        'column_count': None,
    }

    try:
        file_path = uploaded_file.file.path
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.csv':
            # Leer solo las primeras filas para obtener info
            df = pd.read_csv(file_path, nrows=0)
            info['column_count'] = len(df.columns)

            # Contar filas (más eficiente)
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                info['row_count'] = sum(1 for line in f) - 1  # -1 para el header

        elif ext in ['.xls', '.xlsx']:
            xls = pd.ExcelFile(file_path)
            info['type'] = f'Excel File ({len(xls.sheet_names)} sheets)'
            # Leer primera hoja para info básica
            df = pd.read_excel(file_path, nrows=0)
            info['column_count'] = len(df.columns)

    except Exception as e:
        info['error'] = str(e)

    return info


def get_db_info(data_source):
    """
    Obtiene información básica de una conexión de base de datos.
    """
    info = {
        'name': data_source.name,
        'type': data_source.get_engine_display(),
        'created_at': data_source.created_at,
        'status': 'Active' if data_source.is_active else 'Inactive',
        'host': data_source.host if data_source.engine != 'sqlite' else 'Local file',
        'database': data_source.db_name if data_source.engine != 'sqlite' else data_source.sqlite_path,
        'table_count': None,
    }

    try:
        # Intentar conectar y obtener info de tablas
        db_url = _build_sqlalchemy_url(data_source)
        engine = create_engine(db_url)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        info['table_count'] = len(tables)
        info['connection_status'] = 'Connected'
    except Exception as e:
        info['connection_status'] = 'Connection failed'
        info['error'] = str(e)

    return info


@login_required
@require_http_methods(["POST"])
def set_active_source_view(request):
    """
    Establece una fuente de datos como activa y redirige a su análisis.
    """
    source_type = request.POST.get('source_type')
    source_id = request.POST.get('source_id')

    if not source_type or not source_id:
        messages.error(request, "Invalid request.")
        return redirect('data_sources')

    if source_type == 'file':
        try:
            uploaded_file = UploadedFile.objects.get(id=source_id, user=request.user)
            messages.success(request, f"Now analyzing: {uploaded_file.name}")
            return redirect('analyze_file', file_id=uploaded_file.id)
        except UploadedFile.DoesNotExist:
            messages.error(request, "File not found.")
            return redirect('data_sources')

    elif source_type == 'db':
        try:
            db_source = DataSource.objects.get(id=source_id, user=request.user)
                        # Desactivar otras conexiones
            DataSource.objects.filter(user=request.user, is_active=True).update(is_active=False)

            # Activar la seleccionada
            db_source.is_active = True
            db_source.save()

            messages.success(request, f"Now connected to: {db_source.name}")
            return redirect('analyze_db', db_id=db_source.id)
        except DataSource.DoesNotExist:
            messages.error(request, "Database connection not found.")
            return redirect('data_sources')

    else:
        messages.error(request, "Invalid source type.")
        return redirect('data_sources')


@login_required
def quick_switch_view(request):
    """
    Vista rápida para cambiar entre fuentes sin recargar completamente.
    Usada desde el navbar o header.
    """
    if request.method == 'POST':
        source_type = request.POST.get('source_type')
        source_id = request.POST.get('source_id')

        if source_type == 'file':
            return redirect('analyze_file', file_id=source_id)
        elif source_type == 'db':
            # Activar la conexión
            try:
                db_source = DataSource.objects.get(id=source_id, user=request.user)
                DataSource.objects.filter(user=request.user, is_active=True).update(is_active=False)
                db_source.is_active = True
                db_source.save()
                return redirect('analyze_db', db_id=source_id)
            except DataSource.DoesNotExist:
                messages.error(request, "Database not found.")

    return redirect('data_sources')