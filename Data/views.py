from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import UploadFileForm
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from .user_forms import CustomUserCreationForm
import os
import pandas as pd


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
                return redirect('home')
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


def upload_file_view(request):
    """
    Sube archivo, guarda en media/uploads, lee con pandas
    y calcula métricas básicas para columnas numéricas.
    """
    table_html = None
    stats = {}
    stats_checked = False

    
    answer = None   # puede ser texto o una tabla (list[dict] / list[list])
    loading = False
    error = ""

    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            upload_path = os.path.join('media', 'uploads')
            os.makedirs(upload_path, exist_ok=True)
            file_path = os.path.join(upload_path, file.name)

            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            ext = os.path.splitext(file.name)[1].lower()
            try:
                if ext == '.csv':
                    try:
                        df = pd.read_csv(file_path, encoding='utf-8-sig', sep=None, engine='python')
                    except Exception:
                        df = pd.read_csv(file_path, encoding='latin-1', sep=None, engine='python')
                elif ext == '.xlsx':
                    df = pd.read_excel(file_path)
                else:
                    raise ValueError("Unsupported file extension")

                messages.success(
                    request,
                    f"File uploaded successfully! {len(df)} rows loaded."
                )

                table_html = df.head(20).to_html(index=False, classes="data-table", border=0)

                numeric_df = df.select_dtypes(include='number')
                stats_checked = True

                if numeric_df.empty:
                    df_coerced = df.copy()

                    def _coerce_to_numeric(series: pd.Series) -> pd.Series:
                        t = series.astype(str).str.replace(r'\s|\u00A0', '', regex=True)
                        num1 = pd.to_numeric(t, errors='coerce')
                        t_alt = t.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
                        num2 = pd.to_numeric(t_alt, errors='coerce')
                        return num2 if num2.notna().sum() > num1.notna().sum() else num1

                    for col in df_coerced.columns:
                        if df_coerced[col].dtype == 'object':
                            coerced = _coerce_to_numeric(df_coerced[col])
                            if coerced.notna().any():
                                df_coerced[col] = coerced

                    numeric_df = df_coerced.select_dtypes(include='number')

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
                            'mean': r(s.mean()),
                            'median': r(s.median()),
                            'min': r(s.min()),
                            'max': r(s.max()),
                            'count': int(s.count())
                        }
                else:
                    messages.info(request, "No numeric columns were detected in the file.")

                
                # Resumen por columna: tipo y cantidad de nulos (tabla)
                answer = [
                    {
                        "Column": str(c),
                        "Type": str(df[c].dtype),
                        "Nulls": int(df[c].isna().sum()),
                    }
                    for c in df.columns
                ]

            except Exception as e:
                messages.error(request, f"Error processing file: {str(e)}")
                error = str(e)   #mostrar en ResultViewer
    else:
        form = UploadFileForm()

    return render(
        request,
        'upload.html',
        {
            'form': form,
            'table_html': table_html,
            'stats': stats,
            'stats_checked': stats_checked,
            
            'answer': answer,
            'loading': loading,
            'error': error,
        }
    )