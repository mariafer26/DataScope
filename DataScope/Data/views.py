from django.shortcuts import render
from django.shortcuts import render
from django.contrib import messages
from .forms import UploadFileForm
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
import os
import pandas as pd

def home(request):
    return render(request, 'base.html')

def login_view(request):
    return render(request, 'login.html')
def register_view(request):
    return render(request, 'register.html')

def upload_file_view(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            
            # Guardar en carpeta 'media/uploads'
            upload_path = os.path.join('media', 'uploads')
            os.makedirs(upload_path, exist_ok=True)
            file_path = os.path.join(upload_path, file.name)

            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            # Procesar archivo con pandas
            ext = os.path.splitext(file.name)[1].lower()
            try:
                if ext == '.csv':
                    df = pd.read_csv(file_path)
                elif ext == '.xlsx':
                    df = pd.read_excel(file_path)

                # Aquí puedes hacer validaciones extra o guardar en DB
                messages.success(request, f"File uploaded successfully! {len(df)} rows loaded.")

            except Exception as e:
                messages.error(request, f"Error processing file: {str(e)}")

    else:
        form = UploadFileForm()

    return render(request, 'upload.html', {'form': form})
