from django.shortcuts import render
from django.shortcuts import render
from django.contrib import messages
from .forms import UploadFileForm
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
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
                    df = pd.read_csv(file_path)
                elif ext == '.xlsx':
                    df = pd.read_excel(file_path)
                messages.success(request, f"File uploaded successfully! {len(df)} rows loaded.")

            except Exception as e:
                messages.error(request, f"Error processing file: {str(e)}")

    else:
        form = UploadFileForm()

    return render(request, 'upload.html', {'form': form})
