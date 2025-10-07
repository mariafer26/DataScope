from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser 


class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True, label="Correo electrónico")

    class Meta:
        model = CustomUser
        fields = ["username", "email", "password1", "password2"]

    def save(self, commit=True):
        user = super().save(commit=False)
        # Asigna automáticamente el rol "STANDARD" a nuevos registros
        user.role = "STANDARD"
        if commit:
            user.save()
        return user
