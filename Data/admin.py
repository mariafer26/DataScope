from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser, UploadedFile, Query


# Personaliza cómo se mostrará el usuario en el admi
class CustomUserAdmin(UserAdmin):
    model = CustomUser
    list_display = ("username", "email", "role", "is_staff", "is_active")
    list_filter = ("role", "is_staff", "is_active")
    fieldsets = (
        (None, {"fields": ("username", "email", "password")}),
        ("Permissions", {"fields": ("role", "is_staff", "is_active", "is_superuser")}),
        ("Important dates", {"fields": ("last_login", "date_joined")}),
    )
    add_fieldsets = (
        (
            None,
            {
                "classes": ("wide",),
                "fields": (
                    "username",
                    "email",
                    "role",
                    "password1",
                    "password2",
                    "is_staff",
                    "is_active",
                ),
            },
        ),
    )
    search_fields = ("username", "email")
    ordering = ("username",)


# Registra los modelos en el admin
admin.site.register(CustomUser, CustomUserAdmin)
admin.site.register(UploadedFile)
admin.site.register(Query)
