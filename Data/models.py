from django.db import models
from django.conf import settings
from django.contrib.auth.models import AbstractUser


# Modelo de usuario extendido
class CustomUser(AbstractUser):
    ROLE_CHOICES = (
        ("ADMIN", "Administrator"),
        ("STANDARD", "Standard User"),
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default="STANDARD")

    def is_admin(self):
        return self.role == "ADMIN"

    def is_standard(self):
        return self.role == "STANDARD"


# Modelo de consultas (Query)
class Query(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    query_text = models.TextField()
    ai_response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Query by {self.user.username} at {self.timestamp}"


class FavoriteQuestion(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    question_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Favorite question by {self.user.username}: "{self.question_text[:50]}..."'


# Modelo de archivos subidos
class UploadedFile(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="uploaded_files",
    )
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to="uploads/", null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} subido por {self.user.username}"


class DataSource(models.Model):
    ENGINE_CHOICES = (
        ("postgresql", "PostgreSQL"),
        ("mysql", "MySQL"),
        ("sqlite", "SQLite (file path)"),
    )

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="data_sources"
    )
    name = models.CharField(max_length=100, help_text="Friendly name, e.g. 'Sales DB'")
    engine = models.CharField(max_length=20, choices=ENGINE_CHOICES)

    # Campos para Postgres/MySQL
    host = models.CharField(max_length=255, blank=True, null=True)
    port = models.CharField(max_length=10, blank=True, null=True)
    db_name = models.CharField(max_length=255, blank=True, null=True)
    username = models.CharField(max_length=255, blank=True, null=True)
    password = models.CharField(max_length=255, blank=True, null=True)

    # Campo para SQLite
    sqlite_path = models.CharField(max_length=500, blank=True, null=True)

    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self):
        return f"{self.name} ({self.engine})"


class QueryHistory(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="query_history")
    question = models.TextField()
    result_preview = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user.username} - {self.question[:50]}..."
