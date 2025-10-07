from django.db import models
from django.conf import settings
from django.contrib.auth.models import AbstractUser


# Modelo de usuario extendido
class CustomUser(AbstractUser):
    ROLE_CHOICES = (
        ('ADMIN', 'Administrator'),
        ('STANDARD', 'Standard User'),
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='STANDARD')

    def is_admin(self):
        return self.role == 'ADMIN'

    def is_standard(self):
        return self.role == 'STANDARD'



# Modelo de consultas (Query)
class Query(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    query_text = models.TextField()
    ai_response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Query by {self.user.username} at {self.timestamp}"



# Modelo de archivos subidos
class UploadedFile(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="uploaded_files"
    )
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to="uploads/", null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} subido por {self.user.username}"
