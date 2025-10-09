"""
URL configuration for DataScope project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from Data import views

urlpatterns = [
    path("", views.home, name="home"),
    path("admin/", admin.site.urls),
    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", views.logout_view, name="logout"),
    path("upload/", views.upload_file_view, name="upload_file"),
    path("analyze/<int:file_id>/", views.analyze_file_view, name="analyze_file"),
    path("ask/<int:file_id>/", views.ask_question_view, name="ask_question"),
    path("dashboard/", views.dashboard_view, name="dashboard"),
    path("admin-dashboard/", views.admin_dashboard_view, name="admin_dashboard"),
    path("connect/", views.connect_db_view, name="connect_db"),
    path("connections/", views.connections_list_view, name="connections_list"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)