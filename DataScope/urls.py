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
from django.urls import path, reverse_lazy
from django.conf import settings
from django.conf.urls.static import static
from Data import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path("", views.home, name="home"),
    path("admin/", admin.site.urls),
    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", views.logout_view, name="logout"),
    path("upload/", views.upload_file_view, name="upload_file"),
    path("analyze/<int:file_id>/", views.analyze_file_view, name="analyze_file"),
    path("ask/<str:source_type>/<int:source_id>/", views.ask_chat_view, name="ask_chat"),
    path("export/pdf/", views.export_pdf_view, name="export_pdf"),
    path("dashboard/", views.dashboard_view, name="dashboard"),
    path("admin-dashboard/", views.admin_dashboard_view, name="admin_dashboard"),
    path("connect/", views.connect_db_view, name="connect_db"),
    path("analyze-db/<int:db_id>/", views.analyze_db_view, name="analyze_db"),
    path("connections/", views.connections_list_view, name="connections_list"),
    path("history/", views.history_view, name="history"),
    path("llm-comparison/", views.llm_comparison_view, name="llm_comparison"),
    path(
        "select-table/<str:source_type>/<int:source_id>/",
        views.select_table_view,
        name="select_table",
    ),
    path("show-table/", views.show_table_view, name="show_table"),
    # Favorite questions
    path("favorites/", views.favorite_questions_view, name="favorite_questions"),
    path("favorites/delete/<int:question_id>/", views.delete_favorite_question_view, name="delete_favorite_question"),
    path("favorites/use/<int:question_id>/", views.use_favorite_question_view, name="use_favorite_question"),
    # Password recovery
    path('password_reset/', auth_views.PasswordResetView.as_view(template_name='password_reset.html', success_url=reverse_lazy('password_reset_done')), name='password_reset'),
    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(template_name='password_reset_done.html'), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='password_reset_confirm.html', success_url=reverse_lazy('password_reset_complete')), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name='password_reset_complete.html'), name='password_reset_complete'),
    path("data-sources/", views.data_sources_view, name="data_sources"),
    path("set-active-source/", views.set_active_source_view, name="set_active_source"),
    path("quick-switch/", views.quick_switch_view, name="quick_switch"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
