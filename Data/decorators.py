from django.shortcuts import redirect
from django.contrib import messages

def admin_required(view_func):
    #Permite acceso solo a usuarios con rol ADMIN
    def _wrapped_view(request, *args, **kwargs):
        if request.user.is_authenticated and getattr(request.user, "role", None) == "ADMIN":
            return view_func(request, *args, **kwargs)
        messages.error(request, "No tienes permiso para acceder a esta página.")
        return redirect("home")
    return _wrapped_view
