from django.utils import timezone
from django.conf import settings
from datetime import timedelta
from django.contrib.auth import logout
from django.contrib import messages


class SessionActivityMiddleware:
    """
    Middleware que:
    1. Detecta cuando una sesión ha expirado
    2. Renueva la sesión si el usuario está activo
    3. Muestra mensaje cuando la sesión expira
    """

    def __init__(self, get_response):
        self.get_response = get_response
        # Intervalo mínimo entre renovaciones (5 minutos)
        self.renewal_interval = timedelta(seconds=30)
        # Tiempo máximo de inactividad (30 minutos)
        self.max_inactive_time = timedelta(seconds=settings.SESSION_COOKIE_AGE)
    
    def __call__(self, request):
        # Solo verificar si el usuario ESTABA autenticado
        was_authenticated = request.user.is_authenticated
        
        # Si el usuario está autenticado, verificar expiración
        if was_authenticated:
            expired = self.check_and_handle_expiry(request)
            
            if not expired:
                # Si no expiró, actualizar actividad
                self.update_session_activity(request)
        
        # Ejecutar la vista
        response = self.get_response(request)
        
        return response
    
    def check_and_handle_expiry(self, request):
        """
        Verifica si la sesión expiró por inactividad.
        Si expiró, hace logout y marca para mostrar mensaje.
        """
        now = timezone.now()
        last_activity_str = request.session.get('last_activity')
        
        # Si no hay timestamp de última actividad, crear uno
        if not last_activity_str:
            request.session['last_activity'] = now.isoformat()
            request.session.modified = True
            return False
        
        try:
            last_activity = timezone.datetime.fromisoformat(last_activity_str)
            time_inactive = now - last_activity
            
            # Si han pasado más de 30 minutos sin actividad
            if time_inactive > self.max_inactive_time:
                print(f"⚠️ Session expired for user: {request.user.username}")
                print(f"   Last activity: {last_activity}")
                print(f"   Time inactive: {time_inactive}")
                
                # Guardar el username antes de hacer logout
                username = request.user.username
                
                # Hacer logout del usuario
                logout(request)
                
                # Guardar mensaje en una cookie temporal (ya que la sesión se destruyó)
                # Lo manejaremos en la vista de login
                request.COOKIES['session_expired'] = 'true'
                
                return True
                
        except (ValueError, TypeError) as e:
            print(f"Error parsing last_activity: {e}")
            pass
        
        return False
    
    def update_session_activity(self, request):
        """
        Actualiza el timestamp de última actividad.
        """
        now = timezone.now()
        session = request.session
        
        # Obtener última actividad registrada
        last_activity_str = session.get('last_activity')
        
        # Si no existe o si han pasado más de 5 minutos, renovar
        should_renew = True
        if last_activity_str:
            try:
                last_activity = timezone.datetime.fromisoformat(last_activity_str)
                time_since_activity = now - last_activity
                
                # Solo renovar si han pasado más de 5 minutos
                should_renew = time_since_activity > self.renewal_interval
            except (ValueError, TypeError):
                should_renew = True
        
        if should_renew:
            # Actualizar timestamp de última actividad
            session['last_activity'] = now.isoformat()
            session.modified = True
            
            print(f"✓ Session renewed for user: {request.user.username} at {now}")