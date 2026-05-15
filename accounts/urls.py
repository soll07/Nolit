from django.urls import path
from . import views
<<<<<<< HEAD
 
app_name = "accounts"
 
urlpatterns = [
    path("signup/",  views.signup,       name="signup"),
    path("login/",   views.login_view,   name="login"),
    path("logout/",  views.logout_view,  name="logout"),
    path("persona/", views.persona,      name="persona"),
]