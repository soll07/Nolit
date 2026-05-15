from django.urls import path
from . import views
 
app_name = "planner"
 
urlpatterns = [
    path("",           views.planner_index,                    name="index"),
    path("calculate/", views.PlannerCalculateView.as_view(),   name="calculate"),
]