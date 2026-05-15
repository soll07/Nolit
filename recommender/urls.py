from django.urls import path
from .views import ChatView, RecommendView
 
app_name = "recommender"
 
urlpatterns = [
    path("chat/",      ChatView.as_view(),      name="chat"),
    # path("recommend/", RecommendView.as_view(), name="recommend"),
]