from django.urls import path

from . import views

urlpatterns = [
    path("chat/", views.ChatView.as_view(), name="chat"),
    path("health/", views.health_check, name="health"),
]
