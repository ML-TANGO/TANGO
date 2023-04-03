from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('yoloe_status', views.InfoList, name='InfoList'),
    path('start', views.start, name='StartYoloE'),
    path('stop', views.stop, name='StopYoloE'),
    path('status_request', views.status_request, name='StatusRequestYoloE'),
]

