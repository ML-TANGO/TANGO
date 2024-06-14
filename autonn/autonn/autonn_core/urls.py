from django.urls import path
from . import views

urlpatterns = [
    path('autonn_list', views.InfoList, name='InfoList'),
    path('start', views.start, name='StartAutoNN'),
    path('status_request', views.status_request, name='StatusRequestAutoNN'),
]

