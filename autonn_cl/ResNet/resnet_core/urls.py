"""autonn/Resnet/resnet_core/urls.py
This file contains the url patterns for the ResNet app.
"""

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('resnet_status', views.InfoList, name='InfoList'),
    path('start', views.start, name='StartResNet'),
    path('stop', views.stop, name='StopResNet'),
    path('status_request', views.status_request, name='StatusRequestResNet'),
    path('get_ready_for_test', views.get_ready_for_test, name='GetReadyForTest'),
]