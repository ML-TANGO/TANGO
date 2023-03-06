'''
urls.py
'''

from django.urls import path

from . import views

app_name = 'neckNAS'

urlpatterns = [
    path('', views.index),
    path("nas_status", views.InfoList, name="InfoList"),
    path("start", views.start, name="StartNeckNAS"),
    path("stop", views.stop, name="StopNeckNAS"),
    path("status_request", views.status_request, name="StatusRequestNeckNAS"),
    path("get_ready_for_test", views.get_ready_for_test, name="GetReadyForTest")
]
