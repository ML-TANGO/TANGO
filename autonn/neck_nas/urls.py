'''
urls.py
'''

from django.urls import path

from . import views

app_name = 'neck_nas'

urlpatterns = [
    path('', views.index),
    path("nas_status", views.InfoList, name="InfoList"),
    # path("api/neck", views.create_net, name="create_net"),
    path("start", views.start, name="StartNeckNAS"),
    path("stop", views.stop, name="StopNeckNAS"),
    path("status_request", views.status_request, name="StatusRequestNeckNAS")
]
