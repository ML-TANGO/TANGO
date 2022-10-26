'''
urls.py
'''

from django.urls import path

from . import views

app_name = 'bnas'

urlpatterns = [
    path('', views.index),
    path("infos", views.InfoList, name="InfoList"),
    path("start", views.start, name="StartBNAS"),
    path("stop", views.stop, name="StopBNAS"),
    path("status_request", views.status_request, name="StatusRequestBNAS"),
    # unit test
    path("create_net", views.create_net, name="RunBNAS")
]
