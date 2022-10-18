'''
urls.py
'''

from django.urls import path

from . import views

# from rest_framework import routers

app_name = 'bnas'

# router = routers.DefaultRouter()
# router.register('URS', views.URSView, basename='user_reqs')

urlpatterns = [
    path('', views.index),
    path("infos", views.InfoList, name="InfoList"),
    path("start", views.start, name="StartBNAS"),
    path("stop", views.stop, name="StopBNAS"),
    path("status_request", views.status_request, name="StatusRequestBNAS")
]
