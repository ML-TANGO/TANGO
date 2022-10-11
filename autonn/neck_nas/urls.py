'''
urls.py
'''

from django.urls import path

from . import views

app_name = 'neck_nas'

urlpatterns = [
    # path('', views.index),
    # path("api/neck_info", views.InfoList, name="information_for_nas"),
    # path("api/neck", views.create_net, name="create_net"),
    path("start", views.start, name="start_neck_nas"),
    path("stop", views.stop, name="stop_neck_nas"),
    path("status-request", views.status_request, name="status_request")
]
