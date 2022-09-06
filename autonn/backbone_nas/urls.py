'''
urls.py
'''

from django.urls import path

from . import views

# from rest_framework import routers

app_name = 'backboneNAS'

# router = routers.DefaultRouter()
# router.register('URS', views.URSView, basename='user_reqs')

urlpatterns = [
    path('', views.index),
    # path(r'^', include(router.urls), name = ""),
    path("api/urs", views.URSList, name="URSList"),
    path("api/backbone", views.create_net, name="create_net")
]
