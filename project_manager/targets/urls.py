"""urls module for tango
This module for admin.
Attributes:

Todo:
"""

from django.urls import path, include,re_path

from . import views

urlpatterns = [
    re_path(r'^target_create', views.target_create, name='target_create'), 
    re_path(r'^target_read', views.target_read, name='target_read'), 
    re_path(r'^target_update', views.target_update, name='target_update'), 
    re_path(r'^target_delete', views.target_delete, name='target_delete'), 
    re_path(r'^target_info', views.target_info, name='target_info'), 
]

