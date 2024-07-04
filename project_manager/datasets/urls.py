"""urls module for tango
This module for admin.
Attributes:

Todo:
"""

from django.urls import path, include,re_path

from . import views


urlpatterns = [
    re_path(r'^get_dataset_list', views.get_dataset_list, name='get_dataset_list'), 
    re_path(r'^get_dataset_info', views.get_dataset_info, name='get_dataset_info'), 
    re_path(r'^get_folders_size', views.get_folders_size, name='get_folders_size'), 
    re_path(r'^get_folders_file_count', views.get_folders_file_count, name='get_folders_file_count'), 

    re_path(r'^download_coco', views.download_coco, name='download_coco'), 
    re_path(r'^download_imagenet', views.download_imagenet, name='download_imagenet'), 
    re_path(r'^download_voc', views.download_voc, name='download_voc'), 
    re_path(r'^download_chest_xray_dataset', views.download_chest_xray_dataset, name='download_chest_xray_dataset'), 
    re_path(r'^is_exist_user_kaggle_json', views.is_exist_user_kaggle_json, name='is_exist_user_kaggle_json'), 
    re_path(r'^setup_user_kaggle_api', views.setup_user_kaggle_api, name='setup_user_kaggle_api'), 
]

