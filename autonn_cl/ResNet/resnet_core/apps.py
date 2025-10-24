"""autonn/Resnet/resnet_core/apps.py
This file contains the configuration for the ResNet app.
"""
from django.apps import AppConfig


class ResNetCoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'resnet_core'
