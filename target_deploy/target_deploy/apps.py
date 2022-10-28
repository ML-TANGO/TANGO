"""
apps.py
"""
from django.apps import AppConfig


class TargetDeployConfig(AppConfig):
    """
    TargetDeployConfig
    """
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'target_deploy'
