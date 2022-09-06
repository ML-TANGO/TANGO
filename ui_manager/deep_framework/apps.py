"""app module for deep_frameowrk
This module for app.
Attributes:

Todo:

"""

from django.contrib import admin

# Register your models here.rom django.apps import AppConfig


class DeepFrameworkConfig(AppConfig):
    """DeepFramworkConfig class
    Note:
    Args:
        AppConfig
    Attributes:
    """

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'deep_framework'
