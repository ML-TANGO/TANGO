"""app module for tango
This module for app.
Attributes:

Todo:

"""
from django.apps import AppConfig


class TangoConfig(AppConfig):
    """TangoConfig class
    Note:
    Args:
        AppConfig
    Attributes:
    """

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'tango'
