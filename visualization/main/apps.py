"""
high level support for doing this and that.
"""


from django.apps import AppConfig


class MainConfig(AppConfig):
    # pylint: disable=too-few-public-methods
    """A dummy docstring."""
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main'
