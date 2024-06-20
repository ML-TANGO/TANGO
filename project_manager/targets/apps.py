import sys
from django.apps import AppConfig


class TargetsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'targets'

    def ready(self) -> None:
        print(sys.argv)
        if ("runserver" in sys.argv) or ("uwsgi" in sys.argv):
            from .load_targets import load_targets
            load_targets()
        return super().ready()
