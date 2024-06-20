from django.apps import AppConfig
import sys


class DatasetsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'datasets'

    def ready(self) -> None:
        print(sys.argv)
        if ("runserver" in sys.argv) or ("uwsgi" in sys.argv):
            from .views import dataset_start_scirpt

            dataset_start_scirpt()
        return super().ready()