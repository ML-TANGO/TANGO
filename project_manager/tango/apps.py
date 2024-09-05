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

    def ready(self) -> None:
        from .models import WorkflowOrder, Project
        WorkflowOrder.objects.filter(workflow_name="codeGen").update(workflow_name="code_gen")
        Project.objects.filter(container="codeGen").update(container="code_gen")
        
        return super().ready()
