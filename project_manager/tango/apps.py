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

        try:
            from .models import WorkflowOrder, Project
            WorkflowOrder.objects.filter(workflow_name="codeGen").update(workflow_name="code_gen")
            Project.objects.filter(container="codeGen").update(container="code_gen")
        except WorkflowOrder.DoesNotExist:
            print("WorkflowOrder Model 존재하지 않음...")
        except Project.DoesNotExist:
            print("Project Model 존재하지 않음...")
        except Exception as error:
            print("tango app ready error", error)
            print("존재하는 WorkflowOrder.objects가 없음..")
        
        return super().ready()
