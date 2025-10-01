from ..models import Project
from ..enums import TaskType

def get_hyperparameter_file_name(project_id):
    try:
        project = Project.objects.get(id = project_id)  # Project id로 검색

        if project.task_type == TaskType.DETECTION:
            return "hyp.scratch.p5.yaml"
        elif project.task_type == TaskType.SEGMENTATION:
            return "hyp.scratch.seg.yaml"   
        elif project.task_type == TaskType.CLASSIFICATION:
            return "hyp.scratch.cls.yaml"
        
        return None
    except Exception as e:
        return None



def get_arguments_file_name(project_id):
    try:
        project = Project.objects.get(id = project_id)  # Project id로 검색

        if project.task_type == TaskType.DETECTION:
            return "args-detection.yaml"
        elif project.task_type == TaskType.SEGMENTATION:
            return "args-segmentation.yaml"
        elif project.task_type == TaskType.CLASSIFICATION:
            return "args-classification.yaml"
        
        return None
    except Exception as e:
        return None