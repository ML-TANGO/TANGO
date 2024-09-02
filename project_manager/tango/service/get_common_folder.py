import os
import glob

from ..models import Project

def get_project_name(project_id):
    project_info = Project.objects.get(id = project_id)
    return project_info.project_name

def get_folder_structure(folder_path):
    try:
        structure = []
    
        project_id_to_name = {}

        for path, dirs, files in os.walk(folder_path):
            path_split = path.split("/")
            # path_split = ["", "shared", "common", "user_id", "project_id", "path1..", "path2..."]

            # project_id로 되어있는 폴더 명을 project_name으로 변경
            if len(path_split) == 4:
                for dir in dirs:
                    project_name = get_project_name(dir)

                    if path_split[-1] not in project_id_to_name:
                        project_id_to_name[path_split[-1]] = {}
                    project_id_to_name[path_split[-1]][dir] = project_name

                    dir = project_name
                
                dirs = list(project_id_to_name[path_split[-1]].values())

            # 각 하위 Path의 project_id로 되어있는 경로를 project_name으로 변경
            if len(path_split) > 4 :
                if path_split[4] in project_id_to_name[path_split[3]]:
                    path_split[4] = project_id_to_name[path_split[3]][path_split[4]]
    
            join_path = "/".join(path_split)
            structure.append({ "dirs": dirs, "path": join_path, "org_path": path, "files": files, })
    
        return structure
    except Exception as error:
        print("error", error)
        return []
    