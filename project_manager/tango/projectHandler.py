import os
import requests
import docker
# import zipfile
import shutil
import json
import textwrap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(BASE_DIR))

class DockerContainerInfo:
    def __init__(self, _key, _port, _hostname, _docker_name, _display_name):
            self.key = _key
            self.port = _port
            self.hostname = _hostname
            self.docker_name = _docker_name
            self.display_name = _display_name

    def getURL(self):
        return f"{self.hostname}:{self.port}"

CONTAINER_INFO = {
    "autonn" : DockerContainerInfo("autonn", 8100, "autonn", "autonn", "Auto NN"),
    "codeGen": DockerContainerInfo("codeGen", 8888, "codeGen", "code_gen", "Code Gen"),
    "code_gen": DockerContainerInfo("codeGen", 8888, "codeGen", "code_gen", "Code Gen"),
    "cloud_deploy": DockerContainerInfo("imagedeploy", 8890, "cloud-deploy", "cloud_deploy", "Image Deploy"),
    "kube_deploy": DockerContainerInfo("imagedeploy", 8901, "kube-deploy", "kube_deploy", "Image Deploy"),
    "ondevice_deploy": DockerContainerInfo("imagedeploy", 8891, "ondevice", "ondevice_deploy", "Image Deploy"),
}


#region API REQUEST ...................................................................................................
def call_api_handler(container, path, user_id, project_id, target_info):
    """
    Container task Status Request Handler

    Args:
        container : container
        user_id : user_id
        project_id : project_id
        target_info : If the container is imagedenploy, receive hostname and port as targetinfo

    Returns:
        request info
    """

    try:
        target = None
        if container != 'imagedeploy':
            target = CONTAINER_INFO[container]
        else :
            target = CONTAINER_INFO[get_deploy_container(target_info)]

        result = call_container_api(target.getURL(), path, user_id, project_id)
        return result
    
    except Exception as error:
        print('request_handler - error : ' + str(error))
        # return None
        raise

def call_container_api(host, path, user_id, project_id):
    """
    Request container API

    Args:
        host : hostname (Defined in docker-compose.yml) 
        user_id : user_id
        project_id : project_id

    Returns:
        request info
    """
    try:
        url = 'http://' + host + '/' + path
        headers = {
            'Content-Type' : 'text/plain'
        }
        payload = {
            'user_id' : user_id,
            'project_id' : project_id,
        }
        response = requests.get(url, headers=headers, params=payload, timeout = 5)
        
        if response.status_code == 404:
            print("404 ERROR")
            raise Exception("not found")
        
        print_roundtrip_text = print_roundtrip(response, path, host)
        return json.dumps({'response': str(response.content, 'utf-8').replace('"',''), 'request_info': str(print_roundtrip_text)})
    except Exception:
        print("CALL ERROR ---------------------------------------")
        raise

#endregion

#region Get Docker Logs ...............................................................................................

def get_docker_log_handler(container, last_logs_timestamp):
    """
    Docker-compose log Get function

    Args:
        container : container
        last_logs_timestamp(int) : Finally, the time stamp that received the docker log

    Returns:
        docker log
    """

    client = docker.from_env()
    dockerContainerName = CONTAINER_INFO[container].docker_name
    containerList = client.containers.list()
    container = next(item for item in containerList if dockerContainerName in str(item.name))
    logs = ''
    if int(last_logs_timestamp) == 0:
        logs = container.logs(timestamps = True)
    else:
        logs = container.logs(timestamps = True, since = int(last_logs_timestamp))
    if logs == None:
        return ''
    return logs.decode('utf-8')
#endregion

#region Get Container Info ............................................................................................
def get_deploy_container(deploy_type):
    """
    Get host name and port with target info for image deployment

    Args:
        deploy_type : deploy_type (Defined in shared/common/user_id/project_id/projct_info.yaml)

    Returns:
        host name, port
    """

    if deploy_type == 'Cloud' :
        return "cloud_deploy"
    elif deploy_type == 'K8S' or deploy_type == 'K8S_Jetson_Nano':
        return "kube_deploy"
    else:
        # ondevice 등등.... 
        return "ondevice_deploy"
#endregion


def print_roundtrip(response, path, container):
    """
    Transform function to display API call result as log

    Args:
        response : response
        path : response
        container : response

    Returns:
        display log
    """

    format_headers = lambda d: '\n'.join(f'{k}: {v}' for k, v in d.items())

    return str(textwrap.dedent('''
        ---------------- {path} ----------------
        Project Manager --> {container}
        {req.method} {req.url}
        {reqhdrs}
        ---------------- response ----------------
        {res.status_code} {res.reason} {res.url}
        {reshdrs}

        response : {res.text}
        ------------------------------------------
    ''').format(
        req=response.request, 
        res=response, 
        reqhdrs=format_headers(response.request.headers), 
        reshdrs=format_headers(response.headers), 
        path=path,
        container=container
    ))

# 로그에 보여질 Container 명으로 변경
def get_log_container_name(container):
    """
    Name of the container to be displayed in the log
    (Actions to unify as one when displayed in the log)

    Args:
        container : container

    Returns:
        string
    """

    if container == 'bms':
        return 'BMS'
    elif container == 'yoloe' or container == 'autonn-resnet' or container == 'autonn':
        return 'Auto NN'
    elif container == 'codeGen' or container == 'code_gen' or container == 'codegen':
        return 'Code Gen'
    elif container == 'imagedeploy':
        return 'Image Deploy'
    elif container == 'viz2code':
        return 'viz2code'

def nn_model_zip(user_id, project_id):
    """
    Convert path to zip file to shared/common/user_id/project_id/nn_model

    Args:
        user_id : user_id
        project_id : project_id

    Returns:
        zip file
    """

    file_path = os.path.join(root_path, "shared/common/{0}/{1}".format(str(user_id), str(project_id)))
    os.chdir(file_path)

    print("start folder_zip shutil")
    print("file_path  : " + str(file_path))
    print(os.walk(os.path.join(file_path, 'nn_model')))

    a = shutil.make_archive("nn_model", 'zip', os.path.join(file_path, "nn_model"))
    print(a)

    print("end folder_zip")


    return os.path.join(file_path, "nn_model.zip")

def nn_model_unzip(user_id, project_id, file):
    """
    unzip file

    Args:
        user_id : user_id
        project_id : project_id
        file : zip file

    Returns:
        zip file
    """
    file_path = os.path.join(root_path, "shared/common/{0}/{1}".format(str(user_id), str(project_id)))
    save_path = os.path.join(file_path, 'nn_model.zip')

    with open(save_path, 'wb') as destination_file:
        for chunk in file.chunks():
            destination_file.write(chunk)

    filename = save_path
    extrack_dir = os.path.join(file_path, 'nn_model')
    archive_format = "zip"

    shutil.unpack_archive(filename, extrack_dir, archive_format)

    return os.path.join(file_path, "nn_model.zip")

def create_text_file_if_not_exists(file_path):
    """
    create text file if not exists

    Args:
        file_path : file_path        
    """
    try:
        # Try to open the file in read mode to check if it exists
        with open(file_path, 'r'):
            pass
    except FileNotFoundError:
        # If the file doesn't exist, create it
        with open(file_path, 'w') as f:
            f.write("This is a new text file.")
        print(f"File '{file_path}' created.")

def update_project_log_file(user_id, project_id, log):
    """
    Add/save log to shared/common/user_id/project_id/log.txt file

    Args:
        user_id : user_id        
        project_id : project_id        
        log : Log to update
    """
        
    log_file_path = os.path.join(root_path, "shared/common/{0}/{1}".format(str(user_id), str(project_id)), 'log.txt')
    # create_text_file_if_not_exists(log_file_path)

    with open(log_file_path, 'a+') as f:
        f.write(log)


def findIndexByDictList(list, find_column, find_value):
    """
    Finds the index of a dictionary in a list of dictionaries, where the value of the specified column matches the specified value.

    Args:
        list: A list of dictionaries.
        find_column: The column name to search for the match.
        find_value: The value to match against.

    Returns:
        The index of the matching dictionary, or `None` if no match is found.
    """

    index = None
    for i, person in enumerate(list):
        print(person[find_column])
        print(find_value)
        if person[find_column] == find_value:
            index = i
            break
    
    return index

# status_report를 log 형식에 맞게 변환하는 작업
def status_report_to_log_text(request, project_info):
    user_id = request.GET['user_id']
    project_id = request.GET['project_id']
    container_id = request.GET['container_id']
    container_info = CONTAINER_INFO[container_id]
    result = request.GET['status']

    headers = ''
    for header, value in request.META.items():
        if not header.startswith('HTTP'):
            continue
        header = '-'.join([h.capitalize() for h in header[5:].lower().split('_')])
        headers += '{}: {}\n'.format(header, value)
    log_str =  str(project_info.current_log) 
    log_str += '---------------- Status Report ----------------'
    log_str += "\n" + container_info.display_name + " --> Project Manager"
    log_str += "\n" + str(request)
    log_str += "\n" + "method : " + request.method
    log_str += "\n" + headers
    log_str += '---------------- Params ----------------'
    log_str += '\nuser_id : '+ str(user_id)
    log_str += '\nproject_id : '+ str(project_id)
    log_str += '\ncontainer_id : '+ str(container_id)
    log_str += '\nstatus : '+ str(result)
    log_str += '\n----------------------------------------'
    log_str += '\n\n'


