import os
import requests
import docker
# import zipfile
import shutil
import json
import textwrap
from datetime import datetime, timedelta, timezone

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
    "autonn_cl" : DockerContainerInfo("autonn_cl", 8102, "autonn-cl", "autonn_cl", "Auto NN CL"),
    "code_gen": DockerContainerInfo("code_gen", 8888, "codeGen", "code_gen", "Code Gen"),
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

# def call_container_api(host, path, user_id, project_id):
#     """
#     Request container API

#     Args:
#         host : hostname (Defined in docker-compose.yml) 
#         user_id : user_id
#         project_id : project_id

#     Returns:
#         request info
#     """
#     try:
#         url = 'http://' + host + '/' + path
#         headers = {
#             'Content-Type' : 'text/plain'
#         }
#         payload = {
#             'user_id' : user_id,
#             'project_id' : project_id,
#         }
#         response = requests.get(url, headers=headers, params=payload, timeout = 5)
        
#         if response.status_code == 404:
#             print("404 ERROR")
#             raise Exception("not found")
        
#         print_roundtrip_text = print_roundtrip(response, path, host)
#         return json.dumps({'response': str(response.content, 'utf-8').replace('"',''), 'request_info': str(print_roundtrip_text)})
#     except Exception:
#         print("CALL ERROR ---------------------------------------")
#         raise
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
        return json.dumps({'response': str(response.content, 'utf-8').replace('"',''), 
                          'request_info': str(print_roundtrip_text)})
    except requests.exceptions.ConnectionError as e:
        container_name = host.split(':')[0]
        error_msg = (f"컨테이너 '{container_name}'에 연결할 수 없습니다. "
                    f"컨테이너가 실행 중이지 않습니다.")
        print(f"[ERROR] {error_msg}")
        raise Exception(error_msg)
    except requests.exceptions.Timeout as e:
        container_name = host.split(':')[0]
        error_msg = f"컨테이너 '{container_name}' 응답 시간 초과"
        print(f"[ERROR] {error_msg}")
        raise Exception(error_msg)
    except Exception as e:
        print("CALL ERROR ---------------------------------------")
        print(f"Error: {str(e)}")
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
        docker log (string): 로그 문자열 또는 에러 메시지
    """
    try:
        client = docker.from_env()
        dockerContainerName = CONTAINER_INFO[container].docker_name
        
        def _match_container(c):
            # Prefer compose service label to avoid substring collisions
            labels = getattr(c, "labels", None) or {}
            service_name = labels.get('com.docker.compose.service')
            if service_name:
                return service_name == dockerContainerName
            name = str(c.name)
            return (
                name == dockerContainerName
                or name.endswith(f"_{dockerContainerName}_1")
                or name.endswith(f"-{dockerContainerName}-1")
            )

        # 중지된 컨테이너도 포함하여 검색, 없으면 None 반환
        container_obj = next(
            (c for c in client.containers.list(all=True) if _match_container(c)),
            None
        )
        
        if container_obj is None:
            print(f"[WARN] Container '{dockerContainerName}' not found.")
            return f"[Container Not Found] '{dockerContainerName}'", last_logs_timestamp
        
        # 컨테이너가 중지된 경우 체크
        if container_obj.status != 'running':
            print(f"[WARN] Container '{dockerContainerName}' " + 
                  f"is not running (status: {container_obj.status}).")
            return (f"[Container Not Running] '{dockerContainerName}' " +
                    f"status: {container_obj.status}"), last_logs_timestamp
        
        logs = ''
        start_ts = float(last_logs_timestamp or 0)
        if start_ts <= 0:
            logs = container_obj.logs(timestamps=True, stream=False)
        else:
            logs = container_obj.logs(timestamps=True, since=start_ts, stream=False)
            # If nothing came back, retry from the beginning once to resync
            if not logs:
                logs = container_obj.logs(timestamps=True, stream=False)
        
        if logs is None:
            return '', last_logs_timestamp

        formatted, latest_ts = format_logs_with_kst(
            logs.decode('utf-8'), dockerContainerName, last_logs_timestamp
        )
        return formatted, latest_ts
        
    except KeyError as e:
        error_msg = f"Container key not found in CONTAINER_INFO: {container}"
        print(f"[ERROR] {error_msg}")
        return f"[Config Error] {error_msg}", last_logs_timestamp
    except Exception as e:
        error_msg = f"Error getting docker logs: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return f"[Error] {error_msg}", last_logs_timestamp
#endregion


def format_logs_with_kst(raw_logs: str, container_name: str, last_ts: float):
    """Convert docker log timestamps to KST (for cursor only) and output without timestamp.
    Returns: (formatted_logs, latest_timestamp_for_cursor)
    """
    if not raw_logs:
        return "", last_ts

    lines = raw_logs.splitlines()
    formatted = []
    latest_ts = last_ts or 0
    cutoff_ts = last_ts or 0
    for line in lines:
        formatted_line, ts_epoch = _format_log_line(line, container_name)
        # Skip lines older than the stored cursor (e.g., previous project runs)
        if ts_epoch and ts_epoch < cutoff_ts:
            continue
        formatted.append(formatted_line)
        if ts_epoch:
            latest_ts = max(latest_ts, ts_epoch)
    # If no timestamp was parsed, advance using current time to avoid replaying the same chunk
    if latest_ts == (last_ts or 0):
        latest_ts = datetime.now(timezone.utc).timestamp()
    return "\n".join(formatted), latest_ts


def _format_log_line(line: str, container_name: str):
    """
    Remove the leading docker timestamp for display, keep the remainder intact.
    """
    # Preserve leading spaces/content except a trailing newline
    line_raw = line.rstrip('\n')
    if not line_raw:
        return line_raw, None
    import re

    # Match a timestamp prefix but preserve all following whitespace/content verbatim
    m = re.match(r'^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)', line_raw)
    if m:
        ts_raw = m.group("ts")
        remainder = line_raw[m.end():]  # keep leading spaces/tabs after timestamp exactly
        try:
            dt_utc = _parse_docker_ts(ts_raw)
            decorated = f"{container_name} |{remainder}" if remainder else container_name
            return decorated, dt_utc.timestamp()
        except Exception:
            # Even if parsing fails, drop the ts and keep the remainder
            now_utc = datetime.now(timezone.utc)
            decorated = f"{container_name} |{remainder}" if remainder else container_name
            return decorated, now_utc.timestamp()

    # Fallback: no timestamp detected; keep the line as-is
    now_utc = datetime.now(timezone.utc)
    decorated = f"{container_name} |{line_raw}" if line_raw else container_name
    return decorated, now_utc.timestamp()


def _parse_docker_ts(ts_raw: str) -> datetime:
    """
    Parse docker timestamps such as:
      2025-12-08T07:29:40.12345678Z
      2025-12-08T07:29:40.123Z
      2025-12-08T07:29:40Z
    Return UTC datetime.
    """
    # Separate fractional and timezone
    # Match date + time, optional fraction, optional tz
    import re

    m = re.match(
        r'^(?P<base>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})'
        r'(?P<frac>\.\d+)?'
        r'(?P<tz>Z|[+-]\d{2}:?\d{2})?$',
        ts_raw
    )
    if not m:
        raise ValueError("unrecognized timestamp")

    base = m.group("base")
    frac = m.group("frac") or ""
    tz = m.group("tz") or "+00:00"

    # Normalize fraction to microseconds (6 digits)
    if frac:
        digits = frac[1:]  # drop leading dot
        if len(digits) > 6:
            digits = digits[:6]
        digits = digits.ljust(6, "0")
        frac_norm = f".{digits}"
    else:
        frac_norm = ".000000"

    # Normalize timezone to ±HH:MM
    if tz == "Z":
        tz = "+00:00"
    elif len(tz) == 5 and tz[3] != ":":
        # e.g. +0900 -> +09:00
        tz = f"{tz[:3]}:{tz[3:]}"

    iso_str = f"{base}{frac_norm}{tz}"
    return datetime.fromisoformat(iso_str)

#region Get Container Info ............................................................................................
def get_deploy_container(deploy_type):
    """
    Get host name and port with target info for image deployment

    Args:
        deploy_type : deploy_type (Defined in shared/common/user_id/project_id/projct_info.yaml)

    Returns:
        host name, port
    """

    type_lower = deploy_type.lower()

    if type_lower == 'cloud' or deploy_type == 'kt_cloud' or deploy_type == 'GCP' or deploy_type == 'AWS':
        return "cloud_deploy"
    elif type_lower == 'k8s' or type_lower == 'k8s_jetson_nano':
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
    elif container == 'autonn':
        return 'Auto NN'
    elif container == 'code_gen':
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
