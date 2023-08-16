import os
import requests
import asyncio
import docker

#region API REQUEST ...................................................................................................

async def start_handler(continer, user_id, project_id):
    print(continer)
    print(user_id)
    print(project_id)

    host, port = get_container_info(continer)
    start_task = asyncio.create_task(continer_start_api(host + ':' + port, user_id, project_id))
    result = await start_task
    # host, port = get_container_info(continer)
    # result = continer_start_api(host + ':' + port, user_id, project_id)
    return result

async def continer_start_api(host, user_id, project_id):
    url = 'http://' + host + '/start'
    headers = {
        'Content-Type' : 'text/plain'
    }
    payload = {
        'user_id' : user_id,
        'project_id' : project_id,
    }
    response = requests.get(url, headers=headers, params=payload)

    return response.json()


#################################################################################################################

async def request_handler(continer, user_id, project_id):
    print(continer)
    print(user_id)
    print(project_id)

    # host, port = get_container_info(continer)
    # start_task = asyncio.create_task(continer_request_api(host + ':' + port, user_id, project_id))
    # result = await start_task
    try:
        # host, port = get_container_info(continer)
        # result = continer_request_api(host + ':' + port, user_id, project_id)
        # return result


        host, port = get_container_info(continer)
        start_task = asyncio.create_task(continer_request_api(host + ':' + port, user_id, project_id))
        result = await start_task
        return result
    
    except Exception as error:
        print('request_handler - error : ' + str(error))
        return None

async def continer_request_api(host, user_id, project_id):
    url = 'http://' + host + '/status_request'
    headers = {
        'Content-Type' : 'text/plain'
    }
    payload = {
        'user_id' : user_id,
        'project_id' : project_id,
    }
    response = requests.get(url, headers=headers, params=payload)

    return response.json()


#endregion

#region Get Docker Logs ...............................................................................................

def get_docker_log_handler(container, last_logs_timestamp):
    client = docker.from_env()
    containerList = client.containers.list()
    container = next(item for item in containerList if container in str(item.name))
    logs = ''
    if int(last_logs_timestamp) == 0:
        logs = container.logs(timestamps = True)
    else:
        logs = container.logs(timestamps = True, since = int(last_logs_timestamp))
    return logs.decode('utf-8')
#endregion

#region Get Container Info ............................................................................................
def get_container_info(host_name):
    ports_by_container = {
        'bms' : "8081",
        'yoloe' : "8090",
    }
    return host_name, ports_by_container[host_name]

def get_docker_container_name(container):

    containerName = ''
    if container == 'init' :
        containerName = ''
    elif container == 'bms' :
        containerName = 'bms'
    elif container == 'yoloe' :
        containerName = 'autonn_yoloe'
    elif container == 'labelling' :
        containerName = 'labelling'
    elif container == 'autonn_bb' :
        containerName = 'autonn_bb'
    elif container == 'autonn_nk' :
        containerName = 'autonn_nk'
    else :
        containerName = 'bms'

    return str(containerName)
#endregion