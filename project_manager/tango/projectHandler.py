import os
import requests
import asyncio
import docker
import json
import textwrap

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
    print_roundtrip_text = print_roundtrip(response)
    print(print_roundtrip_text)
    print(response.content.decode('utf-8'))

    # return response.json()
    # return str(response.content.decode('utf-8'))
    return json.dumps({'response': str(response.content, 'utf-8').replace('"',''), 'request_info': str(print_roundtrip_text)})



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
    print("response")
    print(response)

    # return response.json()
    # return str(response.content.decode('utf-8'))
    return str(response.content, 'utf-8').replace('"','')


#endregion

#region Get Docker Logs ...............................................................................................

def get_docker_log_handler(container, last_logs_timestamp):
    client = docker.from_env()
    dockerContainerName = get_docker_container_name(container)
    containerList = client.containers.list()
    container = next(item for item in containerList if dockerContainerName in str(item.name))
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
        'codeGen' : "8888",
        'imageDepoly' : "8890",
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
    elif container == 'codeGen' :
        containerName = 'code_gen'
    elif container == 'imageDepoly' :
        containerName = 'cloud_deploy'
    else :
        containerName = 'bms'

    return str(containerName)
#endregion


def print_roundtrip(response, *args, **kwargs):
    format_headers = lambda d: '\n'.join(f'{k}: {v}' for k, v in d.items())
    print('')
    return str(textwrap.dedent('''
        ---------------- request ----------------
        {req.method} {req.url}
        {reqhdrs}
        ---------------- response ----------------
        {res.status_code} {res.reason} {res.url}
        {reshdrs}

        response : {res.text}
    ''').format(
        req=response.request, 
        res=response, 
        reqhdrs=format_headers(response.request.headers), 
        reshdrs=format_headers(response.headers), 
    ))



# 로그에 보여질 Container 명으로 변경
def get_log_container_name(container):
    if container == 'bms':
        return 'BMS'
    elif container == 'yoloe':
        return 'Auto NN'
    elif container == 'codeGen' or container == 'code_gen' or container == 'codegen':
        return 'Code Gen'
    

def db_container_name(container):
    if container == 'bms' or container == 'BMS':
        return 'bms'
    elif container == 'yoloe':
        return 'yoloe'
    elif container == 'codeGen' or container == 'code_gen' or container == 'codegen':
        return 'codeGen'