import os
import json, yaml
import multiprocessing as mp
import random
import warnings

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework import viewsets, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
import requests

from .models import Info, Node, Edge, Pth
from .serializers import InfoSerializer
from .serializers import NodeSerializer
from .serializers import EdgeSerializer
from .serializers import PthSerializer

from .tango.main.select import run_autonn
from .tango.main.visualize import export_pth, export_yml
from .tango.main.export import export_weight, export_config

PROCESSES = {}

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
warnings.filterwarnings(action='ignore', category=UserWarning)

# @api_view(['GET', 'POST'])
# @csrf_exempt
# def InfoList(request):
#     '''
#     List all autonn process information,
#     or create info for a new autonn process.
#     '''
#     if request.method == 'GET':
#         infoList = Info.objects.all()
#         return Response(infoList, status=status.HTTP_200_OK)

#     elif request.method == 'POST':
#         # Fetching the form data
#         usrId = request.data['user_id']
#         prjId = request.data['project_id']
#         prcId = request.data['process_id']
#         target = request.data['target']
#         # uploadedFile = request.FILES["data_yaml"]
#         task = request.data['task']
#         sts = request.data['status']

#         # Saving the information in the database
#         newInfo = Info(
#             userid=usrId,
#             project_id=prjId,
#             process_id=prcId,
#             target_device=target,
#             # data_yaml=uploadedFile,
#             task=task,
#             status=sts,
#         )
#         newInfo.save()

#         return Response("created", status=status.HTTP_201_CREATED)


# @api_view(['GET', 'POST'])
# @csrf_exempt
# def node_list(request):
#     '''
#     List all nodes, or create a new node.
#     '''
#     # print('node')
#     if request.method == 'GET':
#         # print('get')
#         nodes = Node.objects.all()
#         serializer = NodeSerializer(nodes, many=True)
#         return Response(serializer.data)

#     elif request.method == "POST":
#         # print('post')
#         serializer = NodeSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#     # nodes = Node.objects.all()
#     # serializer = NodeSerializer(nodes, many=True)
#     # return Response(serializer.data)


# @api_view(['GET', 'POST'])
# @csrf_exempt
# def edge_list(request):
#     '''
#     List all edges, or create a new edge.
#     '''
#     # print('edge')
#     if request.method == 'GET':
#         # print('get')
#         edges = Edge.objects.all()
#         serializer = EdgeSerializer(edges, many=True)
#         return Response(serializer.data)

#     elif request.method == 'POST':
#         serializer = EdgeSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#     # edges = Edge.objects.all()
#     # serializer = EdgeSerializer(edges, many=True)
#     # return Response(serializer.data)


@api_view(['GET', 'POST'])
@csrf_exempt
def pth_list(request):
    '''
    Make a PyTorch Model generated with Viz
    '''
    if request.method == 'GET':
        pth = Pth.objects.all()
        serializer = PathSerializer(pth, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        # id -------------------------------------------------------------------
        # infos = Info.objects.all()
        # info = infos[infos.count()-1] # tenace note: it is rational, assuming only one and the latest project is running
        info_dict = request.data
        userid = info_dict['userid']
        project_id = info_dict['project_id']
        PROJ_PATH = '/shared/common/'+str(userid)+'/'+str(project_id)

        # basemodel.pt ---------------------------------------------------------
        file_path = PROJ_PATH+'/basemodel.pth'
        created_model = export_pth(file_path)
        # file_path = (os.getcwd() + '/model_' + name + '.pt').replace("\\", '/')
        # torch.save(created_model, file_path)

        # basemodel.yaml -------------------------------------------------------
        yaml_path = PROJ_PATH+'/basemodel.yml'
        model_name = info_dict['model_type'] + info_dict['model_size']
        created_yaml = export_yml(model_name, yaml_path)
        # with open(yaml_path, 'w') as f:
        #             yaml.dump(created_yaml, f)

        # DB create ------------------------------------------------------------
        serializer = PthSerializer(data={'userid': userid,
                                         'project_id': project_id,
                                         'model_pth': file_path,
                                         'model_yml': yaml_path})
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
def start(request):
    """
    API for project manager having autonn start
    """
    # print("_________GET /start_____________")
    params = request.query_params
    userid = params['user_id']
    project_id = params['project_id']

    synchronize_project_manager_db(userid)

    try:
        info = Info.objects.get(userid=userid, project_id=project_id)
        if info.status in ['started', 'running']:
            # duplicate project
            return Response("failed", status=status.HTTP_406_NOT_ACCEPTABLE, content_type="text/plain")
    except Info.DoesNotExist:
        # new project
        info = Info(userid=userid, project_id=project_id)
        info.save()

    try:
        # data_yaml, proj_yaml = get_user_requirements(userid, project_id)

        pr = mp.Process(target = process_autonn, args=(userid, project_id))
        pr_id = get_process_id()
        PROCESSES[pr_id] = pr
        PROCESSES[pr_id].start()

        # info.target_device=str(proj_yaml)
        # info.data_yaml=str(data_yaml)
        info.status="started"
        info.progress="setting"
        info.process_id = pr_id
        info.save()
        return Response("started", status=status.HTTP_200_OK, content_type="text/plain")
    except Exception as e:
        print(f"[AutoNN GET/start] exception: {e}")
        info.status="failed"
        info.save()
        return Response("failed", status=status.HTTP_400_BAD_REQUEST, content_type="text/plain")


@api_view(['GET'])
def status_request(request):
    """
    API for project manager pooling autonn status
    """
    # print("_________GET /status_request_____________")
    params = request.query_params
    userid = params['user_id']
    project_id = params['project_id']

    try:
        info = Info.objects.get(userid=userid, project_id=project_id)
    except Info.DoesNotExist:
        # empty project
        return Response("ready", status=status.HTTP_204_NO_CONTENT, content_type='text/plain')

    # already done for any reason
    if info.status == "completed":
        return Response("completed", status=status.HTTP_208_ALREADY_REPORTED, content_type='text/plain')
    elif info.status == "failed":
        return Response("failed", status=status.HTTP_410_GONE, content_type='text/plain')

    try:
        if PROCESSES[str(info.process_id)].is_alive():
            # the project is running
            info.status = "running"
            info.save()
            return Response("running", status=status.HTTP_200_OK, content_type='text/plain')
        else:
            # the project is done
            if info.status == "completed":
                return Response("completed", status=status.HTTP_208_ALREADY_REPORTED, content_type='text/plain')
            elif info.status == "failed":
                return Response("failed", status=status.HTTP_410_GONE, content_type='text/plain')
    except KeyError as e:
        print(f"[AutoNN GET/status_request] exception: {e}")
        info.status = "failed"
        info.save()
        return Response("failed", status=status.HTTP_400_BAD_REQUEST, content_type='text/plain')


def status_report(userid, project_id, status="success"):
    """
    Report status to project manager when the autonn process ends
    """
    try:
        url = 'http://projectmanager:8085/status_report'
        headers = {
            'Content-Type' : 'text/plain'
        }
        payload = {
            'container_id' : "autonn",
            'user_id' : userid,
            'project_id' : project_id,
            'status' : status
        }
        response = requests.get(url, headers=headers, params=payload)

        info = Info.objects.get(userid=userid, project_id=project_id)
        info.status = status
        # process_done = PROCESSES.pop(str(info.process_id))
        # process_done.close()
        # info.process_id = ''
        info.save()
    except Exception as e:
        print(f"[AutoNN status_report] exception: {e}")


def process_autonn(userid, project_id):
    '''
    1. Run autonn (setup w/bms - pre-train - nas - hpo - train - test)
    2. Export model weights and architecure (pt, yaml)
    3. Report status to PM (completed / failed)
    '''
    try:
        # ------- actual process --------
        final_model = run_autonn(userid, project_id, viz2code="False", nas="False", hpo="False")

        info = Info.objects.get(userid=userid, project_id=project_id)
        target_acc = info.device
        convert =   [   'torchscript',  # convert to traced model(torchscript)
                        # 'onnx',         # convert to onnx
                        'onnx_end2end', # convert to onnx with nms (only for detection)
                        # 'engine',       # convert to tensor RT (need onnx model first)
                        # 'openvino',     # convert to openvino
                        # 'coreml',       # convert to coreml(for ios)
                        # 'saved_model',  # convert to tensorflow saved model
                        # 'pb',           # convert to tensorflow graph (need keras model first)
                        # 'tflite',       # convert to tensorflow lite (need keras model first)
                        # 'edgetpu',      # convert to tensorflow tpu
                        # 'tfjs',         # convert to tensorflow javascript
                    ]

        # remove duplicates
        convert = list(set(convert))

        export_weight(final_model, userid, project_id, target_acc, convert)
        #export_config(userid, project_id)

        status_report(userid, project_id, "completed")
        print("=== wait for 10 sec to avoid thread exception =============")
        import time
        time.sleep(10)
        return
    except Exception as e:
        print(f"[AutoNN process_autonn] exception: {e}")
        status_report(userid, project_id, "failed")


def get_process_id():
    """
    Assign a new random number into a process
    """
    while True:
        pr_num = str(random.randint(100000, 999999))
        try:
            temp = PROCESSES[pr_num]
        except KeyError:
            break
    return pr_num


def synchronize_project_manager_db(user):
    infos = Info.objects.all()

    for i in infos:
        uid, pid = i.userid, i.project_id
        if uid == user:
            shared_dir = f"/shared/common/{uid}/{pid}"
            # print(f"{i}: {patshared_dir}")
            if not os.path.isdir(shared_dir):
                # print("it will be deleted!")
                i.delete()
                try:
                    if PROCESSES[str(i.process_id)].is_alive():
                        # print(f"process-{i.process_id} will close!")
                        process_done = PROCESSES.pop(str(info.process_id))
                        process_done.close()
                except:
                    pass
                try:
                    pth = Pth.objects.get(userid=uid, project_id=pid)
                    pth.delete()
                except Pth.DoesNotExist:
                    pass


class InfoView(viewsets.ModelViewSet):
    '''
    Info View
    '''
    serializer_class = InfoSerializer
    queryset = Info.objects.all()


class NodeView(viewsets.ModelViewSet):
    '''
    Node View
    '''
    serializer_class = NodeSerializer
    queryset = Node.objects.all()

    def print_serializer(self):
        '''
        print serializer class
        '''
        print("Node serializer")

    def print_objects(self):
        '''
        print objects
        '''
        print("Node objects")


class EdgeView(viewsets.ModelViewSet):
    '''
    Edge View
    '''
    serializer_class = EdgeSerializer
    queryset = Edge.objects.all()

    def print_serializer(self):
        '''
        print serializer class
        '''
        print("Edge serializer")

    def print_objects(self):
        '''
        print objects
        '''
        print("Edge objects")


class PthView(viewsets.ModelViewSet):
    # pylint: disable=too-many-ancestors
    '''
    Pth View
    '''
    serializer_class = PthSerializer
    queryset = Pth.objects.all()

    def print_serializer(self):
        '''
        print serializer class
        '''
        print("Pth serializer")

    def print_objects(self):
        '''
        print objects
        '''
        print("Pth objects")
