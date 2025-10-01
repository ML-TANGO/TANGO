import os
import random
import warnings
import multiprocessing as mp

import requests

from django.views.decorators.csrf import csrf_exempt
from rest_framework import status, viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Info, Node, Edge, Pth
from .serializers import InfoSerializer, NodeSerializer, EdgeSerializer, PthSerializer
from .tango.main.select import run_autonn
from .tango.main.visualize import export_pth, export_yml

DEBUG = False
PROCESSES = {}

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
warnings.filterwarnings(action='ignore', category=UserWarning)


@api_view(['GET', 'POST'])
def pth_list(request):
    """
    Make a PyTorch Model generated with Viz
    """
    if request.method == 'GET':
        pths = Pth.objects.all()
        serializer = PthSerializer(pths, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        info_dict = request.data
        userid = info_dict.get('userid')
        project_id = info_dict.get('project_id')

        if not userid or not project_id:
            return Response("Missing userid/project_id", status=status.HTTP_400_BAD_REQUEST)

        proj_path = f'/shared/common/{userid}/{project_id}'
        file_path = os.path.join(proj_path, 'basemodel.pth')
        yaml_path = os.path.join(proj_path, 'basemodel.yml')

        export_pth(file_path)
        model_name = info_dict.get('model_type', '') + info_dict.get('model_size', '')
        export_yml(model_name, yaml_path)

        serializer = PthSerializer(data={
            'userid': userid,
            'project_id': project_id,
            'model_pth': file_path,
            'model_yml': yaml_path,
        })

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# 돌아가는 중으로 간주할 상태들(원하는 대로 조정)
ACTIVE_STATUSES = ["started", "running", "resumed"]

@api_view(['GET'])
def active_info(request):
    # 1) 우선 '돌아가는 중' 상태 중에서 가장 최근(updated_at 우선, 그다음 id) 하나
    qs = Info.objects.filter(status__in=ACTIVE_STATUSES).order_by('-updated_at', '-id')
    info = qs.first()

    # 2) 그런 게 없으면 전체 중에서 가장 최근 하나
    if not info:
        info = Info.objects.order_by('-updated_at', '-id').first()

    if not info:
        return Response({"exists": False}, status=status.HTTP_200_OK)

    return Response({
        "exists": True,
        "userid": info.userid,
        "project_id": info.project_id,
        "status": info.status,
        "progress": info.progress,
        "model_type": info.model_type,
        "is_yolo": (info.model_type == "yolov9"),
    }, status=status.HTTP_200_OK)

@api_view(['GET'])
def start(request):
    """
    API for project manager having autonn start
    """
    params = request.query_params
    userid = params.get('user_id')
    project_id = params.get('project_id')

    synchronize_project_manager_db(userid)

    try:
        info = Info.objects.get(userid=userid, project_id=project_id)
        print(f"\n[AutoNN GET/start] Process exists with status [{info.status}]")
        info.print()

        if info.best_net and os.path.isfile(info.best_net):
            print(f"[AutoNN GET/start] Best model exists: {info.best_net}")

    except Info.DoesNotExist:
        info = Info(userid=userid, project_id=project_id)
        info.save()

    try:
        if str(info.process_id) in PROCESSES:
            print(f"[AutoNN GET/start] Terminating previous process #{info.process_id}")
            zombie_process = PROCESSES.pop(str(info.process_id))
            if zombie_process.is_alive():
                zombie_process.terminate()
                zombie_process.join()

        pr = mp.Process(target=process_autonn, args=(userid, project_id))
        pr_id = get_process_id()
        PROCESSES[pr_id] = pr
        PROCESSES[pr_id].start()

        info.process_id = pr_id
        info.status = "started"
        info.progress = "ready"
        info.save()

        return Response("started", status=status.HTTP_200_OK)
    except Exception as e:
        print(f"[AutoNN GET/start] exception: {e}")
        info.status = "failed"
        info.save()
        return Response("failed", status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
def resume(request):
    """
    API for project manager having autonn resumed
    """
    params = request.query_params
    userid = params.get('user_id')
    project_id = params.get('project_id')

    synchronize_project_manager_db(userid)

    try:
        info = Info.objects.get(userid=userid, project_id=project_id)
    except Info.DoesNotExist:
        return Response("failed", status=status.HTTP_404_NOT_FOUND)

    try:
        if info.progress != 'oom':
            return Response("failed", status=status.HTTP_200_OK)

        if str(info.process_id) in PROCESSES:
            print(f"[AutoNN GET/resume] Terminating previous process #{info.process_id}")
            zombie_process = PROCESSES.pop(str(info.process_id))
            if zombie_process.is_alive():
                zombie_process.terminate()
                zombie_process.join()

        pr = mp.Process(target=process_autonn, args=(userid, project_id))
        pr_id = get_process_id()
        PROCESSES[pr_id] = pr
        PROCESSES[pr_id].start()

        info.process_id = pr_id
        info.status = "resumed"
        info.save()

        return Response("started", status=status.HTTP_200_OK)
    except Exception as e:
        print(f"[AutoNN GET/resume] exception: {e}")
        info.status = "failed"
        info.save()
        return Response("failed", status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
def stop(request):
    """
    API for project manager having autonn clean
    """
    params = request.query_params
    userid = params.get('user_id')
    project_id = params.get('project_id')

    synchronize_project_manager_db(userid)

    try:
        info = Info.objects.get(userid=userid, project_id=project_id)
        process_done = PROCESSES.pop(str(info.process_id))
        process_done.terminate()
        print(f"[AutoNN Get/stop] TODO: Delete all temporary files")
    except Info.DoesNotExist:
        print(f"[AutoNN GET/stop] Requested process does not exist")

    return Response("success", status=status.HTTP_200_OK)


@api_view(['GET'])
def status_request(request):
    """
    API for project manager pooling autonn status
    """
    params = request.query_params
    userid = params.get('user_id')
    project_id = params.get('project_id')

    try:
        info = Info.objects.get(userid=userid, project_id=project_id)
        if DEBUG:
            print('\n' + '='*15 + ' AutoNN Heart Beat ' + '='*16)
            info.print()
    except Info.DoesNotExist:
        return Response("ready", status=status.HTTP_204_NO_CONTENT)

    try:
        if not PROCESSES[str(info.process_id)].is_alive():
            return Response(info.status, status=status.HTTP_208_ALREADY_REPORTED)
        info.status = "running"
        info.save()
        return Response("running", status=status.HTTP_200_OK)
    except Exception as e:
        print(f"[AutoNN GET/status_request] exception: {e}")
        return Response("failed", status=status.HTTP_400_BAD_REQUEST)


def status_report(userid, project_id, status="success"):
    """
    Report status to project manager when the autonn process ends
    """
    try:
        url = 'http://projectmanager:8085/status_report'
        headers = {'Content-Type': 'text/plain'}
        payload = {
            'container_id': "autonn",
            'user_id': userid,
            'project_id': project_id,
            'status': status
        }
        response = requests.get(url, headers=headers, params=payload)
        print(f"[AutoNN status_report] report = {status}, response = {response.status_code}")

        info = Info.objects.get(userid=userid, project_id=project_id)
        info.status = status
        info.process_id = ''
        info.save()
    except Exception as e:
        print(f"[AutoNN status_report] exception: {e}")


def process_autonn(userid, project_id):
    """
    1. Run autonn pipeline
    2. Export model weights and architecture
    3. Report status to project manager
    """
    info = Info.objects.get(userid=userid, project_id=project_id)
    resume = info.progress == 'oom'

    try:
        run_autonn(userid, project_id, resume=resume, viz2code=False, nas=False, hpo=False)
        info = Info.objects.get(userid=userid, project_id=project_id)
        info.progress = 'autonn ends'
        info.status = "completed"
        info.save()
        status_report(userid, project_id, "completed")

    except Exception as e:
        print(f"[AutoNN process_autonn] exception: {e}")
        import torch, gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        info = Info.objects.get(userid=userid, project_id=project_id)
        if 'CUDA' in str(e):
            print(f"[AutoNN process_autonn] CUDA OOM, retry with reduced batch size next time")
            info.progress = "oom"
        else:
            print(f"[AutoNN process_autonn] General failure")

        info.status = "failed"
        info.model_viz = "not ready"
        info.save()
        status_report(userid, project_id, "failed")


def get_process_id():
    """
    Assign a new random number to a process
    """
    while True:
        pr_num = str(random.randint(100000, 999999))
        if pr_num not in PROCESSES:
            return pr_num


def synchronize_project_manager_db(user):
    """
    Remove Info and Pth entries from DB if the corresponding shared directory doesn't exist.
    Also terminates and removes zombie processes if necessary.
    """
    infos = Info.objects.all()
    for i in infos:
        uid, pid = i.userid, i.project_id
        if uid == user:
            shared_dir = f"/shared/common/{uid}/{pid}"
            if not os.path.isdir(shared_dir):
                i.delete()
                try:
                    if PROCESSES[str(i.process_id)].is_alive():
                        process_done = PROCESSES.pop(str(i.process_id))
                        process_done.terminate()
                except:
                    pass
                try:
                    pth = Pth.objects.get(userid=uid, project_id=pid)
                    pth.delete()
                except Pth.DoesNotExist:
                    pass


class InfoView(viewsets.ModelViewSet):
    """
    API endpoint for listing and managing Info entries.
    """
    serializer_class = InfoSerializer
    queryset = Info.objects.all()


class NodeView(viewsets.ModelViewSet):
    """
    API endpoint for listing and managing Node entries.
    """
    serializer_class = NodeSerializer
    queryset = Node.objects.all()


class EdgeView(viewsets.ModelViewSet):
    """
    API endpoint for listing and managing Edge entries.
    """
    serializer_class = EdgeSerializer
    queryset = Edge.objects.all()


class PthView(viewsets.ModelViewSet):
    """
    API endpoint for listing and managing Pth (PyTorch model export) entries.
    """
    serializer_class = PthSerializer
    queryset = Pth.objects.all()

