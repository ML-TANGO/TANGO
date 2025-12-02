import logging
import multiprocessing as mp
import os
import random
import warnings

import requests
from django.db import transaction
from rest_framework import status, viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Edge, Info, Node, Pth
from .serializers import (
    EdgeSerializer,
    InfoSerializer,
    NodeSerializer,
    PthSerializer,
)
from .tango.main.select import run_autonn
from .tango.main.visualize import export_pth, export_yml
from tango.utils.django_utils import safe_update_info

logger = logging.getLogger(__name__)

DEBUG = False
PROCESSES = {}

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
warnings.filterwarnings(action='ignore', category=UserWarning)


@api_view(['GET', 'POST'])
def pth_list(request):
    """Make a PyTorch Model generated with Viz."""
    if request.method == 'GET':
        pths = Pth.objects.all()
        serializer = PthSerializer(pths, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        info_dict = request.data
        userid = info_dict.get('userid')
        project_id = info_dict.get('project_id')

        if not userid or not project_id:
            return Response(
                "Missing userid/project_id", status=status.HTTP_400_BAD_REQUEST
            )

        proj_path = f'/shared/common/{userid}/{project_id}'
        file_path = os.path.join(proj_path, 'basemodel.pth')
        yaml_path = os.path.join(proj_path, 'basemodel.yml')

        export_pth(file_path)
        model_name = (
            info_dict.get('model_type', '') + info_dict.get('model_size', '')
        )
        export_yml(model_name, yaml_path)

        serializer = PthSerializer(
            data={
                'userid': userid,
                'project_id': project_id,
                'model_pth': file_path,
                'model_yml': yaml_path,
            }
        )

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# States considered as "active" (adjust as needed)
ACTIVE_STATUSES = ["started", "running", "resumed", "failed"]

def get_user_and_project(request):
    if request.method == 'GET':
        params = request.query_params
    else: # POST
        params = request.data
    userid = params.get('user_id') or params.get('userid')
    project_id = params.get('project_id')
    return userid, project_id

@api_view(['GET'])
def active_info(request):
    # 1) Pick latest entry among active statuses (updated_at, then id)
    qs = Info.objects.filter(
        status__in=ACTIVE_STATUSES
    ).order_by('-updated_at', '-id')
    info = qs.first()

    # 2) Otherwise pick the latest entry overall
    if not info:
        info = Info.objects.order_by('-updated_at', '-id').first()

    if not info:
        return Response({"exists": False}, status=status.HTTP_200_OK)

    return Response(
        {
            "exists": True,
            "userid": info.userid,
            "project_id": info.project_id,
            "status": info.status,
            "progress": info.progress,
            "model_type": info.model_type,
            "is_yolo": (info.model_type == "yolov9"),
        },
        status=status.HTTP_200_OK,
    )


@api_view(['GET', 'POST'])
def start(request):
    """API for project manager having autonn start."""
    userid, project_id = get_user_and_project(request)

    synchronize_project_manager_db(userid)

    try:
        with transaction.atomic():
            info, _ = Info.objects.select_for_update().get_or_create(
                userid=userid, project_id=project_id
            )
            logger.info(
                "[AutoNN GET/start] Process exists with status [%s]",
                info.status,
            )
            info.print()

            if info.best_net and os.path.isfile(info.best_net):
                logger.info("[AutoNN GET/start] Best model exists: %s", info.best_net)
            previous_pid = info.process_id

        entry = PROCESSES.pop(str(previous_pid), None)
        if entry:
            logger.info(
                "[AutoNN %s/start] Terminating previous process #%s",
                request.method, previous_pid,
            )
            zombie_proc = entry.get("process")
            if entry.get("stop_event"):
                entry["stop_event"].set()
            if zombie_proc and zombie_proc.is_alive():
                zombie_proc.terminate()
                zombie_proc.join()

        stop_event = mp.Event()
        pr = mp.Process(target=process_autonn, args=(userid, project_id, stop_event))
        pr_id = get_process_id()
        PROCESSES[pr_id] = {"process": pr, "stop_event": stop_event}
        PROCESSES[pr_id]["process"].start()

        with transaction.atomic():
            info = Info.objects.select_for_update().get(pk=info.pk)
            info.process_id = pr_id
            info.status = "started"
            info.progress = "ready"
            info.save(update_fields=["process_id", "status", "progress"])

        return Response("started", status=status.HTTP_200_OK)
    except Exception as e:
        logger.warning("[AutoNN %s/start] exception: %s", request.method, e)
        info.status = "failed"
        info.save(update_fields=["status"])
        return Response("failed", status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'POST'])
def resume(request):
    """API for project manager having autonn resumed."""
    userid, project_id = get_user_and_project(request)

    synchronize_project_manager_db(userid)

    try:
        with transaction.atomic():
            info = Info.objects.select_for_update().get(
                userid=userid, project_id=project_id
            )
            if info.progress != 'oom':
                return Response("failed", status=status.HTTP_200_OK)
            previous_pid = info.process_id

        entry = PROCESSES.pop(str(previous_pid), None)
        if entry:
            logger.info(
                "[AutoNN %s/resume] Terminating previous process #%s",
                request.method, previous_pid
            )
            zombie_proc = entry.get("process")
            if entry.get("stop_event"):
                entry["stop_event"].set()
            if zombie_proc and zombie_proc.is_alive():
                zombie_proc.terminate()
                zombie_proc.join()

        stop_event = mp.Event()
        pr = mp.Process(target=process_autonn, args=(userid, project_id, stop_event))
        pr_id = get_process_id()
        PROCESSES[pr_id] = {"process": pr, "stop_event": stop_event}
        PROCESSES[pr_id]["process"].start()

        with transaction.atomic():
            info = Info.objects.select_for_update().get(pk=info.pk)
            info.process_id = pr_id
            info.status = "resumed"
            info.save(update_fields=["process_id", "status"])

        return Response("started", status=status.HTTP_200_OK)
    except Exception as e:
        logger.warning("[AutoNN %s/resume] exception: %s", request.method, e)
        info.status = "failed"
        info.save()
        return Response("failed", status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'POST'])
def stop(request):
    """API for project manager having autonn clean."""
    userid, project_id = get_user_and_project(request)

    synchronize_project_manager_db(userid)

    try:
        info = Info.objects.get(userid=userid, project_id=project_id)
        entry = PROCESSES.pop(str(info.process_id), None)
        safe_update_info(userid=userid, project_id=project_id, status="stopping")
        if entry:
            stop_event = entry.get("stop_event")
            proc = entry.get("process")
            if stop_event:
                stop_event.set()
            if proc:
                proc.join(timeout=30)
                if proc.is_alive():
                    logger.warning(
                        "[AutoNN %s/stop] force terminating process #%s",
                        request.method, info.process_id
                    )
                    proc.terminate()
                    proc.join()
        logger.info("[AutoNN %s/stop] Requested graceful stop", request.method)
    except Info.DoesNotExist:
        logger.warning("[AutoNN %s/stop] Requested process does not exist", request.method)

    return Response("success", status=status.HTTP_200_OK)


@api_view(['GET'])
def status_request(request):
    """API for project manager pooling autonn status."""
    params = request.query_params
    userid = params.get('user_id')
    project_id = params.get('project_id')

    try:
        info = Info.objects.get(userid=userid, project_id=project_id)
        if DEBUG:
            logger.info('=' * 15 + ' AutoNN Heart Beat ' + '=' * 16)
            info.print()
    except Info.DoesNotExist:
        return Response("ready", status=status.HTTP_204_NO_CONTENT)

    try:
        entry = PROCESSES.get(str(info.process_id))
        proc = entry.get("process") if entry else None
        if not proc or not proc.is_alive():
            return Response(
                info.status, status=status.HTTP_208_ALREADY_REPORTED
            )
        info.status = "running"
        info.save()
        return Response("running", status=status.HTTP_200_OK)
    except Exception as e:
        logger.warning("[AutoNN GET/status_request] exception: %s", e)
        return Response("failed", status=status.HTTP_400_BAD_REQUEST)


def status_report(userid, project_id, status="success"):
    """Report status to project manager when the autonn process ends."""
    try:
        url = 'http://projectmanager:8085/status_report'
        headers = {'Content-Type': 'text/plain'}
        payload = {
            'container_id': "autonn",
            'user_id': userid,
            'project_id': project_id,
            'status': status,
        }
        response = requests.get(url, headers=headers, params=payload)
        logger.info(
            "[AutoNN status_report] report=%s response=%s",
            status,
            response.status_code,
        )

        with transaction.atomic():
            info = Info.objects.select_for_update().get(
                userid=userid, project_id=project_id
            )
            info.status = status
            info.process_id = ''
            info.save(update_fields=["status", "process_id"])
    except Exception as e:
        logger.warning("[AutoNN status_report] exception: %s", e)


def process_autonn(userid, project_id, stop_event=None):
    """Run the autonn pipeline and report status."""
    info = Info.objects.get(userid=userid, project_id=project_id)
    resume = (info.progress == 'oom')

    try:
        stopped = run_autonn(
            userid,
            project_id,
            resume=resume,
            viz2code=False,
            nas=False,
            hpo=False,
            stop_event=stop_event,
        )
        info = Info.objects.get(userid=userid, project_id=project_id)
        if stopped:
            info.progress = 'stopped'
            info.status = "stopped"
            info.save(update_fields=["progress", "status"])
            status_report(userid, project_id, "stopped")
        else:
            info.progress = 'autonn ends'
            info.status = "completed"
            info.save(update_fields=["progress", "status"])
            status_report(userid, project_id, "completed")

    except Exception as e:
        logger.warning("[AutoNN process_autonn] exception: %s", e)
        logger.exception("An error occurred in run_autonn()\n")
        logger.warning('=' * 80)

        import gc
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        info = Info.objects.get(userid=userid, project_id=project_id)
        if 'CUDA out of memory' in str(e):
            logger.warning(
                "[AutoNN process_autonn] CUDA Out of Memory, "
                "retry with reduced batch size next time"
            )
            info.progress = "oom"
            info.print()
            logger.warning('=' * 80)
        else:
            if stop_event and stop_event.is_set():
                logger.warning("[AutoNN process_autonn] Stop requested, exiting gracefully")
            else:
                logger.warning("[AutoNN process_autonn] General failure")

        info.status = "stopped" if (stop_event and stop_event.is_set()) else "failed"
        info.model_viz = "not ready"
        info.save(update_fields=["status", "model_viz"])
        status_report(userid, project_id, info.status)


def get_process_id():
    """Assign a new random number to a process."""
    while True:
        pr_num = str(random.randint(100000, 999999))
        if pr_num not in PROCESSES:
            return pr_num


def synchronize_project_manager_db(user):
    """
    Remove Info and Pth entries from DB if the corresponding shared directory
    doesn't exist. Also terminates and removes zombie processes if necessary.
    """
    infos = Info.objects.all()
    for i in infos:
        uid, pid = i.userid, i.project_id
        if uid == user:
            shared_dir = f"/shared/common/{uid}/{pid}"
            if not os.path.isdir(shared_dir):
                i.delete()
                try:
                    entry = PROCESSES.get(str(i.process_id))
                    proc = entry.get("process") if entry else None
                    if proc and proc.is_alive():
                        if entry.get("stop_event"):
                            entry["stop_event"].set()
                        process_done = PROCESSES.pop(str(i.process_id))
                        proc = process_done.get("process")
                        if proc:
                            proc.terminate()
                except Exception:
                    pass
                try:
                    pth = Pth.objects.get(userid=uid, project_id=pid)
                    pth.delete()
                except Pth.DoesNotExist:
                    pass


class InfoView(viewsets.ModelViewSet):
    """API endpoint for listing and managing Info entries."""
    serializer_class = InfoSerializer
    queryset = Info.objects.all()


class NodeView(viewsets.ModelViewSet):
    """API endpoint for listing and managing Node entries."""
    serializer_class = NodeSerializer
    queryset = Node.objects.all()


class EdgeView(viewsets.ModelViewSet):
    """API endpoint for listing and managing Edge entries."""
    serializer_class = EdgeSerializer
    queryset = Edge.objects.all()


class PthView(viewsets.ModelViewSet):
    """API endpoint for managing Pth (PyTorch model export) entries."""
    serializer_class = PthSerializer
    queryset = Pth.objects.all()
