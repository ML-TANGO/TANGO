import os
from contextlib import suppress
from typing import Dict, Iterable, Sequence, Tuple, Optional, Any
from django.db import transaction

def is_distributed() -> bool:
    return os.environ.get("WORLD_SIZE", "1") not in ("1", "", None)

def is_rank0() -> bool:
    return os.environ.get("RANK", "0") == "0"

def _ensure_django_ready() -> bool:
    # 이미 준비되어 있으면 True
    try:
        from django.apps import apps
        if apps.ready:
            return True
    except Exception:
        pass
    # 준비 안 되어 있으면 셋업 시도
    try:
        import django
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "autonn_core.settings")  # 프로젝트에 맞게
        django.setup()
        return True
    except Exception:
        return False

def _get_model(app_label: str, model_name: str):
    from django.apps import apps
    return apps.get_model(app_label, model_name)

# generic func-1: partial update for one
def safe_partial_update(
    app_label: str,
    model_name: str,
    filters: Dict,
    updates: Dict,
    *,
    rank0_only: bool = True
    ) -> bool:
    """
    지정 필드만 원자적으로 업데이트, 존재 안하면 False 반환
        - 기본: DDP(WORLD_SIZE>1)에서는 수행 안 함(rank0_only=True로 rank0에서 수행할 수는 있음)
        - AUTONN_ALLOW_ORM != '1'이면 수행 안 함
    """
    if is_distributed() and (rank0_only and not is_rank0()):
        return False
    if os.environ.get("AUTONN_ALLOW_ORM", "1") != "1":
        return False
    if not _ensure_django_ready():
        return False

    Model = _get_model(app_label, model_name)
    if Model is None:
        return False

    model_fields = {f.name for f in Model._meta.get_fields() if getattr(f, "concrete", False)}
    protected = {"id", "userid", "project_id"}
    updates = {k: v for k, v in updates.items() if k in model_fields and k not in protected}
    if not updates:
        return False

    rows = Model.objects.filter(**filters).update(**updates)
    return rows > 0

# generic func-2: update or create for one
def safe_update_or_create(
    app_label: str,
    model_name: str,
    filters: Dict,
    defaults: Dict,
    *,
    rank0_only: bool = True
) -> Tuple[bool, bool]:
    """
    update_or_create wrapper
    return (updated, created)
    """
    if is_distrubuted() and (rank0_only and not is_rank0()):
        return False
    if os.environ.get("AUTONN_ALLOW_ORM", "1") != "1":
        return False
    if not _ensure_django_ready():
        return False

    Model = _get_model(app_label, model_name)
    if Model is None:
        return False, False

    model_fields = {f.name for f in Model._meta.get_fields() if getattr(f, "concrete", False)}
    protected = {"id", "userid", "project_id"}
    defaults = {k: v for k, v in defaults.items() if k in model_fields and k not in protected}

    obj, created = Model.objects.update_or_create(defaults= defaults, **filters)
    return (not created), created

# generic func-3: upsert for many
def safe_bulk_update_or_create(
    app_label: str,
    model_name: str,
    items: Iterable[Dict],
    key_fields: Sequence[str],
    update_fields: Sequence[str],
    *,
    rank0_only: bool = True,
    batch_size: int = 1000,
) -> Tuple[int, int]:
    """
    items: 각 항목은 {...key_fields..., ...update_fields...}로 구성
    key_fields로 filter, update_fields만 부분 업데이트
    성능: 만건 이상이면 DB별 upsert를 고려하고, 보통은 수백~수천건이면 충분
    return (updated_count, created_count)
    """
    if is_distributed() and (rank0_only and not is_rank0()):
        return 0, 0
    if os.environ.get("AUTONN_ALLOW_ORM", "1") != "1":
        return 0, 0
    if not _ensure_django_ready():
        return 0, 0

    Model = _get_model(app_label, model_name)
    if Model is None:
        return 0, 0

    model_fields = {f.name for f in Model._meta.get_fields() if getattr(f, "concrete", False)}
    protected = {"id", "userid", "project_id"}
    update_fields = [f for f in update_fields if f in model_fields and f not in protected]

    updated = created = 0
    with transaction.atomic():
        for item in items:
            filters = {k: item[k] for k in key_fields}
            defaults = {k: item[k] for k in update_fields if k in item}
            obj, was_created = Model.objects.update_or_create(default=defaults, **filters)
            if was_created:
                created += 1
            else:
                updated += 1
    return updated, created

# generic func-4: get one instance
def safe_get_instance(
    app_label: str,
    model_name: str,
    filters: Dict,
    only_fields: Optional[Sequence[str]] = None,
    *,
    rank0_only: bool = True,
):
    """
    단일 인스턴스 반환(읽기용)
        - only_field로 최소 필드만 로딩
    """
    if is_distributed() and (rank0_only and not is_rank0()):
        return None
    if os.environ.get("AUTONN_ALLOW_ORM", "1") != "1":
        return None
    if not _ensure_django_ready():
        return None

    Model = _get_model(app_label, model_name)
    if Model is None:
        return None

    qs = Model.objects.filter(**filters)
    if only_fields:
        qs = qs.only(*only_fields)
    return qs.first()

# generic func-5: get partial fields
def safe_get_values(
    app_label: str,
    model_name: str,
    filters: Dict,
    fields: Sequence[str],
    *,
    rank0_only: bool = True,
) -> Optional[Dict]:
    """
    단일 레코드의 다수의 특정 필드만 dict로 가져오기
    없으면 None 반환
    """
    if is_distributed() and (rank0_only and not is_rank0()):
        return None
    if os.environ.get("AUTONN_ALLOW_ORM", "1") != "1":
        return None
    if not _ensure_django_ready():
        return None

    Model = _get_model(app_label, model_name)
    if Model is None:
        return None

    q = Model.objects.filter(**filters).values(*fields)
    return q.first() # None or dict

#generic func-6: get only one field
def safe_get_field(
    app_label: str,
    model_name: str,
    filters: Dict,
    field: str,
    *,
    rank0_only: bool = True,
) -> Optional[Any]:
    """
    단일 레코드의 단 한개의 필드 값만 스칼라로 가져오기
    없으면 None 반환
    """
    if is_distributed() and (rank0_only and not is_rank0()):
        return None
    if os.environ.get("AUTONN_ALLOW_ORM", "1") != "1":
        return None
    if not _ensure_django_ready():
        return None

    Model = _get_model(app_label, model_name)
    if Model is None:
        return None

    q = Model.objects.filter(**filters).values_list(field, flat=True)
    return q.first() # Scalar or None

# thin wrappers for 'Info'
def safe_update_info(userid: str, project_id: str, **fields) -> bool:
    return safe_partial_update(
        "autonn_core", "Info",
        filters={"userid": userid, "project_id": project_id},
        updates=fields,
    )

def safe_upsert_info(userid: str, project_id: str, **fields) -> bool:
    return safe_update_or_create(
        "autonn_core", "Info",
        filters={"userid": userid, "project_id": project_id},
        defaults=fields,
    )

def safe_get_info_values(userid: str, project_id: str, fields) -> Optional[Dict]:
    return safe_get_values(
        "autonn_core", "Info",
        {"userid": userid, "project_id": project_id},
        fields,
    )

def safe_get_info_field(userid: str, project_id: str, field: str) -> Optional[Any]:
    return safe_get_field(
        "autonn_core", "Info",
        {"userid": userid, "project_id": project_id},
        field,
    )

#def safe_update_info(
#    userid: str,
#    project_id: str,
#    *,
#    rank0_only: bool = True,
#    create_if_missing: bool = False,
#    **fields,  # status=..., progress=..., batch_size=..., epoch=..., best_acc=..., best_net=...
#) -> bool:
#    """
#    Info 레코드에서 지정된 필드만 부분 업데이트
#        - DDP(WORLD_SIZE>1)면 rank0에서만 실행(rank0_only=True)하거나 아무것도 하지 않음
#        - 환경변수 AUTONN_ALLOW_ORM != "1" 으로 설정하면 아무것도 하지 않음
#        - Info 존재하면 원자적 update, 없으면 create_if_missing=True일 때 Info 생성
#    반환값: True(뭔가 변경/생성), False(스킵 혹은 없음)
#    """
#    if is_distributed() and not (rank0_only and is_rank0()):
#        return False
#    if os.environ.get("AUTONN_ALLOW_ARM", "1") != "1":
#        return False
#    if not _ensure_django_ready():
#        return False
#
#    Info = _get_model("autonn_core", "Info")
#    if Info is None:
#        return False
#
#    # 모델 실제 필드만 필터링 (+ 보호필드 제외)
#    model_fields = {f.name for f in Info._meta.get_fields() if getattr(f, "concrete", False)}
#    protected = {"id", "userid", "project_id"}
#    updates = {k: v for k, v in fields.items() if k in model_fields and k not in protected}
#    if not updates:
#        return False
#
#    rows = Info.objects.filter(userid=userid, project_id=project_id).update(**updates)
#    if rows > 0:
#        return True
#
#    if create_if_missing:
#        payload = {"userid": userid, "project_id": project_id}
#        payload.update(updates)
#        with suppress(Exception):
#            Info.objects.create(**payload)
#            return True
#    return False

# thin wrappers for 'Node' and 'Edge'
def safe_update_node(*, order: int, **fields) -> bool:
    return safe_partial_update(
        "autonn_core", "Node",
        filters={"order": order},
        updates=fields,
    )

def safe_update_edge(*, id: int, **fields) -> bool:
    return safe_partial_update(
        "autonn_core", "Edge",
        filters={"id": id},
        updates=fields,
    )


