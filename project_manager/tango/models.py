"""model module for tango
This module for model.
Attributes:

Todo:
"""

import jwt
from datetime import datetime, timedelta

import oauth2_provider.oauth2_backends
from django.conf import settings

from django.db import models, migrations
from django.contrib.auth.models import User

from django.db import models
from django.contrib.postgres.fields import ArrayField  


class AuthUser(models.Model):
    """AuthUser class
    Note:
    Args:
        models.Model
    Attributes:
    """
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.BooleanField()
    username = models.CharField(unique=True, max_length=150)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.BooleanField()
    is_active = models.BooleanField()
    date_joined = models.DateTimeField()

    class Meta:
        """AuthUser Meta class
        Note:
        Args:
          None
        Attributes:
        """
        managed = False
        db_table = 'auth_user'

class UserSetting(models.Model):
    """UserSetting class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)
    user = models.CharField(blank=True, null=True, max_length=50)                       # 생성자

    project_update_cycle = models.IntegerField(default = 10)
    autton_update_cycle = models.IntegerField(default = 1)

    class Meta:
        """UserSetting Meta class
        Note:
        Args:
          None
        Attributes:
        """
        # managed = False

        # DB 테이블 이름
        db_table = 'user_setting'

class Project(models.Model):
    """Project class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)                                                    # 프로젝트 아이디
    project_name = models.CharField(blank=True, null=True, max_length=30)                      # 프로젝트 이름
    project_description = models.CharField(blank=True, null=True, max_length=200)              # 프로젝트 설명
    create_user = models.CharField(blank=True, null=True, max_length=50)                       # 생성자
    create_date = models.CharField(blank=True, null=True, max_length=50)                       # 생성 날짜
    project_type = models.CharField(blank=True, null=True, max_length=50)          # 프로젝트 워크플로우 진행 방식 - auto, manual

    # target = models.CharField(blank=True, null=True, max_length=50, default='')
    target = models.ForeignKey("targets.Target", related_name="target", on_delete=models.PROTECT, db_column="target", blank=True, null=True)
    dataset = models.CharField(blank=True, null=True, max_length=50, default='')
    version = models.IntegerField(blank=False, null=False, default=0)

    task_type = models.CharField(blank=True, null=True, max_length=50, default='')
    autonn_dataset_file = models.CharField(blank=True, null=True, max_length=50, default='')
    autonn_basemodel = models.CharField(blank=True, null=True, max_length=50, default='')
    nas_type = models.CharField(blank=True, null=True, max_length=50, default='')
    deploy_weight_level = models.CharField(blank=True, null=True, max_length=50, default='')
    deploy_precision_level = models.CharField(blank=True, null=True, max_length=50, default='')
    deploy_processing_lib = models.CharField(blank=True, null=True, max_length=50, default='')
    deploy_user_edit = models.CharField(blank=True, null=True, max_length=50, default='')
    deploy_input_method = models.CharField(blank=True, null=True, max_length=50, default='')
    deploy_input_data_path = models.CharField(blank=True, null=True, max_length=50, default='')
    deploy_output_method = models.CharField(blank=True, null=True, max_length=50, default='')

    deploy_input_source = models.CharField(blank=True, null=True, max_length=50, default='') # 20230710

    container = models.CharField(blank=True, null=True, max_length=50, default='')               # 신경망 생성 단계
    container_status = models.CharField(blank=True, null=True, max_length=50, default='')        # 신경망 생성 상태


    last_logs_timestamp = models.FloatField(blank=True, null=True, default=0)               # 마지막 로그 출력 시간
    last_log_container = models.CharField(blank=True, null=True, max_length=50, default='') # 마지막 로그를 불러온 컨테이너
    current_log = models.TextField(blank=True, null=True, default='')                       # last_logs_timestamp 이후로 찍힌 로그

    autonn_retry_count = models.IntegerField(blank=True, null=True, default=0)

    class Meta:
        """Project Meta class
        Note:
        Args:
          None
        Attributes:
        """
        # managed = False

        # DB 테이블 이름
        db_table = 'project'


# Oauth 등록 애플리케이션 정보 테이블
class Oauth2ProviderApplication(models.Model):
    """Oauth2ProviderApplication class
    Note:
    Args:
        models.Model
    Attributes:
    """

    id = models.BigAutoField(primary_key=True)
    client_id = models.CharField(unique=True, max_length=100)
    redirect_uris = models.TextField()
    client_type = models.CharField(max_length=32)
    authorization_grant_type = models.CharField(max_length=32)
    client_secret = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    user = models.ForeignKey(User, models.DO_NOTHING, blank=True, null=True)
    skip_authorization = models.BooleanField()
    created = models.DateTimeField()
    updated = models.DateTimeField()
    algorithm = models.CharField(max_length=5)

    class Meta:
        """Oauth2ProviderApplication Meta class
        Note:
        Args:
          None
        Attributes:
        """
        managed = False
        db_table = 'oauth2_provider_application'


class Oauth2ProviderAccesstoken(models.Model):
    """Oauth2ProviderAccesstoken class
    Note:
    Args:
        models.Model
    Attributes:
    """

    id = models.BigAutoField(primary_key=True)
    token = models.CharField(unique=True, max_length=255)
    expires = models.DateTimeField()
    scope = models.TextField()
    application = models.ForeignKey('Oauth2ProviderApplication', models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(User, models.DO_NOTHING, blank=True, null=True)
    created = models.DateTimeField()
    updated = models.DateTimeField()
    source_refresh_token = models.OneToOneField('Oauth2ProviderRefreshtoken', models.DO_NOTHING, blank=True, null=True)
    id_token = models.OneToOneField('Oauth2ProviderIdtoken', models.DO_NOTHING, blank=True, null=True)

    class Meta:
        """Oauth2ProviderAccesstoken Meta class
        Note:
        Args:
          None
        Attributes:
        """

        managed = False
        db_table = 'oauth2_provider_accesstoken'


class Oauth2ProviderIdtoken(models.Model):
    """Oauth2ProviderIdtoken class
    Note:
    Args:
        models.Model
    Attributes:
    """

    id = models.BigAutoField(primary_key=True)
    jti = models.UUIDField(unique=True)
    expires = models.DateTimeField()
    scope = models.TextField()
    created = models.DateTimeField()
    updated = models.DateTimeField()
    application = models.ForeignKey(Oauth2ProviderApplication, models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(User, models.DO_NOTHING, blank=True, null=True)

    class Meta:
        """Oauth2ProviderIdtoken Meta class
        Note:
        Args:
          None
        Attributes:
        """

        managed = False
        db_table = 'oauth2_provider_idtoken'


class Oauth2ProviderRefreshtoken(models.Model):
    """Oauth2ProviderRefreshtokenclass
    Note:
    Args:
        models.Model
    Attributes:
    """

    id = models.BigAutoField(primary_key=True)
    token = models.CharField(max_length=255)
    access_token = models.OneToOneField(Oauth2ProviderAccesstoken, models.DO_NOTHING, blank=True, null=True)
    application = models.ForeignKey(Oauth2ProviderApplication, models.DO_NOTHING)
    user = models.ForeignKey(User, models.DO_NOTHING)
    created = models.DateTimeField()
    updated = models.DateTimeField()
    revoked = models.DateTimeField(blank=True, null=True)

    class Meta:
        """Oauth2ProviderRefreshtoken Meta class
        Note:
        Args:
          None
        Attributes:
        """

        managed = False
        db_table = 'oauth2_provider_refreshtoken'
        unique_together = (('token', 'revoked'),)


class WorkflowOrder(models.Model):
    """WorkflowOrder class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)                                 # order 아이디 : 기본키
    workflow_name = models.CharField(blank=True, null=True, max_length=30)  # workflow 이름 : (BMS, yoloe 등등..)
    order = models.IntegerField(blank=True, null=True)                      # 워크플로우 순서
    project_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디

    class Meta:
        """WorkflowOrder Meta class
        Note:
        Args:
          None
        Attributes:
        """
        # managed = False

        # DB 테이블 이름
        db_table = 'workflow_order'


# AUTONN STATUS 관련 테이블 ========================================================================

class Hyperparameter(models.Model):
    """Hyperparameter class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)                                 # order 아이디 : 기본키
    # autonn_status_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디
    # project_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디

    lr0 = models.FloatField(blank=True, null=True)
    lrf = models.FloatField(blank=True, null=True)
    lrc = models.FloatField(blank=True, null=True)
    momentum = models.FloatField(blank=True, null=True)
    weight_decay = models.FloatField(blank=True, null=True)
    warmup_epochs = models.FloatField(blank=True, null=True)
    warmup_momentum = models.FloatField(blank=True, null=True)
    warmup_bias_lr = models.FloatField(blank=True, null=True)
    box = models.FloatField(blank=True, null=True)
    cls = models.FloatField(blank=True, null=True)
    cls_pw = models.FloatField(blank=True, null=True)
    obj = models.FloatField(blank=True, null=True)
    obj_pw = models.FloatField(blank=True, null=True)
    iou_t = models.FloatField(blank=True, null=True)
    anchor_t = models.FloatField(blank=True, null=True)
    fl_gamma = models.FloatField(blank=True, null=True)
    hsv_h = models.FloatField(blank=True, null=True)
    hsv_s = models.FloatField(blank=True, null=True)
    hsv_v = models.FloatField(blank=True, null=True)
    degrees = models.FloatField(blank=True, null=True)
    translate = models.FloatField(blank=True, null=True)
    scale = models.FloatField(blank=True, null=True)
    shear = models.FloatField(blank=True, null=True)
    perspective = models.FloatField(blank=True, null=True)
    flipud = models.FloatField(blank=True, null=True)
    fliplr = models.FloatField(blank=True, null=True)
    mosaic = models.FloatField(blank=True, null=True)
    mixup = models.FloatField(blank=True, null=True)
    copy_paste = models.FloatField(blank=True, null=True)
    paste_in = models.FloatField(blank=True, null=True)
    loss_ota = models.FloatField(blank=True, null=True)
    label_smoothing = models.FloatField(blank=True, null=True)
    momentum_group0 = models.FloatField(blank=True, null=True)
    lr_group0 = models.FloatField(blank=True, null=True)
    momentum_group1 = models.FloatField(blank=True, null=True)
    lr_group1 = models.FloatField(blank=True, null=True)
    momentum_group2 = models.FloatField(blank=True, null=True)
    lr_group2 = models.FloatField(blank=True, null=True)

    class Meta:
        """Hyperparameter Meta class
        Note:
        Args:
          None
        Attributes:
        """
        # managed = False

        # DB 테이블 이름
        db_table = 'hyperparameter'

class Arguments(models.Model):
    """Arguments class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)                                 # order 아이디 : 기본키
    # autonn_status_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디
    # project_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디

    weights = models.TextField(blank=True, null=True, default='')
    resume = models.BooleanField(blank=True, null=True)
    cfg = models.TextField(blank=True, null=True, default='')
    data = models.TextField(blank=True, null=True, default='')
    hyp = models.TextField(blank=True, null=True, default='')
    epochs = models.IntegerField(blank=True, null=True)
    batch_size = models.IntegerField(blank=True, null=True)
    total_batch_size = models.IntegerField(blank=True, null=True)
    img_size = ArrayField(models.IntegerField(blank=True, null=True), blank=True, null=True)
    react = models.BooleanField(blank=True, null=True)
    nosave = models.BooleanField(blank=True, null=True)
    notest = models.BooleanField(blank=True, null=True)
    noautoanchor = models.BooleanField(blank=True, null=True)
    evolve = models.BooleanField(blank=True, null=True)
    cache_images = models.BooleanField(blank=True, null=True)
    image_weights = models.BooleanField(blank=True, null=True)
    multi_scale = models.BooleanField(blank=True, null=True)
    single_cls = models.BooleanField(blank=True, null=True)
    adam = models.BooleanField(blank=True, null=True)
    quad = models.BooleanField(blank=True, null=True)
    sync_bn = models.BooleanField(blank=True, null=True)
    linear_lr = models.BooleanField(blank=True, null=True)
    global_rank = models.IntegerField(blank=True, null=True)
    local_rank = models.IntegerField(blank=True, null=True)
    world_size = models.IntegerField(blank=True, null=True)
    workers = models.IntegerField(blank=True, null=True)
    bucket = models.TextField(blank=True, null=True, default='')
    project = models.TextField(blank=True, null=True, default='')
    name = models.TextField(blank=True, null=True, default='')
    exist_ok = models.BooleanField(blank=True, null=True)
    save_dir = models.TextField(blank=True, null=True, default='')
    label_smoothing = models.FloatField(blank=True, null=True)
    upload_dataset = models.BooleanField(blank=True, null=True)
    bbox_interval = models.IntegerField(blank=True, null=True)
    save_period = models.IntegerField(blank=True, null=True)
    artifact_alias = models.TextField(blank=True, null=True, default='')
    freeze = ArrayField(models.IntegerField(blank=True, null=True), blank=True, null=True)
    metric = models.TextField(blank=True, null=True, default='')
    device = models.TextField(blank=True, null=True)
    entity = models.TextField(blank=True, null=True, default='')

    class Meta:
        """Arguments Meta class
        Note:
        Args:
          None
        Attributes:
        """
        # managed = False

        # DB 테이블 이름
        db_table = 'arguments'

class System(models.Model):
    """System class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)                                 # order 아이디 : 기본키
    # autonn_status_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디
    # project_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디

    # torch_version = models.TextField(blank=True, null=True)       
    # devices = models.TextField(blank=True, null=True)       
    # gpu_model = models.TextField(blank=True, null=True)       
    # memory = models.FloatField(blank=True, null=True)
    
    torch = models.TextField(blank=True, null=True)
    cuda = models.TextField(blank=True, null=True)
    cudnn = models.TextField(blank=True, null=True)
    gpus = models.TextField(blank=True, null=True)

    class Meta:
        """System Meta class
        Note:
        Args:
          None
        Attributes:
        """
        # managed = False

        # DB 테이블 이름
        db_table = 'system'

class Basemodel(models.Model):
    """Basemodel class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)                                 # order 아이디 : 기본키
    # autonn_status_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디
    # project_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디

    model_name = models.TextField(blank=True, null=True)       
    model_size = models.TextField(blank=True, null=True)       

    class Meta:
        """Basemodel Meta class
        Note:
        Args:
          None
        Attributes:
        """
        # managed = False

        # DB 테이블 이름
        db_table = 'basemodel'
 
class ModelSummary(models.Model):
    """ModelSummary class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)                                 # order 아이디 : 기본키
    # autonn_status_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디
    # project_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디

    layers = models.IntegerField(blank=True, null=True)
    parameters = models.BigIntegerField(blank=True, null=True)
    gradients = models.BigIntegerField(blank=True, null=True)
    flops = models.FloatField(blank=True, null=True, default=0.0)

    class Meta:
        """ModelSummary Meta class
        Note:
        Args:
          None
        Attributes:
        """
        # managed = False

        # DB 테이블 이름
        db_table = 'model_summary'

class BatchSize(models.Model):
    """BatchSize class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)                                
    # autonn_status_id = models.IntegerField(blank=True, null=True)        
    # project_id = models.IntegerField(blank=True, null=True)              

    low = models.IntegerField(blank=True, null=True)
    high = models.IntegerField(blank=True, null=True)

    class Meta:
        """BatchSize Meta class
        Note:
        Args:
          None
        Attributes:
        """
        # managed = False

        # DB 테이블 이름
        db_table = 'batch_size'

class TrainDataset(models.Model):
    """TrainDataset class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)                                 # order 아이디 : 기본키
    # autonn_status_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디
    # project_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디

    found = models.IntegerField(blank=True, null=True)
    missing = models.IntegerField(blank=True, null=True)
    empty = models.IntegerField(blank=True, null=True)
    corrupted = models.IntegerField(blank=True, null=True)
    current = models.IntegerField(blank=True, null=True)
    total = models.IntegerField(blank=True, null=True)

    class Meta:
        """TrainDataset Meta class
        Note:
        Args:
          None
        Attributes:
        """
        # managed = False

        # DB 테이블 이름
        db_table = 'train_dataset'

class ValDataset(models.Model):
    """ValDataset class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)                                 # order 아이디 : 기본키
    # autonn_status_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디
    # project_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디

    found = models.IntegerField(blank=True, null=True)
    missing = models.IntegerField(blank=True, null=True)
    empty = models.IntegerField(blank=True, null=True)
    corrupted = models.IntegerField(blank=True, null=True)
    current = models.IntegerField(blank=True, null=True)
    total = models.IntegerField(blank=True, null=True)

    class Meta:
        """ValDataset Meta class
        Note:
        Args:
          None
        Attributes:
        """
        # managed = False

        # DB 테이블 이름
        db_table = 'val_dataset'

class Anchor(models.Model):
    """Anchor class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)                                 # order 아이디 : 기본키
    # autonn_status_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디
    # project_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디

    anchor2target_ratio = models.FloatField(blank=True, null=True)
    best_possible_recall = models.FloatField(blank=True, null=True)

    class Meta:
        """Anchor Meta class
        Note:
        Args:
          None
        Attributes:
        """
        # managed = False

        # DB 테이블 이름
        db_table = 'anchor'

class TrainStart(models.Model):
    """TrainStart class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)                                 # order 아이디 : 기본키
    # autonn_status_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디
    # project_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디

    status = models.TextField(blank=True, null=True)       
    epochs = models.IntegerField(blank=True, null=True)       

    class Meta:
        """TrainStart Meta class
        Note:
        Args:
          None
        Attributes:
        """
        # managed = False

        # DB 테이블 이름
        db_table = 'train_start'

class TrainLossLatest(models.Model):
    """TrainLossLatest class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)                                 # order 아이디 : 기본키
    # autonn_status_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디
    # project_id = models.IntegerField(blank=True, null=True)                 # workflow가 속한 프로젝트 아이디

    epoch = models.IntegerField(blank=True, null=True)
    total_epoch = models.IntegerField(blank=True, null=True)
    gpu_mem = models.TextField(blank=True, null=True)
    box = models.FloatField(blank=True, null=True)
    obj = models.FloatField(blank=True, null=True)
    cls = models.FloatField(blank=True, null=True)
    total = models.FloatField(blank=True, null=True)
    label = models.IntegerField(blank=True, null=True)
    step = models.IntegerField(blank=True, null=True)
    total_step = models.IntegerField(blank=True, null=True) 
    time = models.TextField(blank=True, null=True)

    class Meta:
        """TrainLossLatest Meta class
        Note:
        Args:
          None
        Attributes:
        """
        # managed = False

        # DB 테이블 이름
        db_table = 'train_loss_latest'

class TrainLossLastStep(models.Model):
    """TrainLossLastStep class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)                                 
    # # autonn_status_id = models.IntegerField(blank=True, null=True)             
    # train_loss_latest_id = models.IntegerField(blank=True, null=True)                 
    project_id = models.IntegerField(blank=False, null=False, default=0)
    project_version = models.IntegerField(blank=True, null=True, default=0)
    is_use = models.BooleanField(blank=False, null=False, default=True)

    epoch = models.IntegerField(blank=True, null=True)
    total_epoch = models.IntegerField(blank=True, null=True)
    gpu_mem = models.TextField(blank=True, null=True)
    box = models.FloatField(blank=True, null=True)
    obj = models.FloatField(blank=True, null=True)
    cls = models.FloatField(blank=True, null=True)
    total = models.FloatField(blank=True, null=True)
    label = models.IntegerField(blank=True, null=True)
    step = models.IntegerField(blank=True, null=True)
    total_step = models.IntegerField(blank=True, null=True) 
    time = models.TextField(blank=True, null=True)

    class Meta:
        """TrainLossLastStep Meta class
        Note:
        Args:
          None
        Attributes:
        """
        # managed = False

        # DB 테이블 이름
        db_table = 'train_loss_laststep'

class ValAccuracyLatest(models.Model):
    """ValAccuracyLatest class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)                                 
    # # autonn_status_id = models.IntegerField(blank=True, null=True)             
    train_loss_latest_id = models.IntegerField(blank=True, null=True)                 
    # project_id = models.IntegerField(blank=True, null=True)                 

    epoch = models.IntegerField(blank=True, null=True)   
    total_epoch = models.IntegerField(blank=True, null=True)   
    class_type = models.TextField(blank=True, null=True)   
    images = models.IntegerField(blank=True, null=True)   
    labels = models.BigIntegerField(blank=True, null=True)   
    P = models.FloatField(blank=True, null=True)   
    R = models.FloatField(blank=True, null=True)   
    mAP50 = models.FloatField(blank=True, null=True)   
    mAP50_95 = models.FloatField(blank=True, null=True)   
    step = models.IntegerField(blank=True, null=True)   
    total_step = models.IntegerField(blank=True, null=True)   
    time = models.TextField(blank=True, null=True)   

    class Meta:
        """ValAccuracyLatest Meta class
        Note:
        Args:
          None
        Attributes:
        """
        # managed = False

        # DB 테이블 이름
        db_table = 'val_accuracy_latest'

class ValAccuracyLastStep(models.Model):
    """ValAccuracyLastStep class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)                                 
    # # autonn_status_id = models.IntegerField(blank=True, null=True)             
    # train_loss_latest_id = models.IntegerField(blank=True, null=True)                 
    project_id = models.IntegerField(blank=False, null=False, default=0)
    project_version = models.IntegerField(blank=True, null=True, default=0)
    is_use = models.BooleanField(blank=False, null=False, default=True)

    # epoch = models.IntegerField(blank=True, null=True)   
    # total_epoch = models.IntegerField(blank=True, null=True)   
    class_type = models.TextField(blank=True, null=True)   
    images = models.IntegerField(blank=True, null=True)   
    labels = models.BigIntegerField(blank=True, null=True)   
    P = models.FloatField(blank=True, null=True)   
    R = models.FloatField(blank=True, null=True)   
    mAP50 = models.FloatField(blank=True, null=True)   
    mAP50_95 = models.FloatField(blank=True, null=True)   
    step = models.IntegerField(blank=True, null=True)   
    total_step = models.IntegerField(blank=True, null=True)   
    time = models.TextField(blank=True, null=True)   

    class Meta:
        """ValAccuracyLastStep Meta class
        Note:
        Args:
          None
        Attributes:
        """
        # managed = False

        # DB 테이블 이름
        db_table = 'val_accuracy_laststep'

class EpochSummary(models.Model):
    """EpochSummary class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)   
    project_id = models.IntegerField(blank=False, null=False, default=0)
    project_version = models.IntegerField(blank=True, null=True, default=0)
    is_use = models.BooleanField(blank=False, null=False, default=True)                              

    total_epoch = models.IntegerField(blank=True, null=True, default=0)      # 전체 epoch = x축의 오른쪽 끝 좌표
    current_epoch = models.IntegerField(blank=True, null=True, default=0)    # 현재 epoch = 지금 점을 찍을 x 좌표
    train_loss_box = models.FloatField(blank=True, null=True, default=0.0)     # (학습 training) 현재 epoch의 box loss 누적평균값
    train_loss_obj = models.FloatField(blank=True, null=True, default=0.0)     # (학습 training) 현재 epoch의 object loss 누적평균값
    train_loss_cls = models.FloatField(blank=True, null=True, default=0.0)     # (학습 training) 현재 epoch의 class average loss 누적평균값
    train_loss_total = models.FloatField(blank=True, null=True, default=0.0)   # (학습 training) 현재 epoch의 box, obj, cls loss의 가중평균값
    val_acc_P = models.FloatField(blank=True, null=True, default=0.0)          # (검증 validation) 현재 epoch의 Precision 누적평균값
    val_acc_R = models.FloatField(blank=True, null=True, default=0.0)           # (검증 validation) 현재 epoch의 Recall 누적평균값 
    val_acc_map50 = models.FloatField(blank=True, null=True, default=0.0)      # (검증 validation) 현재 epoch의 mAP50 (PR-curve 적분값; IoU=50% 기준) 
    val_acc_map = models.FloatField(blank=True, null=True, default=0.0)        # (검증 valication) 현재 epoch의 mAP50-95 (PR-curve 적분값; IoU=50%에서 95%의 가중평균값)
    epoch_time = models.FloatField(blank=True, null=True, default=0.0)     # 현재 epoch의 준비 + 학습 + 검증 까지 걸린 총 시간(단위:s) 막대
    total_time = models.FloatField(blank=True, null=True, default=0.0)     # 지금까지 걸린 시간 누적 (단위: s) 꺽은선

    class Meta:
        """EpochSummary Meta class
        Note:
        Args:
          None
        Attributes:
        """
        # managed = False

        # DB 테이블 이름
        db_table = 'epoch_summary'

class AutonnStatus(models.Model):
    """AutonnStatus class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)                
    project = models.ForeignKey("Project", related_name="project", on_delete=models.PROTECT, db_column="project", )
    progress= models.IntegerField(blank=True, null=True, default=0, db_column="progress" )

    hyperparameter = models.ForeignKey("Hyperparameter", related_name="hyperparameter", on_delete=models.PROTECT, db_column="hyperparameter", )
    arguments = models.ForeignKey("Arguments", related_name="arguments", on_delete=models.PROTECT, db_column="arguments", )
    system = models.ForeignKey("System", related_name="system", on_delete=models.PROTECT, db_column="system", )
    basemodel = models.ForeignKey("Basemodel", related_name="basemodel", on_delete=models.PROTECT, db_column="basemodel", )
    model_summary = models.ForeignKey("ModelSummary", related_name="model_summary", on_delete=models.PROTECT, db_column="model_summary", )
    batch_size = models.ForeignKey("BatchSize", related_name="batch_size", on_delete=models.PROTECT, db_column="batch_size", )
    train_dataset = models.ForeignKey("TrainDataset", related_name="train_dataset", on_delete=models.PROTECT, db_column="train_dataset", )
    val_dataset = models.ForeignKey("ValDataset", related_name="val_dataset", on_delete=models.PROTECT, db_column="val_dataset", )
    anchor = models.ForeignKey("Anchor", related_name="anchor", on_delete=models.PROTECT, db_column="anchor", )
    train_start = models.ForeignKey("TrainStart", related_name="train_start", on_delete=models.PROTECT, db_column="train_start", )
    train_loss_latest = models.ForeignKey("TrainLossLatest", related_name="train_loss_latest", on_delete=models.PROTECT, db_column="train_loss_latest",)
    val_accuracy_latest = models.ForeignKey("ValAccuracyLatest", related_name="val_accuracy_latest", on_delete=models.PROTECT, db_column="val_accuracy_latest", )
    # epoch_summary = models.ForeignKey("EpochSummary", related_name="epoch_summary", on_delete=models.PROTECT, db_column="epoch_summary", null=True, blank=True)

    class Meta:
        """AutonnStatus Meta class
        Note:
        Args:
          None
        Attributes:
        """
        # managed = False

        # DB 테이블 이름
        db_table = 'autonn_status'


