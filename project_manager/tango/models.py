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
    target = models.CharField(blank=True, null=True, max_length=100)                           # 타겟 정보
    target_yaml_path = models.CharField(blank=True, null=True, max_length=1024, default='')    # 타겟 yaml 파일 경로
    dataset_path = models.CharField(blank=True, null=True, max_length=200, default='')         # 사용 데이터 셋 경로
    data_yaml_path = models.CharField(blank=True, null=True, max_length=1024, default='')      # 데이터 셋 yaml 파일 경로

    step = models.IntegerField(blank=True, null=True, default=0)                               # 신경망 생성 단계
    step_status = models.BooleanField(blank=True, null=True, default=False)                    # 신경망 생성 상태

    # 타입 정보 ( detection, segmentation... )
    type = models.CharField(blank=True, null=True, max_length=50)

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


class Target(models.Model):
    """Target class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)                                        # 타겟 아이디 : 기본키
    target_name = models.CharField(blank=True, null=True, max_length=30)           # 타겟 이름
    create_user = models.CharField(blank=True, null=True, max_length=50)           # 생성자
    create_date = models.CharField(blank=True, null=True, max_length=50)           # 생성 날짜
    target_cpu = models.CharField(blank=True, null=True, max_length=200)           # 타겟 정보 - cpu
    target_gpu = models.CharField(blank=True, null=True, max_length=200)           # 타겟 정보 - gpu
    target_memory = models.CharField(blank=True, null=True, max_length=200)        # 타겟 정보 - memory
    target_model = models.CharField(blank=True, null=True, max_length=200)         # 타겟 정보 - 하드웨어 모델
    target_image = models.CharField(blank=True, null=True, max_length=10485760)    # 타겟 이미지

    class Meta:
        """Target Meta class
        Note:
        Args:
          None
        Attributes:
        """
        # managed = False

        # DB 테이블 이름
        db_table = 'target'


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

