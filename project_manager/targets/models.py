from django.db import models

# Create your models here.

class Target(models.Model):
    """Target class
    Note:
    Args:
        models.Model
    Attributes:
    """
    id = models.AutoField(primary_key=True)                                             # 타겟 아이디 : 기본키
    target_name = models.CharField(blank=True, null=True, max_length=30)                # 타겟 이름
    create_user = models.CharField(blank=True, null=True, max_length=50)                # 생성자
    create_date = models.CharField(blank=True, null=True, max_length=50)                # 생성 날짜
    target_info = models.CharField(blank=True, null=True, max_length=30)                # 타겟 정보
    target_engine = models.CharField(blank=True, null=True, max_length=30)              # 타겟 정보 - engine
    target_os = models.CharField(blank=True, null=True, max_length=50)                  # 타겟 정보 - os
    target_cpu = models.CharField(blank=True, null=True, max_length=50)                 # 타겟 정보 - cpu
    target_acc = models.CharField(blank=True, null=True, max_length=50)                 # 타겟 정보 - acc
    target_memory = models.CharField(blank=True, null=True, max_length=50)              # 타겟 정보 - memory
    nfs_ip = models.CharField(blank=True, null=True, max_length=50)                     # 타겟 정보 - nfs_ip (for k8s)
    nfs_path = models.CharField(blank=True, null=True, max_length=50)                   # 타겟 정보 - nfs_path (for k8s)
    target_host_ip = models.CharField(blank=True, null=True, max_length=50)             # 타겟 정보 - host_ip
    target_host_port = models.CharField(blank=True, null=True, max_length=50)           # 타겟 정보 - host_port
    target_host_service_port = models.CharField(blank=True, null=True, max_length=30)   # 타겟 정보 - host_service_port
    target_image = models.CharField(blank=True, null=True, max_length=10485760)         # 타겟 이미지

    order = models.IntegerField(default=0, blank=False, null=False)         # 타겟 이미지

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
