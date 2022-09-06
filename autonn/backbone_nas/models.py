'''
models.py
'''

from django.db import models


class URS(models.Model):
    '''User Requirement Specifications'''
    id = models.AutoField(primary_key=True)
    # 타겟 디바이스 정보
    target_device = models.CharField(blank=True, null=True,
                                     max_length=100, default='galaxy_s10')
    # 사용 데이터셋 경로 포함
    data_yaml = models.FileField(upload_to="temp_files/", default='')
    # 타입 정보 ( detection, segmentation... )
    type = models.CharField(blank=True, null=True,
                            max_length=50, default='detection')
