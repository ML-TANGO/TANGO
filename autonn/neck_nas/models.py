'''
models.py
'''

from django.db import models


class NasContainerInfo(models.Model):
    '''Neck NAS Container Information'''
    id = models.AutoField(primary_key=True)

    # user id
    userid = models.CharField(blank=True, null=True, max_length=50, default='')

    # project id
    project_id = models.CharField(blank=True, null=True, max_length=50, default='')

    # target device
    target_device = models.CharField(blank=True, null=True, max_length=100, default='rk3399pro')

    # datasets
    data_yaml = models.FileField(upload_to="temp_files/", default='')

    # task ( detection, segmentation... )
    task = models.CharField(blank=True, null=True, max_length=50, default='detection')

    # status ( ready, running, stopped, finished )
    status = models.CharField(blank=True, null=True, max_length=100, default='ready')
