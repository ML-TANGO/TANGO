from django.db import models


class Info(models.Model):
    '''ResNet Container Information'''
    id = models.AutoField(primary_key=True)

    # user id
    userid = models.CharField(blank=True, null=True, max_length=50, default='')

    # project id
    project_id = models.CharField(blank=True, null=True, max_length=50, default='')

    # target device
    target_yaml = models.FileField(upload_to="temp_files/", default='')

    # datasets
    data_yaml = models.FileField(upload_to="temp_files/", default='')

    # task ( detection, segmentation... )
    task = models.CharField(blank=True, null=True, max_length=50, default='classification')

    # status ( ready, running, stopped, finished )
    status = models.CharField(blank=True, null=True, max_length=10, default='ready')

    # # thread index ( 0, 1, ..., N )
    # thread_id = models.IntegerField(blank=True, null=True)

    # process index ( 0, 1, ..., N )
    process_id = models.CharField(blank=True, null=True, max_length=50)
