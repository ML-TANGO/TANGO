from django.db import models

class Info(models.Model):
    '''
        General information for an AutoNN project
    '''
    id = models.AutoField(primary_key=True)
    userid = models.CharField(blank=True, null=True, max_length=50, default='')
    project_id = models.CharField(blank=True, null=True, max_length=50, default='')
    target_yaml = models.FileField(upload_to="temp_files/", default='')
    data_yaml = models.FileField(upload_to="temp_files/", default='')
    task = models.CharField(blank=True, null=True, max_length=50, default='detection')
    status = models.CharField(blank=True, null=True, max_length=10, default='ready')
    process_id = models.CharField(blank=True, null=True, max_length=50)
    model_type = models.CharField(blank=True, null=True, max_length=50, default='Not Selected')
    model_size = models.CharField(blank=True, null=True, max_length=50, default='Not Selected')

