from django.db import models

class Info(models.Model):
    '''
        General information for an AutoNN project
    '''
    # ID -----------------------------------------------------------------------
    id = models.AutoField(primary_key=True)
    userid = models.CharField(blank=True, null=True, max_length=50, default='')
    project_id = models.CharField(blank=True, null=True, max_length=50, default='')
    process_id = models.CharField(blank=True, null=True, max_length=50)

    # From user ----------------------------------------------------------------
    target = models.CharField(blank=True, null=True, max_length=50, default='pc')
    dataset = models.CharField(blank=True, null=True, max_length=50, default='coco')
    task = models.CharField(blank=True, null=True, max_length=50, default='detection')

    # From autonn --------------------------------------------------------------
    status = models.CharField(blank=True, null=True, max_length=10, default='ready')
    progress = models.CharField(blank=True, null=True, max_length=20, default='unknown')
    model_type = models.CharField(blank=True, null=True, max_length=50, default='Not Selected')
    model_size = models.CharField(blank=True, null=True, max_length=50, default='Not Selected')
    batch_size = models.IntegerField(blank=True, null=True)

    # For checkpoint -----------------------------------------------------------
    batch_multiplier = models.FloatField(blank=True, default=1.0)
    epoch = models.IntegerField(blank=True, null=True)


class Node(models.Model):
    '''
        nn.Module (Nueral Network Module) as a node
    '''
    objects = models.Manager()
    order = models.IntegerField(primary_key=True)
    layer = models.CharField(max_length=200)
    parameters = models.TextField()


class Edge(models.Model):
    '''
        Connection among nn.Modules as an edge
    '''
    objects = models.Manager()
    id = models.IntegerField(primary_key=True)
    prior = models.IntegerField()
    next = models.IntegerField()

