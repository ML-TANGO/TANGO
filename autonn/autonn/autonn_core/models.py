from django.db import models
from django.utils import timezone

class Info(models.Model):
    '''
    General information for an AutoNN project
    '''
    # ID -----------------------------------------------------------------------
    id = models.AutoField(primary_key=True)
    userid = models.CharField(max_length=50, blank=True, null=True, default='')
    project_id = models.CharField(max_length=50, blank=True, null=True, default='')
    process_id = models.CharField(max_length=50, blank=True, null=True)

    # From user ----------------------------------------------------------------
    target = models.CharField(max_length=50, blank=True, null=True, default='pc')
    device = models.CharField(max_length=50, blank=True, null=True, default='cpu')
    dataset = models.CharField(max_length=50, blank=True, null=True, default='coco')
    task = models.CharField(max_length=50, blank=True, null=True, default='detection')

    # From autonn --------------------------------------------------------------
    status = models.CharField(max_length=10, blank=True, null=True, default='ready')
    progress = models.CharField(max_length=20, blank=True, null=True, default='unknown')
    model_type = models.CharField(max_length=50, blank=True, null=True, default='not selected')
    model_size = models.CharField(max_length=50, blank=True, null=True, default=' ')
    model_viz = models.CharField(max_length=50, blank=True, null=True, default='not ready')
    batch_size = models.IntegerField(blank=True, null=True, default=16)

    # For checkpoint -----------------------------------------------------------
    batch_multiplier = models.FloatField(blank=True, default=1.0)
    epoch = models.IntegerField(blank=True, default=-1)
    best_acc = models.FloatField(blank=True, default=0.0)
    best_net = models.CharField(max_length=300, blank=True, null=True, default='')

    # Timestamps ---------------------------------------------------------------
    created_at = models.DateTimeField(default=timezone.now, editable=False)  # 생성 시점
    updated_at = models.DateTimeField(auto_now=True)                         # 매 저장시점

    class Meta:
        indexes = [
            models.Index(fields=['status', '-updated_at']),
            models.Index(fields=['-updated_at']),
        ]
        ordering = ['-updated_at', '-id']  # 기본 정렬

    def __str__(self):
        return f"{self.userid}/{self.project_id}"

    def print(self):
        print('-' * 50)
        print(f'uid        : {self.userid}')
        print(f'pid        : {self.project_id}')
        print('-' * 50)
        print(f'status     : {self.status}')
        print(f'progress   : {self.progress}')
        print(f'model      : {self.model_type}{self.model_size}')
        print(f'visualizer : {self.model_viz}')
        print('-' * 50)
        print(f'batch_size : {self.batch_size}')
        print(f'bs factor  : {self.batch_multiplier:.1f}')
        print(f'last epoch : {self.epoch}')
        print(f'best mAP   : {self.best_acc:.8f}')
        print(f'best model : {self.best_net}')
        print('-' * 50)

    def reset(self):
        '''
        Reset all values except IDs to default
        '''
        print('-- reset with default --')
        self.target = 'pc'
        self.device = 'cpu'
        self.dataset = 'coco'
        self.task = 'detection'
        self.status = 'ready'
        self.progress = 'unknown'
        self.model_type = 'not selected'
        self.model_size = ' '
        self.model_viz = 'not ready'
        self.batch_size = 16
        self.batch_multiplier = 1.0
        self.epoch = -1
        self.best_acc = 0.0
        self.best_net = ''


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


class Pth(models.Model):
    '''
    PyTorch model generated from Viz
    '''
    userid = models.CharField(max_length=50, blank=True, null=True, default='')
    project_id = models.CharField(max_length=50, blank=True, null=True, default='')
    model_pth = models.CharField(max_length=200, blank=True, null=True, default='')
    model_yml = models.CharField(max_length=200, blank=True, null=True, default='')


