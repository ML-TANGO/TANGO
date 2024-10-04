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
    device = models.CharField(blank=True, null=True, max_length=50, default='cpu')
    dataset = models.CharField(blank=True, null=True, max_length=50, default='coco')
    task = models.CharField(blank=True, null=True, max_length=50, default='detection')

    # From autonn --------------------------------------------------------------
    status = models.CharField(blank=True, null=True, max_length=10, default='ready')
    progress = models.CharField(blank=True, null=True, max_length=20, default='unknown')
    model_type = models.CharField(blank=True, null=True, max_length=50, default='not selected')
    model_size = models.CharField(blank=True, null=True, max_length=50, default=' ')
    model_viz = models.CharField(blank=True, null=True, max_length=50, default='not ready')
    batch_size = models.IntegerField(blank=True, null=True, default=16)

    # For checkpoint -----------------------------------------------------------
    batch_multiplier = models.FloatField(blank=True, default=1.0)
    epoch = models.IntegerField(blank=True, default=-1)
    best_acc = models.FloatField(blank=True, default=0.0)
    best_net = models.CharField(blank=True, null=True, max_length=300, default='')

    def __str__(self):
        uid = str(self.userid)
        pid = str(self.project_id)
        return uid + '/' + pid

    def print(self):
        print('-'*50)
        # print(f'id         : {self.id}')/
        print(f'uid        : {self.userid}')
        print(f'pid        : {self.project_id}')
        # print(f'process id : {self.process_id}')
        print('-'*50)
        print(f'status     : {self.status}')
        print(f'progress   : {self.progress}')
        print(f'model      : {self.model_type}{self.model_size}')
        print(f'visualizer : {self.model_viz}')
        # print('-'*50)
        # print(f'target     : {self.target}')
        # print(f'device     : {self.device}')
        # print(f'dataset    : {self.dataset}')
        # print(f'task       : {self.task}')
        print('-'*50)
        print(f'batch_size : {self.batch_size}')
        print(f'bs factor  : {self.batch_multiplier:.1f}')
        print(f'last epoch : {self.epoch}')
        print(f'best mAP   : {self.best_acc:.8f}')
        print(f'best model : {self.best_net}')
        print('-'*50)

    def reset(self):
        '''
        reset all values except ids with default
        '''
        print('-- reset with default --')
        self.target = 'pc'
        self.deivce = 'cpu'
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
    # ID -----------------------------------------------------------------------
    userid = models.CharField(blank=True, null=True, max_length=50, default='')
    project_id = models.CharField(blank=True, null=True, max_length=50, default='')

    # from Viz -----------------------------------------------------------------
    model_pth = models.CharField(blank=True, null=True, max_length=200, default='')
    model_yml = models.CharField(blank=True, null=True, max_length=200, default='')


