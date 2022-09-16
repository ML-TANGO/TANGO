from django.db import models

class Node(models.Model):
    order = models.IntegerField(primary_key=True)
    layer = models.CharField(max_length=200)
    parameters = models.TextField()

class Edge(models.Model):
    id = models.IntegerField(primary_key=True)
    prior = models.IntegerField()
    next = models.IntegerField()

class Pth(models.Model):
    model_output = models.CharField(max_length=200)

class Architecture(models.Model):
    architecture = models.FileField()
