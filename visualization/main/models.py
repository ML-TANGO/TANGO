"""
high level support for doing this and that.
"""
from django.db import models


class Node(models.Model):
    # pylint: disable=too-few-public-methods, missing-class-docstring
    objects = models.Manager()
    order = models.IntegerField(primary_key=True)
    layer = models.CharField(max_length=200)
    parameters = models.TextField()


class Edge(models.Model):
    # pylint: disable=too-few-public-methods, missing-class-docstring
    objects = models.Manager()
    id = models.IntegerField(primary_key=True)
    prior = models.IntegerField()
    next = models.IntegerField()


class Pth(models.Model):
    # pylint: disable=too-few-public-methods, missing-class-docstring
    objects = models.Manager()
    model_output = models.CharField(max_length=200)


class Architecture(models.Model):
    # pylint: disable=too-few-public-methods, missing-class-docstring
    objects = models.Manager()
    architecture = models.FileField()


class Start(models.Model):
    # pylint: disable=too-few-public-methods, missing-class-docstring
    msg = models.CharField(max_length=200, null=True, default='')
    user_id = models.CharField(max_length=200, null=True, default='')
    project_id = models.CharField(max_length=200, null=True, default='')


class Status(models.Model):
    # pylint: disable=too-few-public-methods, missing-class-docstring
    msg = models.CharField(max_length=200, null=True, default='')
    user_id = models.CharField(max_length=200, null=True, default='')
    project_id = models.CharField(max_length=200, null=True, default='')


class Running(models.Model):
    # pylint: disable=too-few-public-methods, missing-class-docstring
    objects = models.Manager()
    msg = models.CharField(max_length=200, null=True, default='')
    timestamp = models.CharField(max_length=200, null=True, default='')


class Stop(models.Model):
    # pylint: disable=too-few-public-methods, missing-class-docstring
    msg = models.CharField(max_length=200, null=True, default='')
    user_id = models.CharField(max_length=200, null=True, default='')
    project_id = models.CharField(max_length=200, null=True, default='')
