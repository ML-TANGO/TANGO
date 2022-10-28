"""
high level support for doing this and that.
"""
from django.contrib import admin
from .models import Node
from .models import Edge
from .models import Pth


class NodeAdmin(admin.ModelAdmin):
    # pylint: disable=too-few-public-methods, missing-class-docstring
    list = ('order', 'layer', 'parameters')


class EdgeAdmin(admin.ModelAdmin):
    # pylint: disable=too-few-public-methods, missing-class-docstring
    list = ('prior', 'next')


class PthAdmin(admin.ModelAdmin):
    # pylint: disable=too-few-public-methods, missing-class-docstring
    list = ('model')


admin.site.register(Node, NodeAdmin)
admin.site.register(Edge, EdgeAdmin)
admin.site.register(Pth, PthAdmin)
