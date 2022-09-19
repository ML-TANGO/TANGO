from django.contrib import admin
from .models import Node
from .models import Edge
from .models import Pth

class NodeAdmin(admin.ModelAdmin):
    list=('order','layer','parameters')

class EdgeAdmin(admin.ModelAdmin):
    list=('prior','next')

class PthAdmin(admin.ModelAdmin):
    list=('model')

admin.site.register(Node, NodeAdmin)
admin.site.register(Edge, EdgeAdmin)
admin.site.register(Pth, PthAdmin)