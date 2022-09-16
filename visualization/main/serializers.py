from rest_framework import serializers
from .models import Node
from .models import Edge
from .models import Pth
from .models import Architecture



class NodeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Node
        fields = '__all__'


class EdgeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Edge
        fields = '__all__'


class PthSerializer(serializers.ModelSerializer):
    class Meta:
        model = Pth
        fields = ('model_output',)


class ArchitectureSerializer(serializers.ModelSerializer):
    class Meta:
        model = Architecture
        fields = ('architecture',)
