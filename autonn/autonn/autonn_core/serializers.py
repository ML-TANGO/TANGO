"""
high level support for doing this and that.
"""
from rest_framework import serializers

from .models import Node
from .models import Edge


class NodeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Node
        fields = '__all__'


class EdgeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Edge
        fields = '__all__'
