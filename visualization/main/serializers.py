"""
high level support for doing this and that.
"""
from rest_framework import serializers
from .models import Node
from .models import Edge
from .models import Pth
from .models import Architecture


class NodeSerializer(serializers.ModelSerializer):
    # pylint: disable-msg=too-few-public-methods
    """A dummy docstring."""
    class Meta:  # pylint: disable-msg=too-few-public-methods
        """A dummy docstring."""
        model = Node
        fields = '__all__'


class EdgeSerializer(serializers.ModelSerializer):
    # pylint: disable-msg=too-few-public-methods
    """A dummy docstring."""
    class Meta:  # pylint: disable-msg=too-few-public-methods
        """A dummy docstring."""
        model = Edge
        fields = '__all__'


class PthSerializer(serializers.ModelSerializer):
    # pylint: disable-msg=too-few-public-methods
    """A dummy docstring."""
    class Meta:  # pylint: disable-msg=too-few-public-methods
        """A dummy docstring."""
        model = Pth
        fields = ('model_output',)


class ArchitectureSerializer(serializers.ModelSerializer):
    # pylint: disable-msg=too-few-public-methods
    """A dummy docstring."""
    class Meta:  # pylint: disable-msg=too-few-public-methods
        """A dummy docstring."""
        model = Architecture
        fields = ('architecture',)
