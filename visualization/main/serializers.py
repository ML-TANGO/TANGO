"""
high level support for doing this and that.
"""
from rest_framework import serializers
from .models import Node
from .models import Edge
from .models import Pth
from .models import Architecture
from .models import Start
from .models import Status
from .models import Running
from .models import Stop


# from .models import Stop


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


class StartSerializer(serializers.ModelSerializer):
    # pylint: disable-msg=too-few-public-methods
    """A dummy docstring."""

    class Meta:  # pylint: disable-msg=too-few-public-methods
        """A dummy docstring."""
        model = Start
        fields = '__all__'


class StatusSerializer(serializers.ModelSerializer):
    # pylint: disable-msg=too-few-public-methods
    """A dummy docstring."""

    class Meta:  # pylint: disable-msg=too-few-public-methods
        """A dummy docstring."""
        model = Status
        fields = '__all__'


class RunningSerializer(serializers.ModelSerializer):
    # pylint: disable-msg=too-few-public-methods
    """A dummy docstring."""

    class Meta:  # pylint: disable-msg=too-few-public-methods
        """A dummy docstring."""
        model = Running
        fields = '__all__'


class StopSerializer(serializers.ModelSerializer):
    # pylint: disable-msg=too-few-public-methods
    """A dummy docstring."""
    class Meta:   # pylint: disable-msg=too-few-public-methods
        """A dummy docstring."""
        model = Stop
        fields = '__all__'
