"""serializer module for deep_frameowrk
This module for serializer.
Attributes:

Todo:

"""

from django.contrib import admin

# Register your models here.
from django.contrib.auth.models import User, Group
from rest_framework import serializers


# 사용자 목록
class UserSerializer(serializers.HyperlinkedModelSerializer):
    """UserSerializer class
    Note:
    Args:
        serializers.HyperlinkedModelSerializer
    Attributes:
    """
    class Meta: 
        """UserSerializer.meta class
        Note:
        Args:
            None
        Attributes:
        """

        model = User
        fields = ('url', 'username', 'email', 'groups')


# 사용자 그룹
class GroupSerializer(serializers.HyperlinkedModelSerializer):
    """GroupSerializer class
    Note:
    Args:
        serializers.HyperlinkedModelSerializer
    Attributes:
    """

    class Meta:
        """GroupSerializer.meta class
         Note:
         Args:
            None
         Attributes:
        """

        model = Group
        fields = ('url', 'name')