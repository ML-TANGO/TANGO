"""serializer module for tango
This module for serializer.
Attributes:

Todo:

"""

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
        """UserSerializer Meta class
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
        """GroupSerializer Meta class
         Note:
         Args:
            None
         Attributes:
        """

        model = Group
        fields = ('url', 'name')