'''
serializers.py
'''

from rest_framework import serializers

from .models import Info


class InfoSerializer(serializers.ModelSerializer):
    '''Neck-NAS Info'''
    class Meta:
        '''meta'''
        model = Info
        fields = '__all__'
