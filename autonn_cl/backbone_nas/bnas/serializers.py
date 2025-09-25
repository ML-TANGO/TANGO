'''
serializers.py
'''

from rest_framework import serializers

from .models import Info


class InfoSerializer(serializers.ModelSerializer):
    '''Backbone-NAS Info'''
    class Meta:
        '''meta'''
        model = Info
        fields = '__all__'
