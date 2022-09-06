'''
serializers.py
'''

from rest_framework import serializers

from .models import URS


class URSSerializer(serializers.ModelSerializer):
    '''URS'''
    class Meta:
        '''meta'''
        model = URS
        fields = '__all__'
