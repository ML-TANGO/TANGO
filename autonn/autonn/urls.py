'''
urls.py
'''

from django.conf.urls import url
from . import views


urlpatterns = [
    # 테스트
    url(r'^test', views.test, name='test'),
    # 신경망 생성
    url(r'^create_neural', views.create_neural, name='create_neural'),

]
