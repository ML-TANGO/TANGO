"""
url.py
"""
from django.conf.urls import url
from . import views


urlpatterns = [

    url(r'^test', views.test, name='test'),                                                        # 테스트

    url(r'^create_image', views.create_image, name='create_image'),                                # 이미지 생성

]
