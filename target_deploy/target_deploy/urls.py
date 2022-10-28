"""
urls.py
"""
from django.conf.urls import url
from . import views


urlpatterns = [

    url(r'^test', views.test, name='test'),                                                        # 테스트

    url(r'^deploy_image', views.deploy_image, name='deploy_image'),                                # 이미지 배포

]
