"""urls module for deep_frameowrk
This module for admin.
Attributes:
Todo:
"""

from django.contrib import admin

# Register your models here.

from django.urls import path, include
from django.conf.urls import url
from . import views, viewsProject
from django.contrib import admin

from rest_framework_jwt.views import obtain_jwt_token, verify_jwt_token, refresh_jwt_token

from rest_framework import routers

router = routers.DefaultRouter()
# 사용자 정보 확인
router.register(r'user', views.UserViewSet)                                             
router.register(r'group', views.GroupViewSet)


urlpatterns = [

    url(r'^', include(router.urls)),

    # 로그인
    url(r'^login', views.login, name='login'),
    # 로그아웃
    url(r'^logout', views.logout, name='logout'),

    # 회원 가입 요청
    url(r'^signup', views.signup, name='signup'),
    # 회원 가입 - 아이디 중복 확인
    url(r'^user_id_check', views.user_id_check, name='user_id_check'),

    # 프로젝트 리스트 조회
    url(r'^project_list_get', viewsProject.project_list_get, name='project_list_get'),

    # 프로젝트 생성
    url(r'^project_create', viewsProject.project_create, name='project_create'),

    # 프로젝트 삭제
    url(r'^project_delete', viewsProject.project_delete, name='project_delete'),

    # 프로젝트 이름 수정
    url(r'^project_rename', viewsProject.project_rename, name='project_rename'),

    # 프로젝트 설명 수정
    url(r'^project_description_update', 
        viewsProject.project_description_update, name='project_description_update'),

    # 프로젝트 삭제
    url(r'^project_info', viewsProject.project_info, name='project_info'),

    # 프로젝트 이름 수정
    url(r'^project_update', viewsProject.project_update, name='project_update'),

    # 타겟 yaml 파일 생성
    url(r'^target_check', viewsProject.target_check, name='target_check'),
    # 데이터 셋 유효성 검사
    url(r'^dataset_check', viewsProject.dataset_check, name='dataset_check'),

    # 서버 IP 정보 획득
    url(r'^get_server_ip', views.get_server_ip, name='get_server_ip'),

]
