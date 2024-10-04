"""urls module for tango
This module for admin.
Attributes:

Todo:
"""

from django.urls import path, include,re_path
from django.contrib import admin

from . import views, viewsProject

from rest_framework import routers
from rest_framework_jwt.views import obtain_jwt_token, verify_jwt_token, refresh_jwt_token

router = routers.DefaultRouter()
router.register(r'user', views.UserViewSet)         # 사용자 정보 확인
router.register(r'group', views.GroupViewSet)

urlpatterns = [

    re_path(r'^', include(router.urls)),

    re_path(r'^login', views.login, name='login'),      # 로그인
    re_path(r'^logout', views.logout, name='logout'),   # 로그아웃

    re_path(r'^signup', views.signup, name='signup'),                         # 회원 가입 요청
    re_path(r'^user_id_check', views.user_id_check, name='user_id_check'),    # 회원 가입 - 아이디 중복 확인

    re_path(r'^user_interval_time', views.user_interval_time, name='user_interval_time'),    # 사용자 세팅 - project 갱신 주기, autonn 시각화 갱신 주기 반환

    re_path(r'^project_list_get', viewsProject.project_list_get, name='project_list_get'),   # 프로젝트 리스트 조회
    re_path(r'^project_create', viewsProject.project_create, name='project_create'),         # 프로젝트 생성
    re_path(r'^project_delete', viewsProject.project_delete, name='project_delete'),         # 프로젝트 삭제
    re_path(r'^project_rename', viewsProject.project_rename, name='project_rename'),         # 프로젝트 이름 수정
    re_path(r'^project_info', viewsProject.project_info, name='project_info'),               # 프로젝트 삭제
    re_path(r'^project_update', viewsProject.project_update, name='project_update'),         # 프로젝트 이름 수정
    re_path(r'^project_type', viewsProject.project_type_update, name='project_type_update'), # 프로젝트 워크플로우 진행 방식 수정

    # 프로젝트 설명 수정
    re_path(r'^project_description_update', viewsProject.project_description_update, name='project_description_update'),


    re_path(r'^container_start', viewsProject.container_start, name='container_start'),       # 컨테이너 실행
    re_path(r'^next_pipeline_start', viewsProject.next_pipeline_start, name='next_pipeline_start'),       # 다음 버전의 파이프라인을 실행 (CI/CD pipeline 반복 기능)
    re_path(r'^status_request', viewsProject.status_request, name='status_request'),       # 컨테이너 실행 상태 확인 요청

    re_path(r'^download_nn_model', viewsProject.download_nn_model, name='download_nn_model'), # nn_model 다운로드(외부IDE연동)
    re_path(r'^upload_nn_model', viewsProject.upload_nn_model, name='upload_nn_model'),       # nn_model 업로드(외부IDE연동)

    re_path(r'^set_workflow', viewsProject.set_workflow, name='set_workflow'),       # workflow 셋팅
    
    re_path(r'^get_autonn_status', viewsProject.get_autonn_status, name='get_autonn_status'),       # get autonnstatus

    re_path(r'^get_common_folder_structure', viewsProject.get_common_folder_structure, name='get_common_folder_structure'),

]

