"""urls module for tango
This module for admin.
Attributes:

Todo:
"""

from django.urls import path, include
from django.conf.urls import url
from django.contrib import admin

from . import views, viewsProject, viewsTarget, viewsDataset

from rest_framework import routers
from rest_framework_jwt.views import obtain_jwt_token, verify_jwt_token, refresh_jwt_token

router = routers.DefaultRouter()
router.register(r'user', views.UserViewSet)         # 사용자 정보 확인
router.register(r'group', views.GroupViewSet)

urlpatterns = [

    url(r'^', include(router.urls)),

    url(r'^login', views.login, name='login'),      # 로그인
    url(r'^logout', views.logout, name='logout'),   # 로그아웃

    url(r'^signup', views.signup, name='signup'),                         # 회원 가입 요청
    url(r'^user_id_check', views.user_id_check, name='user_id_check'),    # 회원 가입 - 아이디 중복 확인

    url(r'^project_list_get', viewsProject.project_list_get, name='project_list_get'),   # 프로젝트 리스트 조회
    url(r'^project_create', viewsProject.project_create, name='project_create'),         # 프로젝트 생성
    url(r'^project_delete', viewsProject.project_delete, name='project_delete'),         # 프로젝트 삭제
    url(r'^project_rename', viewsProject.project_rename, name='project_rename'),         # 프로젝트 이름 수정
    url(r'^project_info', viewsProject.project_info, name='project_info'),               # 프로젝트 삭제
    url(r'^project_update', viewsProject.project_update, name='project_update'),         # 프로젝트 이름 수정
    url(r'^project_type', viewsProject.project_type_update, name='project_type_update'), # 프로젝트 워크플로우 진행 방식 수정

    # 프로젝트 설명 수정
    url(r'^project_description_update', viewsProject.project_description_update, name='project_description_update'),

    url(r'^target_check', viewsProject.target_check, name='target_check'),      # 타겟 yaml 파일 생성
    url(r'^dataset_check', viewsProject.dataset_check, name='dataset_check'),   # 데이터 셋 유효성 검사

    url(r'^get_server_ip', views.get_server_ip, name='get_server_ip'),          # 서버 IP 정보 획득

    url(r'^target_create', viewsTarget.target_create, name='target_create'),    # 타겟 생성
    url(r'^target_read', viewsTarget.target_read, name='target_read'),          # 타겟 조회 (리스트)
    url(r'^target_update', viewsTarget.target_update, name='target_update'),    # 타겟 수정
    url(r'^target_delete', viewsTarget.target_delete, name='target_delete'),    # 타겟 삭제
    url(r'^target_info', viewsTarget.target_info, name='target_info'),    # 타겟 정보 가져오기

    url(r'^container_start', viewsProject.container_start, name='container_start'),       # 컨테이너 실행
    url(r'^status_result', viewsProject.status_result, name='status_result'),       # 컨테이너 실행 상태 확인
    url(r'^status_request', viewsProject.status_request, name='status_request'),       # 컨테이너 실행 상태 확인 요청

    url(r'^get_dataset_list', viewsDataset.get_dataset_list, name='get_dataset_list'),       # 컨테이너 실행 상태 확인 요청


    url(r'^download_nn_model', viewsProject.download_nn_model, name='download_nn_model'), # nn_model 다운로드(외부IDE연동)
    url(r'^upload_nn_model', viewsProject.upload_nn_model, name='upload_nn_model'),       # nn_model 업로드(외부IDE연동)

    url(r'^set_workflow', viewsProject.set_workflow, name='set_workflow'),       # workflow 셋팅
]

