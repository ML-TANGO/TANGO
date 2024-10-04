"""urls module for tango
This module for admin.
Attributes:

Todo:
"""

from django.urls import path, include,re_path
from django.contrib import admin

from . import viewsProject

urlpatterns = [
    re_path(r'^status_report', viewsProject.status_report, name='status_report'),  # 컨테이너 상태 값 반환
    re_path(r'^status_update', viewsProject.status_update, name='status_update'),       # 컨테이너 Auto NN 업데이트
]

