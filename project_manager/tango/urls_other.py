"""urls module for tango
This module for admin.
Attributes:

Todo:
"""

from django.urls import path, include
from django.conf.urls import url
from django.contrib import admin

from . import viewsContainer

urlpatterns = [
    url(r'^status_report', viewsContainer.status_report, name='status_report'),  # 컨테이너 상태 값 반환
]
