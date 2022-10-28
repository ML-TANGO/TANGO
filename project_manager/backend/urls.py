"""backend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path, include
from django.contrib import admin
from django.conf.urls import url

from django.views.static import serve
from django.conf import settings

from django.views.generic import TemplateView

import oauth2_provider.views as oauth2_views


# react - public/index.html 연동
class HomeTemplateView(TemplateView):
    template_name = 'index.html'


urlpatterns = [

    path('admin/', admin.site.urls),  # 관리자 페이지 접근
    path('o/', include('oauth2_provider.urls', namespace='oauth2_provider')),

    # 서버 주소 시작
    path('api/', include('tango.urls')),
    path('', include('tango.urls_other')),

    # 정적 파일 주소
    url(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
    url(r'^static/(?P<path>.*)$', serve, {'document_root': settings.STATIC_ROOT}),

    # react - public/index.html 연동
    url("^.*", HomeTemplateView.as_view(), name='home'),

]
