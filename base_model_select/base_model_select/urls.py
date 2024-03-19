"""base_model_select URL Configuration

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
from django.contrib import admin
from django.urls import path, include
from backend import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('start', views.start, name="StartBMS"),
    path("stop", views.stop_api, name="StopBMS"),
    path("status_request", views.status_request, name="StatusRequestBMS"),
    path("get_ready_for_test", views.get_ready_for_test, name="get_ready_for_test"),
    #path('start?user_id=<user_id>&project_id=<project_id>', views.start_api)
    #path('start/', views.start_api),
    #path(include('start'), views.start_api),
    #path(include('stop')),
    #path(include('status_request')),
]
