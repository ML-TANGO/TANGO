"""viz2code URL Configuration

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
from django.urls import path, include, re_path
from django.views.generic import TemplateView
from rest_framework import routers
from main import views

router = routers.DefaultRouter()
router.register('node', views.NodeView, 'node')
router.register('edge', views.EdgeView, 'edge')
router.register('architecture', views.ArchitectureView, 'architecture')
router.register('running', views.RunningView, 'running')
# router.register('pth',views.PthView,'pth')

urlpatterns = [
    path('api/', include(router.urls)),
    # path('', include('main.urls')),
    path('api/pth/', views.pthlist),
    path('start', views.startList),
    path('stop', views.stopList),
    path('status_report', views.statusList),
    path('api/running/', views.runningList),
    path('admin/', admin.site.urls),
    # path('', views.mainList),
    # 220623 merge
    re_path('', TemplateView.as_view(template_name='index.html')),
]
