from django.urls import path, include
from django.views.generic import TemplateView
from django.views.generic.base import RedirectView
from django.contrib.staticfiles.storage import staticfiles_storage
from rest_framework import routers
from . import views

router = routers.DefaultRouter()
router.register('info',     views.InfoView,     'info')
router.register('node',     views.NodeView,     'node')
router.register('edge',     views.EdgeView,     'edge')

urlpatterns = [
    # database -----------------------------------------------------------------
    path('api/', include(router.urls)),

    # action -------------------------------------------------------------------
    path('start', views.start, name='StartAutoNN'),
    path('resume', views.resume, name='ResumeAutoNN'),
    path('stop', views.stop, name='StopAutoNN'),
    path('status_request', views.status_request,  name='StatusRequest'),
    path('api/pth/', views.pth_list, name='PthList'),

    # default html -------------------------------------------------------------
    path('favicon.ico', RedirectView.as_view(url=staticfiles_storage.url('../favicon.ico'))),
    path('', TemplateView.as_view(template_name='index.html')), # /frontend/build/index.html
]
