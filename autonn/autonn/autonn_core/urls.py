from django.urls import path, include
from django.views.generic import TemplateView
from rest_framework import routers
from . import views

router = routers.DefaultRouter()
router.register('node', views.NodeView, 'node')
router.register('edge', views.EdgeView, 'edge')
router.register('pth',  views.PthView,  'pth')

urlpatterns = [
    # database -----------------------------------------------------------------
    path('api/', include(router.urls)),
    path('info', views.InfoList, name='InfoList'),
    path('pth/', views.pth_list, name='PthList'),

    # action -------------------------------------------------------------------
    path('start', views.start, name='StartAutoNN'),
    path('status_request', views.status_request,  name='StatusRequest'),

    # default html -------------------------------------------------------------
    path('', TemplateView.as_view(template_name='index.html')), # /frontend/build/index.html
]
