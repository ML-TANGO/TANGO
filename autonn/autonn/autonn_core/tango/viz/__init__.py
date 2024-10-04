import os, sys
from pathlib import Path

from django.apps import apps
Info = apps.get_model('autonn_core', 'Info')
Node = apps.get_model('autonn_core', 'Node')
Edge = apps.get_model('autonn_core', 'Edge')
