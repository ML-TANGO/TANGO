#!/usr/bin/env python
"""Django's command-line utility for administrative tasks.

Project layout (BASE_DIR):
    ├── Dockerfile / Dockerfile.cu130
    ├── manage.py
    ├── requirements*.txt
    ├── autonn_core
    │       ├─ django core files:
    │       │     __init__.py, admin.py, apps.py, models.py,
    │       │     serializers.py, urls.py, views.py
    │       ├─ migrations/
    │       ├─ datasets/ (coco, coco128, imagenet, voc, ChestXRay, ...)
    │       ├─ tango 💃
    │       │   ├─ common (cfg/, models/ incl. ofa_utils/)
    │       │   ├─ main (detect.py, train.py, val.py, visualize.py, ...)
    │       │   ├─ nas (predictors/, search_algorithm/)
    │       │   ├─ hpo / inference / viz
    │       │   └─ utils
    │       └─ tangochat 🗨️
    │           ├─ common (cfg/, models/)
    │           ├─ loader / tokenizer / tuner / inference
    │           └─ utils
    ├── config (settings.py, urls.py, asgi.py, wsgi.py)
    └── visualization (public/, src/, package.json)
"""

import os
import sys

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    try:
        from django.core.management import execute_from_command_line

    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
