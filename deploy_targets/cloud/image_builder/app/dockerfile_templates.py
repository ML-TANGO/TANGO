DOCKERFILE_TEMPLATE_DEFAULT = r"""# syntax = docker/dockerfile:1.0-experimental
FROM --platform={{architecture}} {{ src }}
MAINTAINER Backend.AI Manager

ENV PYTHONUNBUFFERED=1 \
{%- if envs|length == 0 %}
    LANG=C.UTF-8
{%- else %}
    LANG=C.UTF-8 \
{%- endif %}
{%- for env in envs %}
    {%- if loop.index0 == (loop.length)-1 %}
    {{ env }}
    {%- else %}
    {{ env }} \
    {%- endif %}
{%- endfor %}

RUN useradd --user-group --create-home --no-log-init --shell /bin/bash work

RUN apt-get update && \
    apt-get install -y \
        ca-certificates {% if allow_root %} sudo {% endif %} \
        wget curl git-core \
        vim-tiny zip unzip \
        python3 python3-pip \
        libssl-dev libgl1-mesa-glx \
        proj-bin libproj-dev \
        libgeos-dev libgeos++-dev libglib2.0-0 \
        mime-support ncurses-term \
{%- if packages['apt']|length == 0 %}
        gcc g++ && \
{%- else %}
        gcc g++ \
{%- endif %}
{%- for custom in packages['apt'] %}
    {%- if loop.index0 == (loop.length)-1 %}
        {{ custom }} && \
    {%- else %}
        {{ custom }} \
    {%- endif %}
{%- endfor %}
    apt-get clean && \
    rm -rf /var/lib/apt/lists/ && \
    rm -rf /root/.cache && \
    rm -rf /tmp/*

{% if allow_root %} 
RUN echo "work ALL=(ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers 
{% endif -%}
RUN ln -sf /usr/share/terminfo/x/xterm-color /usr/share/terminfo/x/xterm-256color

COPY {{ copy_path }} {{ workdir }}
WORKDIR {{ workdir }}

RUN python3 -m pip install -U pip setuptools && \
    python3 -m pip install Pillow && \
    python3 -m pip install h5py && \
    python3 -m pip install ipython && \
    python3 -m pip install jupyter && \
{%- if packages['pypi']|length == 0 %}
    python3 -m pip install jupyterlab
{%- else %}
    python3 -m pip install jupyterlab && \
{%- endif %}
{%- for custom in packages['pypi'] %}
    {%- if loop.index0 == (loop.length)-1 %}
    python3 -m pip install {{ custom }} 
    {%- else %}
    python3 -m pip install {{ custom }} && \
    {%- endif %}
{%- endfor %}

RUN python3 -m pip install -r requirements.txt

{% if packages['conda']|length > 0 -%}
RUN {% for custom in packages['conda'] -%}
        {% if loop.index0 != (loop.length)-1 -%}
            conda install {{ custom }} && \
        {%- else -%}
            conda install {{ custom }} 
        {%- endif %}
    {% endfor %}
{%- endif %}


"""  # noqa

DOCKERFILE_TEMPLATE_APP = r"""# syntax = docker/dockerfile:1.0-experimental
FROM {{ src }}
MAINTAINER Backend.AI Manager

RUN useradd --user-group --create-home --no-log-init --shell /bin/bash work

ENV PYTHONUNBUFFERED=1 \
{%- if envs|length == 0 %}
    LANG=C.UTF-8
{%- else %}
    LANG=C.UTF-8 \
{%- endif %}
{%- for env in envs %}
    {%- if loop.index0 == (loop.length)-1 %}
    {{ env }}
    {%- else %}
    {{ env }} \
    {%- endif %}
{%- endfor %}

RUN apt update -y && \
    apt install -y ncurses-term {% if allow_root %} sudo {% endif %} && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/ && \
    rm -rf /root/.cache && \
    rm -rf /tmp/*

{% if allow_root %} 
RUN echo "work ALL=(ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers 
{% endif -%}
RUN ln -sf /usr/share/terminfo/x/xterm-color /usr/share/terminfo/x/xterm-256color

RUN {{ runtime_path }} -m pip install -U pip setuptools && \
    {{ runtime_path }} -m pip install Pillow && \
    {{ runtime_path }} -m pip install h5py && \
    {{ runtime_path }} -m pip install ipython && \
    {{ runtime_path }} -m pip install jupyter && \
{%- if packages['pip']|length == 0 %}
    {{ runtime_path }} -m pip install jupyterlab
{%- else %}
    {{ runtime_path }} -m pip install jupyterlab && \
{%- endif %}
{%- for custom in packages['pip'] %}
    {%- if loop.index0 == (loop.length)-1 %}
    {{ runtime_path }} -m pip install {{ custom }} 
    {%- else %}
    {{ runtime_path }} -m pip install {{ custom }} && \
    {%- endif %}
{%- endfor %}

{% if packages['conda']|length > 0 -%}
RUN {% for custom in packages['conda'] -%}
        {% if loop.index0 != (loop.length)-1 -%}
            conda install {{ custom }} && \
        {%- else -%}
            conda install {{ custom }} 
        {%- endif %}
    {% endfor %}
{%- endif %}

COPY ./service-defs /etc/backend.ai/service-defs
LABEL ai.backend.kernelspec="1" \
      ai.backend.envs.corecount="OPENBLAS_NUM_THREADS,OMP_NUM_THREADS,NPROC" \
      ai.backend.features="{% if has_ipykernel %}query batch {% endif %}uid-match" \
      ai.backend.resource.min.cpu="{{ min_cpu }}" \
      ai.backend.resource.min.mem="{{ min_mem }}" \
      ai.backend.resource.preferred.shmem="{{ pref_shmem }}" \
      ai.backend.accelerators="{{ accelerators | join(',') }}" \
{%- if 'cuda' is in accelerators %}
      ai.backend.resource.min.cuda.device=1 \
      ai.backend.resource.min.cuda.shares=0.1 \
{%- endif %}
      ai.backend.base-distro="{{ base_distro }}" \
{%- if service_ports %}
      ai.backend.service-ports="{% for item in service_ports -%}
          {{- item['name'] }}:
          {{- item['protocol'] }}:
          {%- if (item['ports'] | length) > 1 -%}
              [{{ item['ports'] | join(',') }}]
          {%- else -%}
              {{ item['ports'][0] }}
          {%- endif -%}
          {{- ',' if not loop.last }}
      {%- endfor %}" \
{%- endif %}
      ai.backend.runtime-type="{{ runtime_type }}" \
      ai.backend.runtime-path="{{ runtime_path }}"

"""  # noqa

DOCKERFILE_TEMPLATE_YOLO5 = r"""
FROM ultralytics/yolov5:latest-arm64
MAINTAINER Backend.AI Manager

RUN useradd --user-group --create-home --no-log-init --shell /bin/bash work

ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    BASE_DIR=/usr/src/app \
    UPLOADS_PATH=/upload \
    MODEL_OUTPUTS=/output

RUN apt-get update -y && \
    apt-get install -y ncurses-term sudo vim net-tools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/ && \
    rm -rf /root/.cache && \
    rm -rf /tmp/*

RUN ln -sf /usr/share/terminfo/x/xterm-color /usr/share/terminfo/x/xterm-256color
# RUN echo "work ALL=(ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers 

RUN python3 -m pip install -U pip setuptools && \
    python3 -m pip install Pillow && \
    python3 -m pip install h5py && \
    python3 -m pip install ipython && \
    python3 -m pip install jupyter && \
    python3 -m pip install jupyterlab && \
    python3 -m pip install fastapi && \
    python3 -m pip install python-multipart && \
    python3 -m pip install uvicorn

COPY ./service-defs /etc/backend.ai/service-defs
WORKDIR /usr/src/app
COPY ./deploy_server /usr/src/app

LABEL ai.backend.kernelspec="1" \
      ai.backend.envs.corecount="OPENBLAS_NUM_THREADS,OMP_NUM_THREADS,NPROC" \
      ai.backend.features="uid-match" \
      ai.backend.resource.min.cpu="2" \
      ai.backend.resource.min.mem="1G" \
      ai.backend.accelerators="cuda" \
      ai.backend.resource.min.cuda.device=1 \
      ai.backend.resource.min.cuda.shares=0.1 \
      ai.backend.base-distro="ubuntu" \
      ai.backend.service-ports="ipython:pty:3000,jupyter:http:8091,jupyterlab:http:8090,vscode:http:8180,tensorboard:http:6006,mlflow-ui:preopen:5000,nniboard:preopen:8080" \
      ai.backend.runtime-type="python" \
      ai.backend.runtime-path="python3"

# RUN python3 deploy_server.py 
"""  # noqa

DOCKERFILE_TEMPLATE_CUSTOM_LABELS = r"""
# Backend.AI specifics
LABEL ai.backend.kernelspec="1" \
      ai.backend.envs.corecount="{{ cpucount_envvars | join(',') }}" \
      ai.backend.features="{% if has_ipykernel %}query batch {% endif %}uid-match" \
      ai.backend.resource.min.cpu="{{ min_cpu }}" \
      ai.backend.resource.min.mem="{{ min_mem }}" \
      ai.backend.resource.preferred.shmem="{{ pref_shmem }}" \
      ai.backend.accelerators="{{ accelerators | join(',') }}" \
{%- if 'cuda' is in accelerators %}
      ai.backend.resource.min.cuda.device=1 \
      ai.backend.resource.min.cuda.shares=0.1 \
{%- endif %}
      ai.backend.base-distro="{{ base_distro }}" \
{%- if service_ports %}
      ai.backend.service-ports="{% for item in service_ports -%}
          {{- item['name'] }}:
          {{- item['protocol'] }}:
          {%- if (item['ports'] | length) > 1 -%}
              [{{ item['ports'] | join(',') }}]
          {%- else -%}
              {{ item['ports'][0] }}
          {%- endif -%}
          {{- ',' if not loop.last }}
      {%- endfor %}" \
{%- endif %}
      ai.backend.runtime-type="{{ runtime_type }}" \
      ai.backend.runtime-path="{{ runtime_path }}"
"""
