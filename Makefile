# ============================================
# Makefile (Unified, Auto by default)
# - Compose v1/v2 자동 감지
# - v1 전용 변환( deploy 제거 + runtime 주입 + env 병합 )
# - build/up/logs 등 모든 기본 타깃이 데이터셋 자동판단 override를 기본 포함
# - 필요할 때만 외부 데이터셋 바인딩(*vol_... 앵커, .env 이용)
# ============================================

# ---- Compose v1/v2 자동 감지
ifndef COMPOSE
COMPOSE := $(shell \
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then \
    echo "docker compose"; \
  elif command -v docker-compose >/dev/null 2>&1; then \
    echo "docker-compose"; \
  else \
    echo ""; \
  fi)
endif

ifeq ($(strip $(COMPOSE)),)
$(error ❌ Neither 'docker compose' nor 'docker-compose' found. Please install Docker Compose.)
endif

# v2 전용 플래그 (v1은 해당 플래그 미지원)
ifeq ($(COMPOSE),docker compose)
COMPOSE_FILE_FLAG := -f docker-compose.yml
CONFIG_ENV_FILE_FLAG := --env-file .env
else
COMPOSE_FILE_FLAG := -f .compose/docker-compose.v1.yml
CONFIG_ENV_FILE_FLAG :=
NEEDS_PREPARE := prepare-v1-compose
endif

SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := help

# .env 값을 make 환경으로 불러오기 (있으면)
-include .env
export

# --------------------------------------------
# 공통 변수
# --------------------------------------------
DATASETS_OVERRIDE := .compose/docker-compose.datasets.yml
COMPOSE_PROJECT_NAME ?= $(shell basename "$$(pwd)" | tr '[:upper:]' '[:lower:]')

# --------------------------------------------
# PHONY
# --------------------------------------------
.PHONY: help run build build-project_manager build-autonn build-autonn_cl build-labelling \
		up up-project_manager up-autonn up-autonn_cl up-% down restart recreate config ps \
		logs logs-pm logs-% exec-pm exec-% migrate seed prepare-v1-compose clean-v1-compose \
        gen-datasets-override validate-host-datasets clean-labelling-db \
		ensure-nvidia-runtime show-docker-runtime

# --------------------------------------------
# 도움말
# --------------------------------------------
help: ## 사용 가능한 명령 목록
	@grep -hE '^[a-zA-Z0-9_%-]+:.*?## ' $(MAKEFILE_LIST) | \
	awk 'BEGIN{FS=":.*## "}; {printf"  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------
# 원샷 워크플로우
# --------------------------------------------
run: build up logs ## (자동) 빌드 → 실행 → 로그 팔로우

# --------------------------------------------
# 기본 빌드/실행 계열 (자동으로 override 병합)
# --------------------------------------------
build: ensure-nvidia-runtime clean-labelling-db $(NEEDS_PREPARE) gen-datasets-override validate-host-datasets ## 전체 이미지 빌드(필요한 데이터셋 외부 바인딩 포함)
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) build

build-labelling: ensure-nvidia-runtime clean-labelling-db $(NEEDS_PREPARE)
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) build labelling

build-project_manager: ensure-nvidia-runtime $(NEEDS_PREPARE) gen-datasets-override validate-host-datasets
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) build project_manager

build-autonn: ensure-nvidia-runtime $(NEEDS_PREPARE) gen-datasets-override validate-host-datasets
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) build autonn

build-autonn_cl: ensure-nvidia-runtime $(NEEDS_PREPARE) gen-datasets-override validate-host-datasets
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) build autonn_cl

build-%: ensure-nvidia-runtime $(NEEDS_PREPARE) ## 특정 이미지만 빌드 (예: make build-code_gen)
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) build $*

up: ensure-nvidia-runtime $(NEEDS_PREPARE) gen-datasets-override validate-host-datasets ## 모든 서비스 시작 (-d, 자동 override 포함)
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) up -d

up-project_manager: ensure-nvidia-runtime $(NEEDS_PREPARE) gen-datasets-override validate-host-datasets
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) up project_manager -d

up-autonn: ensure-nvidia-runtime $(NEEDS_PREPARE) gen-datasets-override validate-host-datasets
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) up autonn -d

up-autonn_cl: ensure-nvidia-runtime $(NEEDS_PREPARE) gen-datasets-override validate-host-datasets
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) up autonn_cl -d

up-%: $(NEEDS_PREPARE) ## 특정 서비스만 시작 (-d, 예: make up-autonn)
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) up -d $*

down: ## 중지 및 제거
	$(COMPOSE) $(COMPOSE_FILE_FLAG) down
	-$(COMPOSE) $(COMPOSE_FILE_FLAG) -f $(DATASETS_OVERRIDE) down

restart: down up ## 재시작

recreate: ensure-nvidia-runtime $(NEEDS_PREPARE) gen-datasets-override validate-host-datasets ## 볼륨/환경 변경 반영해 재생성(빌드X)
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) up -d --force-recreate

config: $(NEEDS_PREPARE) gen-datasets-override ## .env 적용된 최종 compose 확인
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(CONFIG_ENV_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) config

ps: $(NEEDS_PREPARE) ## 컨테이너 상태 보기
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) ps

logs: $(NEEDS_PREPARE) ## 전체 로그 팔로우
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) logs -f || true

logs-%: $(NEEDS_PREPARE) ## 특정 서비스 로그 (예: make logs-autonn_cl)
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) logs -f $* || true

exec-%: ensure-nvidia-runtime ## 특정 서비스 쉘 (예: make exec-autonn)
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) exec $* bash

logs-pm: ensure-nvidia-runtime $(NEEDS_PREPARE) ## project_manager 로그
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) logs -f project_manager || true

exec-pm: ensure-nvidia-runtime ## project_manager 쉘
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) exec project_manager bash

# --------------------------------------------
# Django 보조
# --------------------------------------------
migrate: ## project_manager DB migrate
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) exec project_manager bash -lc 'python manage.py migrate'

seed: ## project_manager loaddata
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(_RUNTIME_DATASETS_FLAG) exec project_manager bash -lc 'python manage.py loaddata base_model_data.json'

# --------------------------------------------
# NVIDIA Docker runtime 보장
# --------------------------------------------
ensure-nvidia-runtime:
	@set -e; \
	if [ "$${SKIP_NVIDIA_RUNTIME_CHECK:-0}" = "1" ]; then \
	  echo "⏭  skip ensure-nvidia-runtime (SKIP_NVIDIA_RUNTIME_CHECK=1)"; exit 0; \
	fi; \
	if docker info 2>/dev/null | grep -iq 'Runtimes:.*nvidia'; then \
	  echo "✓ Docker runtime 'nvidia' already registered"; \
	else \
	  if command -v nvidia-ctk >/dev/null 2>&1; then \
	    echo "→ Registering NVIDIA runtime via nvidia-ctk ..."; \
	    sudo nvidia-ctk runtime configure --runtime=docker; \
	    echo "→ Restarting docker ..."; \
	    sudo systemctl restart docker; \
	    if docker info 2>/dev/null | grep -iq 'Runtimes:.*nvidia'; then \
	      echo "✅ Docker runtime 'nvidia' registered"; \
	    else \
	      echo "❌ Failed to register 'nvidia' runtime. Check docker logs: 'journalctl -u docker -n 200'"; \
	      exit 1; \
	    fi; \
	  else \
	    echo "❌ 'nvidia-ctk' not found (nvidia-container-toolkit 미설치)."; \
	    echo "   설치 후 다시 시도: sudo apt install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"; \
	    exit 1; \
	  fi; \
	fi

show-docker-runtime:
	@docker info 2>/dev/null | grep -i runtime || true
	@echo "daemon.json:"; cat /etc/docker/daemon.json 2>/dev/null || echo "(no /etc/docker/daemon.json)"

# --------------------------------------------
# docker-compose v1용 파일 자동 생성
# - deploy 제거, runtime: nvidia 주입, env 병합
# - 대상 서비스: autonn | autonn_cl
# --------------------------------------------
prepare-v1-compose:
	@echo "🛠  Generating .compose/docker-compose.v1.yml for v1 (merge env keys)..."
	@if [ ! -f docker-compose.yml ]; then echo "❌ docker-compose.yml not found!"; exit 1; fi
	@if [ ! -d .compose ]; then mkdir -p .compose; fi
	@awk '\
	  function is_target_service(line){ return match(line,/^  (autonn|autonn_cl):/); } \
	  BEGIN{ in_svc=0; in_deploy=0; in_env=0; seen_env=0; need_env=0; found_vis=0; found_caps=0; } \
	  { \
	    if (is_target_service($$0)) { in_svc=1; in_deploy=0; in_env=0; seen_env=0; need_env=0; found_vis=0; found_caps=0; } \
	    else if (in_svc && $$0 ~ /^  [^[:space:]].*:/ && !is_target_service($$0)) { \
	      if (in_env){ \
	        if (!found_vis)  print "      - NVIDIA_VISIBLE_DEVICES=$${NVIDIA_VISIBLE_DEVICES:-all}"; \
	        if (!found_caps) print "      - NVIDIA_DRIVER_CAPABILITIES=compute,utility"; \
	        in_env=0; \
	      } \
	      if (need_env && !seen_env){ \
	        print "    environment:"; \
	        print "      - NVIDIA_VISIBLE_DEVICES=$${NVIDIA_VISIBLE_DEVICES:-all}"; \
	        print "      - NVIDIA_DRIVER_CAPABILITIES=compute,utility"; \
	      } \
	      in_svc=0; in_deploy=0; \
	    } \
	    if (in_svc) { \
	      if ($$0 ~ /^[[:space:]]{4}deploy:/){ print "    runtime: nvidia"; need_env=1; in_deploy=1; next; } \
	      if (in_deploy){ match($$0,/^[[:space:]]*/); if (RLENGTH <= 4){ in_deploy=0; } else { next; } } \
	      if ($$0 ~ /^[[:space:]]+gpus:/){ next; } \
	      if ($$0 ~ /^[[:space:]]{4}environment:/){ in_env=1; seen_env=1; print $$0; next; } \
	      if (in_env){ \
	        if ($$0 ~ /^[[:space:]]{4}[^[:space:]]/){ \
	          if (!found_vis)  print "      - NVIDIA_VISIBLE_DEVICES=$${NVIDIA_VISIBLE_DEVICES:-all}"; \
	          if (!found_caps) print "      - NVIDIA_DRIVER_CAPABILITIES=compute,utility"; \
	          in_env=0; \
	        } else { \
	          if ($$0 ~ /^[[:space:]]{6}-[[:space:]]*NVIDIA_VISIBLE_DEVICES=/) { found_vis=1; } \
	          if ($$0 ~ /^[[:space:]]{6}-[[:space:]]*NVIDIA_DRIVER_CAPABILITIES=/) { found_caps=1; } \
	          print $$0; next; \
	        } \
	      } \
	    } \
	    print $$0; \
	  } \
	  END{ \
	    if (in_env){ \
	      if (!found_vis)  print "      - NVIDIA_VISIBLE_DEVICES=$${NVIDIA_VISIBLE_DEVICES:-all}"; \
	      if (!found_caps) print "      - NVIDIA_DRIVER_CAPABILITIES=compute,utility"; \
	    } \
	    if (in_svc && need_env && !seen_env){ \
	      print "    environment:"; \
	      print "      - NVIDIA_VISIBLE_DEVICES=$${NVIDIA_VISIBLE_DEVICES:-all}"; \
	      print "      - NVIDIA_DRIVER_CAPABILITIES=compute,utility"; \
	    } \
	  }' docker-compose.yml > .compose/docker-compose.v1.yml
	@echo "✅ .compose/docker-compose.v1.yml 생성 완료 (deploy 제거 + runtime 주입 + env 병합)"

clean-v1-compose:
	@if [ -f .compose/docker-compose.v1.yml ]; then rm -f .compose/docker-compose.v1.yml && echo "🧹 removed .compose/docker-compose.v1.yml"; else echo "✓ no temp file"; fi

# --------------------------------------------
# 데이터셋 자동판단 override 생성
# - check_shared_datasets.sh 가 $(DATASETS_OVERRIDE) 생성
# - 비어있는 /shared/datasets/* 만 *vol_* 앵커로 외부 바인딩
# --------------------------------------------
gen-datasets-override:
	@echo ">> Generating dataset override based on $(COMPOSE_PROJECT_NAME)_shared ..."
	@mkdir -p .compose
	@res="$$(scripts/check_shared_datasets.sh '$(COMPOSE_PROJECT_NAME)' '$(DATASETS_OVERRIDE)')"; \
	echo ">> $$res"

# override 파일이 바인딩을 포함하는지 판별(있으면 -f 추가)
define _RUNTIME_DATASETS_FLAG
$(shell if grep -q '^services:' '$(DATASETS_OVERRIDE)' 2>/dev/null && grep -q 'volumes:' '$(DATASETS_OVERRIDE)' 2>/dev/null; then echo "-f $(DATASETS_OVERRIDE)"; fi)
endef

# --------------------------------------------
# 외부 경로 유효성 검증: 실제로 바인딩될 항목만 검사
# - override에 포함된 데이터셋만 .env 경로/존재 확인
# --------------------------------------------
validate-host-datasets: gen-datasets-override
	@set -e; \
	has_err=0; \
	check_one() { \
	  local name="$$1"; local envv="$$2"; local val="$${!2}"; \
	  if grep -q "\*vol_$${name}" '$(DATASETS_OVERRIDE)' 2>/dev/null; then \
	    if [ -z "$$val" ]; then echo "⚠️  $$envv not set (skipped)"; \
	    elif [ ! -d "$$val" ]; then echo "⚠️  $$envv dir not found: $$val (skipped)"; \
	    elif [ -z "$$(ls -A "$$val" 2>/dev/null)" ]; then echo "⚠️  $$envv is empty: $$val (skipped)";  \
	    else echo "✓ $$envv OK → $$val"; fi; \
	  fi; \
	}; \
	check_one coco COCODIR; \
	check_one coco128 COCO128DIR; \
	check_one coco128seg COCO128SEGDIR; \
	check_one imagenet IMAGENETDIR; \
	check_one voc VOCDIR; \
	check_one chestxray CHESTXRAYDIR; \
	echo "✅ host dataset paths OK (for the ones that will be bound)"

# --------------------------------------------
# labelling/datadb 폴더가 있으면 삭제
# --------------------------------------------
clean-labelling-db: 
	@if [ -d labelling/datadb ]; then \
		echo "🧹 removing labelling/datadb"; \
		sudo rm -rf -- labelling/datadb; \
	else \
		echo "✓ labelling/datadb 없음 — skip"; \
	fi