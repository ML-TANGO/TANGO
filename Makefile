# Makefile

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
CONFIG_ENV_FILE_FLAG := --env-file .env
else
CONFIG_ENV_FILE_FLAG :=
endif

SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := help

# .env 값을 make 환경으로 불러오기 (없어도 동작은 함)
-include .env
export

.PHONY: help build clean-labelling-db up down restart logs pm-logs autonn-logs config recreate \
        check-datasets run exec-pm exec-autonn migrate seed up-% logs-%

help: ## 사용 가능한 명령 목록
	@grep -hE '^[a-zA-Z0-9_%-]+:.*?## ' $(MAKEFILE_LIST) | \
	awk 'BEGIN{FS=":.*## "}; {printf"  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---- 기본 워크플로우
run: check-datasets build up logs ## 데이터셋 확인→ 빌드→ 실행→ 로그 팔로우

clean-labelling-db: ## (자동) labelling/datadb 폴더가 있으면 삭제
	@if [ -d labelling/datadb ]; then \
		echo "🧹 removing labelling/datadb"; \
		sudo rm -rf -- labelling/datadb; \
	else \
		echo "✓ labelling/datadb 없음 — skip"; \
	fi

build: clean-labelling-db ## 이미지 빌드
	$(COMPOSE) build

up: ## 모든 서비스 시작 (-d)
	$(COMPOSE) up -d

down: ## 중지 및 제거
	$(COMPOSE) down

restart: down up ## 재시작

recreate: ## 볼륨/환경 변경 반영해 재생성(빌드는 안 함)
	$(COMPOSE) up -d --force-recreate

config: ## .env 적용된 최종 compose 확인
	$(COMPOSE) $(CONFIG_ENV_FILE_FLAG) .env config

# ---- 로그/접속/관리
logs: ## 전체 로그 팔로우
	$(COMPOSE) logs -f

pm-logs: ## project_manager 로그
	$(COMPOSE) logs -f project_manager

autonn-logs: ## autonn 로그
	$(COMPOSE) logs -f autonn

exec-pm: ## project_manager 쉘
	$(COMPOSE) exec project_manager bash

exec-autonn: ## autonn 쉘
	$(COMPOSE) exec autonn bash

# 패턴 타겟: make up-autonn / make logs-project_manager 처럼 사용
up-%: ## 특정 서비스만 up (예: make up-autonn)
	$(COMPOSE) up -d $*

logs-%: ## 특정 서비스 로그 (예: make logs-project_manager)
	$(COMPOSE) logs -f $*

# ---- Django 보조
migrate: ## project_manager DB migrate
	$(COMPOSE) exec project_manager bash -lc 'python manage.py migrate'

seed: ## project_manager loaddata
	$(COMPOSE) exec project_manager bash -lc 'python manage.py loaddata base_model_data.json'

# ---- 안전장치
check-datasets: ## 데이터셋 디렉토리 존재/비어있음 체크
	@for d in "$$COCODIR" "$$COCO128DIR" "$$IMAGENETDIR" "$$VOCDIR"; do \
	  if [ -z "$$d" ]; then echo "❌ env 변수 미설정: $$d"; exit 1; fi; \
	  if [ ! -d "$$d" ]; then echo "❌ 디렉토리 없음: $$d"; exit 1; fi; \
	  if [ -z "$$(ls -A "$$d" 2>/dev/null)" ]; then echo "❌ 비어있음: $$d"; exit 1; fi; \
	done; \
	echo "✅ datasets OK"

