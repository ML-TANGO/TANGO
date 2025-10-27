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
COMPOSE_FILE_FLAG :=
CONFIG_ENV_FILE_FLAG := --env-file .env
else
COMPOSE_FILE_FLAG := -f .docker-v1compose.yml
CONFIG_ENV_FILE_FLAG :=
NEEDS_PREPARE := prepare-v1-compose
endif

SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := help

# .env 값을 make 환경으로 불러오기
-include .env
export

.PHONY: help build clean-labelling-db up down restart logs pm-logs autonn-logs config recreate \
        ps check-datasets run exec-pm exec-autonn migrate seed up-% logs-% build-% \
		prepare-v1-compose clean-v1-compose

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

# ---- v1용 파일 자동 생성
prepare-v1-compose: ## docker-compose v1용 파일 자동 생성 (deploy 제거 + runtime 주입 + env 병합)
	@echo "🛠  Generating .docker-v1compose.yml for v1 (merge env keys)..."
	@if [ ! -f docker-compose.yml ]; then echo "❌ docker-compose.yml not found!"; exit 1; fi
	@awk '\
	  # 대상 서비스 판별
	  function is_target_service(line){ return match(line,/^  (autonn|autonn_cl):/); } \
	  \
	  BEGIN{ in_svc=0; in_deploy=0; in_env=0; \
	         seen_env=0; need_env=0; \
	         found_vis=0; found_caps=0; \
	       } \
	  { \
	    # 서비스 시작
	    if (is_target_service($$0)) { \
	      in_svc=1; in_deploy=0; in_env=0; \
	      seen_env=0; need_env=0; found_vis=0; found_caps=0; \
	    } \
	    # 서비스 종료(같은 레벨의 다른 키/서비스 진입)
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
	    \
	    # 서비스 내부 처리
	    if (in_svc) { \
	      # deploy: → runtime 주입, deploy 블록은 스킵
	      if ($$0 ~ /^[[:space:]]{4}deploy:/){ \
	        print "    runtime: nvidia"; \
	        need_env=1; in_deploy=1; next; \
	      } \
	      if (in_deploy){ \
	        match($$0,/^[[:space:]]*/); \
	        if (RLENGTH <= 4){ in_deploy=0; } else { next; } \
	      } \
	      # gpus: 라인 제거
	      if ($$0 ~ /^[[:space:]]+gpus:/){ next; } \
	      \
	      # environment: 블록 진입
	      if ($$0 ~ /^[[:space:]]{4}environment:/){ \
	        in_env=1; seen_env=1; print $$0; next; \
	      } \
	      # environment: 블록 내부 처리(리스트 가정)
	      if (in_env){ \
	        # 블록 종료 감지: 4스페이스 새 키 시작
	        if ($$0 ~ /^[[:space:]]{4}[^[:space:]]/){ \
	          if (!found_vis)  print "      - NVIDIA_VISIBLE_DEVICES=$${NVIDIA_VISIBLE_DEVICES:-all}"; \
	          if (!found_caps) print "      - NVIDIA_DRIVER_CAPABILITIES=compute,utility"; \
	          in_env=0; \
	          # 이후 일반 처리로 현재 줄 출력 \
	        } else { \
	          if ($$0 ~ /^[[:space:]]{6}-[[:space:]]*NVIDIA_VISIBLE_DEVICES=/) { found_vis=1; } \
	          if ($$0 ~ /^[[:space:]]{6}-[[:space:]]*NVIDIA_DRIVER_CAPABILITIES=/) { found_caps=1; } \
	          print $$0; \
	          next; \
	        } \
	      } \
	    } \
	    \
	    # 기본 출력
	    print $$0; \
	  } \
	  END{ \
	    # 파일이 env 블록 중에 끝난 케이스 처리 \
	    if (in_env){ \
	      if (!found_vis)  print "      - NVIDIA_VISIBLE_DEVICES=$${NVIDIA_VISIBLE_DEVICES:-all}"; \
	      if (!found_caps) print "      - NVIDIA_DRIVER_CAPABILITIES=compute,utility"; \
	    } \
	    # 파일이 서비스 중에 끝났고 env가 없는데 deploy는 있었던 케이스 \
	    if (in_svc && need_env && !seen_env){ \
	      print "    environment:"; \
	      print "      - NVIDIA_VISIBLE_DEVICES=$${NVIDIA_VISIBLE_DEVICES:-all}"; \
	      print "      - NVIDIA_DRIVER_CAPABILITIES=compute,utility"; \
	    } \
	  }' docker-compose.yml > .docker-v1compose.yml
	@echo "✅ .docker-v1compose.yml 생성 완료 (deploy 제거 + runtime 주입 + env 병합)"

clean-v1-compose:
	@if [ -f .docker-v1compose.yml ]; then rm -f .docker-v1compose.yml && echo "🧹 removed .docker-v1compose.yml"; else echo "✓ no temp file"; fi

build: $(NEEDS_PREPARE) clean-labelling-db ## 이미지 빌드
	$(COMPOSE) $(COMPOSE_FILE_FLAG) build

build-%: $(NEEDS_PREPARE) ## 특정 서비스만 빌드 (예: make build-autonn)
	$(COMPOSE) $(COMPOSE_FILE_FLAG) build $*

up: $(NEEDS_PREPARE) ## 모든 서비스 시작 (-d)
	$(COMPOSE) $(COMPOSE_FILE_FLAG) up -d

down: ## 중지 및 제거
	$(COMPOSE) $(COMPOSE_FILE_FLAG) down

restart: down up ## 재시작

recreate: ## 볼륨/환경 변경 반영해 재생성(빌드는 안 함)
	$(COMPOSE) $(COMPOSE_FILE_FLAG) up -d --force-recreate

config: $(NEEDS_PREPARE) ## .env 적용된 최종 compose 확인
	$(COMPOSE) $(COMPOSE_FILE_FLAG) $(CONFIG_ENV_FILE_FLAG) config

ps: ## 컨테이너 상태 보기
	$(COMPOSE) $(COMPOSE_FILE_FLAG) ps

# ---- 로그/접속/관리
logs: ## 전체 로그 팔로우
	$(COMPOSE) $(COMPOSE_FILE_FLAG) logs -f || true

pm-logs: ## project_manager 로그
	$(COMPOSE) $(COMPOSE_FILE_FLAG) logs -f project_manager || true

autonn-logs: ## autonn 로그
	$(COMPOSE) $(COMPOSE_FILE_FLAG) logs -f autonn || true

exec-pm: ## project_manager 쉘
	$(COMPOSE) $(COMPOSE_FILE_FLAG) exec project_manager bash

exec-autonn: ## autonn 쉘
	$(COMPOSE) $(COMPOSE_FILE_FLAG) exec autonn bash

# 패턴 타겟: make up-autonn / make logs-project_manager 처럼 사용
up-%: ## 특정 서비스만 up (예: make up-autonn)
	$(COMPOSE) $(COMPOSE_FILE_FLAG) up -d $*

logs-%: ## 특정 서비스 로그 (예: make logs-project_manager)
	$(COMPOSE) $(COMPOSE_FILE_FLAG) logs -f $* || true

exec-%: ## 특정 서비스 쉘 (예: make exec-autonn)
	$(COMPOSE) $(COMPOSE_FILE_FLAG) exec $* bash

# ---- Django 보조
migrate: ## project_manager DB migrate
	$(COMPOSE) $(COMPOSE_FILE_FLAG) exec project_manager bash -lc 'python manage.py migrate'

seed: ## project_manager loaddata
	$(COMPOSE) $(COMPOSE_FILE_FLAG) exec project_manager bash -lc 'python manage.py loaddata base_model_data.json'

# ---- 안전장치
check-datasets: ## 데이터셋 디렉토리 존재/비어있음 체크
	@for d in "$$COCODIR" "$$COCO128DIR" "$$IMAGENETDIR" "$$VOCDIR"; do \
	  if [ -z "$$d" ]; then echo "❌ env 변수 미설정: $$d"; exit 1; fi; \
	  if [ ! -d "$$d" ]; then echo "❌ 디렉토리 없음: $$d"; exit 1; fi; \
	  if [ -z "$$(ls -A "$$d" 2>/dev/null)" ]; then echo "❌ 비어있음: $$d"; exit 1; fi; \
	done; \
	echo "✅ datasets OK"

