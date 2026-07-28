# Makefile — Meteo Liora (MLOps : prédiction de pluie en Australie)
# Stack : MLflow (tracking/registry) + trainer + API FastAPI + Streamlit + Airflow
COMPOSE := docker compose -f App/docker-compose.yml
VENV    := .venv/bin
# Profils = tous les services optionnels, pour un down/clean exhaustif
PROFILES := --profile train --profile orchestration

.DEFAULT_GOAL := help

.PHONY: help \
        build up down restart logs ps status \
        train reload predict test-api health \
        install test lint fmt ci \
        airflow-up airflow-down airflow-logs \
        clean stop-all \
        train-remote deploy-model

## ————— Aide —————
help:            
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

## ————— Stack Docker —————
build:           
	$(COMPOSE) build

up:             
	$(COMPOSE) up mlflow api streamlit

down:           
	$(COMPOSE) $(PROFILES) down

stop-all: down   

restart:        
	$(MAKE) down
	$(MAKE) up

logs:            
	$(COMPOSE) logs -f --tail=100

ps: status       
	$(COMPOSE) ps

## ————— ML : entraînement & inférence —————
train:           
	$(COMPOSE) $(PROFILES) run --rm trainer
	-$(MAKE) reload

reload:         
	@$(COMPOSE) exec -T api python -c "import urllib.request as u; print(u.urlopen(u.Request('http://localhost:8000/reload', method='POST'), timeout=60).read().decode())"

health:         
	@$(COMPOSE) exec -T api python -c "import urllib.request as u; print(u.urlopen('http://localhost:8000/health', timeout=10).read().decode())"

test-api: predict  
predict:        
	@$(COMPOSE) exec -T api python -c "import urllib.request as u, json; p=json.dumps({'Location':'Sydney','Month':7,'RainToday':'Yes','Humidity3pm':80,'Sunshine':3.5,'Pressure3pm':1008,'Rainfall':12,'WindGustSpeed':56,'Cloud3pm':8,'Temp3pm':16}).encode(); r=u.Request('http://localhost:8000/predict', data=p, headers={'Content-Type':'application/json'}, method='POST'); print(u.urlopen(r, timeout=15).read().decode())"

## ————— Qualité (local, via .venv) —————
install:         
	$(VENV)/pip install -r App/requirements/test.txt

lint:         
	cd App && ../$(VENV)/ruff check src/

fmt:          
	cd App && ../$(VENV)/ruff check --fix src/

test:            ## lance les tests unitaires (pytest)
	cd App && ../$(VENV)/pytest src/tests/ -v --tb=short

ci: lint test  

## ————— Orchestration Airflow —————
airflow-up:     
	$(COMPOSE) --profile orchestration up -d airflow-db airflow

airflow-down:   
	$(COMPOSE) --profile orchestration stop airflow airflow-db

airflow-logs:    
	$(COMPOSE) logs -f --tail=100 airflow

## ————— Nettoyage —————
clean:          
	$(COMPOSE) $(PROFILES) down -v

## ————— Déploiement serveur (piloté depuis ce PC → VM 192.168.1.34) —————
SRV         := 192.168.1.36
SRV_MLFLOW  := http://$(SRV):5000
SRV_API     := http://$(SRV):8000
SRV_AIRFLOW := http://$(SRV):8080
DEPLOY_DAG  := rain_australia_deploy_pipeline

train-remote:    ## entraîne via le conteneur trainer local, pousse le modèle au registry MLflow du serveur
	$(COMPOSE) run --rm --no-deps -e MLFLOW_TRACKING_URI=$(SRV_MLFLOW) trainer --model rf --register

deploy-model: train-remote  ## train-remote puis déclenche le DAG distant (fallback : /reload direct)
	@echo "→ Déclenchement du DAG $(DEPLOY_DAG) sur $(SRV_AIRFLOW)…"
	@set -a; . App/.env.prod; set +a; \
	  curl -fsS -u "admin:$$AIRFLOW_ADMIN_PASSWORD" -H 'Content-Type: application/json' \
	    -X POST "$(SRV_AIRFLOW)/api/v1/dags/$(DEPLOY_DAG)/dagRuns" -d '{"conf":{}}' \
	  && echo "  ✅ DAG déclenché." \
	  || (echo "  ⚠️  DAG KO → fallback /reload direct"; curl -fsS -X POST "$(SRV_API)/reload"; echo)
