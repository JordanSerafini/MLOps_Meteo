# Makefile — Meteo Liora (MLOps : prédiction de pluie en Australie)
# Stack : MLflow (tracking/registry) + trainer + API FastAPI + Streamlit + Airflow
COMPOSE := docker compose -f App/docker-compose.yml
VENV    := .venv/bin
# Profils = tous les services optionnels, pour un down/clean exhaustif
PROFILES := --profile train --profile orchestration --profile monitoring --profile drift

.DEFAULT_GOAL := help

.PHONY: help \
        build up down restart logs ps status \
        train reload predict test-api health \
        secrets install test lint fmt ci \
        airflow-up airflow-down airflow-logs \
        monitoring-up monitoring-down \
        drift drift-demo trafic \
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

# Récupère un jeton pour le compte passé en $(1) : `$(call jeton,ADMIN)`.
# L'API est bindée sur la boucle locale, on l'appelle donc depuis l'hôte.
API_LOCAL := http://localhost:8000
define jeton
set -a; . App/.env; set +a; \
curl -fsS -X POST $(API_LOCAL)/token \
     -d "username=$$API_$(1)_USER&password=$$API_$(1)_PASSWORD" \
  | $(VENV)/python -c 'import sys,json; print(json.load(sys.stdin)["access_token"])'
endef

reload:          ## recharge le modèle @champion (compte admin)
	@T=$$($(call jeton,ADMIN)); \
	 curl -fsS -X POST $(API_LOCAL)/reload -H "Authorization: Bearer $$T"; echo

health:          ## état de l'API (endpoint ouvert)
	@curl -fsS $(API_LOCAL)/health; echo

test-api: predict
predict:         ## une prédiction de démonstration (compte client)
	@T=$$($(call jeton,CLIENT)); \
	 curl -fsS -X POST $(API_LOCAL)/predict -H "Authorization: Bearer $$T" \
	      -H 'Content-Type: application/json' \
	      -d '{"Location":"Sydney","Month":7,"RainToday":"Yes","Humidity3pm":80,"Sunshine":3.5,"Pressure3pm":1008,"Rainfall":12,"WindGustSpeed":56,"Cloud3pm":8,"Temp3pm":16}'; echo

## ————— Secrets —————
secrets:         ## génère App/.env et App/.env.api (FORCE=1 pour écraser)
	$(VENV)/python App/scripts/generer_secrets.py $${FORCE:+--force}

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

## ————— Monitoring (Prometheus + Grafana + Pushgateway) —————
monitoring-up:   ## démarre Prometheus (:9090), Grafana (:3000) et le pushgateway
	$(COMPOSE) --profile monitoring up -d prometheus grafana pushgateway
	@echo "  Grafana    http://localhost:3000  (admin / $${GRAFANA_ADMIN_PASSWORD:-admin})"
	@echo "  Prometheus http://localhost:9090"

monitoring-down:
	$(COMPOSE) --profile monitoring stop prometheus grafana pushgateway

## ————— Détection de dérive —————
trafic:          ## envoie N prédictions à l'API (N=100 par défaut, STATION= pour cibler)
	$(VENV)/python App/scripts/simuler_trafic.py --n $${N:-100} $${STATION:+--station $$STATION}

drift:           ## compare le trafic reçu aux données d'entraînement
	$(COMPOSE) --profile drift run --rm drift

drift-demo:      ## démo : injecte du trafic décalé (Portland) puis mesure la dérive
	@echo "→ 500 prédictions depuis Portland (37 % de jours pluvieux contre 22 % en moyenne)…"
	@echo "  (500 = au-dessus du minimum de 400, cf. Docs/CALIBRATION_DRIFT.md)"
	$(VENV)/python App/scripts/simuler_trafic.py --n 500 --station Portland
	@echo "→ mesure de la dérive…"
	$(MAKE) drift

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
