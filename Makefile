COMPOSE := docker compose -f App/docker-compose.yml

.PHONY: build up train down logs clean test-api

build:        ## build toutes les images
	$(COMPOSE) build

up:           ## démarre mlflow + api + streamlit
	$(COMPOSE) up -d mlflow api streamlit

train:        ## lance l'entraînement (one-shot) puis recharge l'API
	$(COMPOSE) --profile train run --rm trainer
	-curl -s -X POST http://localhost:8000/reload && echo

down:         ## stoppe tout
	$(COMPOSE) --profile train down

logs:
	$(COMPOSE) logs -f --tail=100

clean:        ## stoppe + supprime volumes (PERD le modèle/MLflow)
	$(COMPOSE) --profile train down -v

test-api:     ## prédiction d'exemple
	curl -s -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \
	  -d '{"Location":"Sydney","Month":7,"RainToday":"Yes","Humidity3pm":80,"Sunshine":3.5,"Pressure3pm":1008,"Rainfall":12,"WindGustSpeed":56,"Cloud3pm":8,"Temp3pm":16}' ; echo
