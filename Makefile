
.PHONY: all image-model s3-upload cleaned features models scores rds-create rds-ingest clean-docker clean-files
all: data/intermediate/cleaned.csv data/intermediate/features.csv models/res_plots.png models/kmeans.joblib data/final/clusters.csv

image-model:
	docker build -f dockerfiles/Dockerfile -t final-project .

# data acquisition
s3-upload: data/raw/anime_songs.csv
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
	--env-file config/local/config final-project run.py acquire --input=$<

# model pipeline starts here
data/intermediate/cleaned.csv: config/model.yaml
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
	--env-file config/local/config final-project run.py clean \
	--file_output=data/intermediate/cleaned.csv --config=$< \
	--mid_output=data/raw/downloaded.csv

cleaned: data/intermediate/cleaned.csv

data/intermediate/features.csv: data/intermediate/cleaned.csv config/model.yaml
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
	--env-file config/local/config final-project run.py featurize --file_output=$@ --input=$< --config=config/model.yaml

features: data/intermediate/features.csv

data/final/anime_clusters.csv models/kmeans.joblib &: data/intermediate/features.csv config/model.yaml
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
	--env-file config/local/config final-project run.py train --file_output=data/final/anime_clusters.csv \
	--model_output=models/kmeans.joblib \
	--input=$< --config=config/model.yaml

models: data/final/anime_clusters.csv models/kmeans.joblib

data/final/sample_clusters.csv: data/sample/sample_search_songs.csv models/kmeans.joblib
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
	--env-file config/local/config final-project run.py score --file_output=$@ --model=models/kmeans.joblib \
	--input=$<

scores: data/final/sample_clusters.csv

data/final/sample_eval.txt: data/final/sample_clusters.csv
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
	--env-file config/local/config final-project run.py evaluate --file_output=$@ \
	--input=$<

# ingest data to RDS instance
rds-create:
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
	--env-file config/local/config final-project run_rds.py create

rds-ingest: data/final/anime_clusters.csv
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
	--env-file config/local/config final-project run_rds.py add_data \
	--data_path=$<

# web app deployment
image-app:
	docker build -f dockerfiles/Dockerfile.app -t final-project-app .

app: config/flaskconfig.py
	docker run --env-file config/local/config -p 5000:5000 final-project-app

# clean up all docker images and containers
clean-docker:
	docker image rm -f final-project
	docker image rm -f final-project-app
	docker container prune

# clean up all intermediary artifacts
clean-files:
	rm -f data/intermediate/cleaned.csv
	rm -f data/intermediate/features.csv
	rm -f data/raw/downloaded.csv
	rm -f data/final/clusters.csv
	rm -f models/*.png
	rm -f models/*.joblib