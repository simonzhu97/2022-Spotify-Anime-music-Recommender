
.PHONY: model-all everything image-model s3-upload cleaned features models scores evaluate rds-create rds-ingest 
.PHONY: image-app app image-test tests clean-containers clean-images clean-files clean-all
# to run the model pipeline only
model-all: cleaned features models scores evaluate
# to run everything from data acquisition to rds creation
everything: s3-upload model-all rds-create rds-ingest app

image-model:
	docker build -f dockerfiles/Dockerfile -t final-project .

################################# data acquisition #######################################
s3-upload: data/raw/anime_songs.csv
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
	-e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e S3_BUCKET \
	final-project run.py acquire --input=$<

################################# model pipeline starts here #############################
# clean & preprocess
data/intermediate/cleaned.csv: config/model.yaml
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
	-e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e S3_BUCKET \
	final-project run.py clean \
	--file_output=data/intermediate/cleaned.csv --config=$< \
	--mid_output=data/raw/downloaded.csv

cleaned: data/intermediate/cleaned.csv

# generate features
data/intermediate/features.csv models/scalar.joblib &: data/intermediate/cleaned.csv config/model.yaml
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
	final-project run.py featurize --file_output=data/intermediate/features.csv \
	--input=$< --config=config/model.yaml \
	--model_output=models/scalar.joblib

features: data/intermediate/features.csv models/scalar.joblib

# train model
data/final/anime_clusters.csv models/kmeans.joblib &: data/intermediate/features.csv data/intermediate/cleaned.csv config/model.yaml
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
	final-project run.py train --file_output=data/final/anime_clusters.csv \
	--model_output=models/kmeans.joblib \
	--input=$< --config=config/model.yaml --origin_data=data/intermediate/cleaned.csv

models: data/final/anime_clusters.csv models/kmeans.joblib

# score model
models/sample_clusters.csv: data/sample/sample_search_songs.csv models/kmeans.joblib models/scalar.joblib
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
	final-project run.py score --file_output=$@ --model=models/kmeans.joblib \
	--input=$< --scalar=models/scalar.joblib

scores: models/sample_clusters.csv

# evaluate performance
models/sample_eval.txt: models/sample_clusters.csv
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
	final-project run.py evaluate --file_output=$@ \
	--input=$<

evaluate: models/sample_eval.txt

################################# relational data ingestion #############################
rds-create:
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
	-e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e SQLALCHEMY_DATABASE_URI \
	final-project run_rds.py create

rds-ingest: data/final/anime_clusters.csv
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
	-e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e SQLALCHEMY_DATABASE_URI \
	final-project run_rds.py add_data \
	--data_path=$<

################################# web app ##############################################
# web app deployment
image-app:
	docker build -f dockerfiles/Dockerfile.app -t final-project-app .

app: config/flaskconfig.py
	docker run \
	-e SPOTIPY_CLIENT_ID -e SPOTIPY_CLIENT_SECRET -e SQLALCHEMY_DATABASE_URI \
	-p 5000:5000 final-project-app

################################# unit tests ##############################################
# unit tests
image-test:
	docker build -f dockerfiles/Dockerfile.test -t final-project-tests .

tests:
	docker run final-project-tests

################################# utilities ##############################################
# clean up all docker images and containers
clean-containers:
	docker container prune

clean-images:
	docker image rm -f final-project
	docker image rm -f final-project-app
	docker image rm -f final-project-tests

# clean up all intermediary artifacts
clean-files:
	rm -f data/intermediate/*.csv
	rm -f data/raw/downloaded.csv
	rm -f data/final/*.csv
	rm -f models/*.png
	rm -f models/*.joblib
	rm -f models/*.txt

# clean all
clean-all: clean-containers clean-images clean-files