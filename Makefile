S3_PATH = s3://2022-msia423-zhu-simon/data/raw/anime_songs.csv

# all: image run
.PHONY: all image-model s3-upload cleaned features
all: data/intermediate/cleaned.csv data/intermediate/features.csv

image-model:
	docker build -f dockerfiles/Dockerfile -t final-project .

# data acquisition
s3-upload: data/raw/anime_songs.csv
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
	--env-file config/local/config final-project run.py acquire --file_output=${S3_PATH} --input=$<

# model pipeline starts here
data/intermediate/cleaned.csv: config/model.yaml
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
	--env-file config/local/config final-project run.py clean --input=${S3_PATH} \
	--file_output=data/intermediate/cleaned.csv --config=$< \
	--mid_output=data/raw/downloaded.csv

cleaned: data/intermediate/cleaned.csv

data/intermediate/features.csv: data/intermediate/cleaned.csv config/model.yaml
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
	--env-file config/local/config final-project run.py featurize --file_output=$@ --input=$< --config=config/model.yaml

features: data/intermediate/features.csv

models/res_plots.png models/kmeans.joblib &: data/intermediate/features.csv config/model.yaml
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
	--env-file config/local/config final-project run.py train --file_output=$@ --model_output=models/kmeans.joblib \
	--input=$< --config=config/model.yaml

models: models/res_plots.png models/kmeans.joblib

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

clean-files:
	rm -f data/intermediate/cleaned.csv
	rm -f data/intermediate/features.csv
	rm -f data/raw/downloaded.csv
	rm -f models/*.png
	rm -f models/*.joblib