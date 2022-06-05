# MSiA423 Anime Music Recommender
Author: Simon Zhu

# Table of Contents
* [Project Charter](#Project-Charter)
* [Directory Structure ](#Directory-structure)
* [Running the Pipeline ](#running-the-whole-pipeline-aside-from-unit-tests-and-web-app)
	* [0. Build image](#0-build-docker-image-for-the-pipeline)
	* [1. Data Acquisition](#1-data-acquisition)
	* [2. Model Pipeline](#2-Model-Pipeline)
	* [3. Relational Data Ingestion](#3-Relational-Data-Ingestion)
* [Running the web app](#running-the-web-app)
* [Testing](#Testing)
* [Other Utilities](#Other-Utilies)

# Project Charter
## Background
Anime is hand-drawn and computer animation originating from Japan. In Japan, anime describes all animated works, regardless of style or origin. However, outside of Japan, anime usually refers specifically to animation produced in Japan. 

Since the 1980s, this particular medium has witnessed great international success with the rise of subtitles and increasing distribution through streaming services. In fact, streaming services such as Netflix have invested quite some amount in this genre. In 2019, the annual overseas exports of Japanese animation exceeded $10 billion.

<img src="https://github.com/MSIA/2022-msia423-Zhu-Simon-project/blob/main/figures/aot.jpeg" alt="drawing" height="300" width="500"/>

## Vision
Despite rapid growth of overseas exports, stigmas about anime still linger. Take anime music as an example. A vast majority still considers anime music as cute music that might not have much meaning. Nevertheless, in the recent years, one can find traces of jazz, rock and many other elements in anime music. By exposing people to anime music that is similar to other widely-listened music, the project could bring awareness to the diversity of anime music and therefore even piques interests in the anime industry itself. Thus, the target audience of the application is those who are explorative and want to identify other sources of music that fit their taste.

## Mission
The project aims to provide a music recommender application where the app would recommend anime songs similar to the one the user inputs. The recommended songs would be pulled from a [dataset](https://www.kaggle.com/datasets/simonzhu97/popular-spotify-anime-songs) that includes recent hit anime songs on Spotify. 

The application will have a pretrained clustering model where anime songs in the list are grouped into several clusters. In a text input field, the user will input a song and the artist name they like and the app will search for this song on Spotify, extract the corresponding song features, compare them to the centroids of clusters of songs yielded by the pretrained model, find the closest cluster, and recommend the top similar anime songs in that cluster based on a cosine similarity measure. For example, if the user inputs __"City of Stars"__, the app might output the following table, where each row represents a recommended song. 

|      | song name  | genre | link                                     |
| ---: | :--------- | :---- | :--------------------------------------- |
|    1 | One Last Kiss | jazz | https://open.spotify.com/track/5RhWszHMSKzb7KiXk4Ae0M?si=88e08a24b2fb4c63 |
|    ... | ...  | ... | ... |

The table above could be altered as the project proceeds. For instance, other song features such as release dates and which anime it comes from could also be included in the table if time permits.

## Success Criteria

### Model performance metric
A sample test dataset would be created where five random pop songs would be chosen. Then, for each of these five songs, song features are offered via Spotify's API. An example test dataset is as follows.

|      | title  | danceability |energy|...|time_signature|
| ---: | :--------- | :---- |:---|:---|:---|
|    1 | One Last Kiss |0.9| 0.3|...|4|
|    2 | Lost in Paradise | 0.8| 0.1|...|3|
|    3 | Black Catcher | 0.1| 0.2|...|4|
|    ... | ... | ...| ...|...|...|
|    439 | Blue Sky |0.5| 0.8|...|2|

Prior to deployment, the recommendation system would be tested on this dataset. The KMeans model yielded from training will be applied here to assign each song to its closest cluster. The performance metric would then be the silhouette score, which ranges between -1 and 1. That is, how well clusters are apart from each other and clearly distinguished. Currently, the threshold for publishing is set at __silhouette score > 0.2__. That is, the clusters to which the test set songs are assigned have some overlapping but are overall distinct. However, this threshold could be further altered according to circumstances.

### Business metric 
Since the application aims to destigmatize anime and exhibits the diversity of anime music to users, the metric would be how well the recommended music is perceived by the users. After each recommendation, the app would collect users' feedbacks (ratings out of 5, with 5 being most satisfied) on the recommended music. If the average ratings are high, then the diversity of anime music is well-delivered to the users and the goal of the project is met.

# Directory structure 

<details>
  <summary>Click to expand!</summary>

```
├── README.md                         <- You are here
├── app
│   ├── static/                       <- CSS, JS files that remain static
│   ├── templates/                    <- HTML (or other code) that is templated and changes based on a set of inputs│    
│
├── config                            <- Directory for configuration files 
│   ├── local/                        <- Directory for keeping environment variables and other local configurations that *do not sync** to Github 
│   ├── logging/                      <- Configuration of python loggers
│   ├── flaskconfig.py                <- Configurations for Flask API
│   ├── model.yaml                	  <- Configurations for the model pipeline
│
├── data                              <- Folder that contains data used or generated. Only the external/ sample/ and raw/ subdirectories are tracked by git. 
│   ├── external/                     <- External data sources, usually reference data, will be synced with git
│   ├── sample/                       <- Sample data used for code development and testing, will be synced with git
│   ├── raw/                      	  <- Raw data downloaded from Kaggle
│   ├── final/                        <- The final results of the model pipeline
│   ├── intermediate/                 <- The artifacts generated throughout the model pipeline
│
├── deliverables/                     <- Any white papers, presentations, final work products that are presented or delivered to a stakeholder 
|
├── dockerfiles/                      <- Directory for all project-related Dockerfiles 
│   ├── Dockerfile.app                <- Dockerfile for building image to run web app
│   ├── Dockerfile                    <- Dockerfile for building image to run the data acquisition, model pipeline and relational data ingestion
│   ├── Dockerfile.test               <- Dockerfile for building image to run unit tests
│
├── figures/                          <- Generated graphics and figures to be used in reporting, documentation, etc
│
├── models/                           <- Trained model objects (TMOs), model predictions, and model summaries
│
├── notebooks/
│   ├── archive/                      <- Develop notebooks no longer being used.
│   ├── deliver/                      <- Notebooks shared with others / in final state
│   ├── develop/                      <- Current notebooks being used in development.
│
├── reference/                        <- Any reference material relevant to the project
│
├── src/                              <- Source data for the project.
│
├── test/                             <- Files necessary for running model tests (see documentation below) 
│
├── app.py                            <- Flask wrapper for running the web app 
├── run.py                            <- Simplifies the execution of one or more of the src scripts regarding data acquisition and model pipeline
├── run_rds.py                        <- Simplifies the execution of the src scripts regarding relational data ingestion  
├── requirements.txt                  <- Python package dependencies 
```

</details>

---
# Running the whole pipeline aside from unit tests and web app

The whole pipeline contains the following steps:
* Data Acquisition
* Model Pipeline
	- Data cleaning and preprocessing
	- Feature generation
	- Model training
	- Model scoring/prediction
	- Model evaluation
* Relational Data Ingestion
	- Create table in a RDS database
	- Ingest data into the RDS table

## 0. Build docker image for the pipeline
Before completing any task in this section, you need to build the image first.
```bash
docker build -f dockerfiles/Dockerfile -t final-project .
```

or you can use the Makefile by typing
```bash
make image-model
```
__Note: All the following commands should be run from this directory (the root of the repo)__

## 1. Data Acquisition

To upload the raw data from your local path to the S3 bucket, you need to first provide AWS credentials, and also the S3 bukcet url in your current shell.
```bash
export AWS_ACCESS_KEY_ID = ...
export AWS_SECRET_ACCESS_KEY = ...
export S3_BUCKET = ...
```

Then, you can run the following code in the terminal to upload the data.
```bash
docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
-e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e S3_BUCKET \
final-project run.py acquire --input=data/raw/anime_songs.csv
```

or equivalently,

```bash
make s3-upload
```

---
## 2. Model Pipeline
As mentioned above, there are five steps involved in this section.

If you want to run the whole pipeline with one command, do 
```bash
make model-all
```

The docker commands or make commands to run these steps individually are as follows.

### 2.1 Data cleaning and preprocessing

<details>
  <summary>Click to expand!</summary>

```bash
docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
-e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e S3_BUCKET \
final-project run.py clean \
--file_output=data/intermediate/cleaned.csv --config=config/model.yaml \
--mid_output=data/raw/downloaded.csv
```

or equivalently,

```bash
make cleaned
```

</details>

### 2.2 Feature generation

<details>
  <summary>Click to expand!</summary>

```bash
docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
final-project run.py featurize --file_output=data/intermediate/features.csv \
--input=data/intermediate/cleaned.csv --config=config/model.yaml \
--model_output=models/scalar.joblib
```

or equivalently,

```bash
make features
```

</details>

### 2.3 Model training

<details>
  <summary>Click to expand!</summary>

```bash
docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
final-project run.py train --file_output=data/final/anime_clusters.csv \
--model_output=models/kmeans.joblib \
--input=data/intermediate/features.csv --config=config/model.yaml --origin_data=data/intermediate/cleaned.csv
```

or equivalently,

```bash
make models
```

</details>

### 2.4 Model scoring/prediction

<details>
  <summary>Click to expand!</summary>

```bash
docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
final-project run.py score --file_output=models/sample_clusters.csv --model=models/kmeans.joblib \
--input=data/sample/sample_search_songs.csv --scalar=models/scalar.joblib
```

or equivalently,

```bash
make scores
```

</details>

### 2.5 Model evaluation

<details>
  <summary>Click to expand!</summary>

```bash
docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
final-project run.py evaluate --file_output=models/sample_eval.txt \
--input=models/sample_clusters.csv
```

or equivalently,

```bash
make evaluate
```

</details>

---
## 3. Relational Data Ingestion

In order to take actions on the RDS instance on AWS, you would need to first provide AWS credentials, and also the SQLAlchemy database string in your current shell.
The SQLAlchemy database string could take the following formats:

`{dialect}://{user}:{pasword}>@{host}:{port}/{database}`

or

`sqlite:///data/{databasename}.db`

```bash
export AWS_ACCESS_KEY_ID = ...
export AWS_SECRET_ACCESS_KEY = ...
export SQLALCHEMY_DATABASE_URI = ...
```

#### 3.1 Create table in a RDS database

<details>
  <summary>Click to expand!</summary>

To create the database in the location offered in the environment variable, run: 

```bash
docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
-e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e SQLALCHEMY_DATABASE_URI \
final-project run_rds.py create
```

or 

```bash
make rds-create
```

</details>

#### 3.2 Ingest data into the RDS table

<details>
  <summary>Click to expand!</summary>

To add songs to the database:

```bash
docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
-e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e SQLALCHEMY_DATABASE_URI \
final-project run_rds.py add_data \
--data_path=data/final/anime_clusters.csv
```

or

```bash
make rds-ingest
```

</details>

# Running the web app

___Note: You have to be on the Northwestern VPN in order to run this app.__

`config/flaskconfig.py` holds the configurations for the Flask app. It includes the following configurations:

```python
DEBUG = True  # Keep True for debugging, change to False when moving to production 
LOGGING_CONFIG = "config/logging/local.conf"  # Path to file that configures Python logger
HOST = "0.0.0.0" # the host that is running the app. 0.0.0.0 when running locally 
PORT = 5000  # What port to expose app on. Must be the same as the port exposed in dockerfiles/Dockerfile.app 
APP_NAME = "anime_song_recommender"
SQLALCHEMY_TRACK_MODIFICATIONS = True 
SQLALCHEMY_ECHO = False  # If true, SQL for queries made will be printed
MAX_ROWS_SHOW = 10 # Limits the number of rows returned from the database 
FEATURES = ['danceability', 'energy', 'loudness',
       'speechiness', 'acousticness', 'instrumentalness',
        'liveness','valence', 'tempo'] # FEATURES USED WHEN FINDING THE CLOSEST CLUSTER
TOP_N = 10 # NUMBER OF TOP CLOSEST SONGS TO RETURN IN THE SEARCH RESULTS PAGE
```

---
## 0. Build the image 

To build the image, run from this directory (the root of the repo): 

```bash
docker build -f dockerfiles/Dockerfile.app -t final-project-app .
```

or equivalently

```bash
make image-app
```

---
## 1. Running the app

Before you run the Flask app, you would need your spotify api credentials,
you can find these credentials in [this website](https://developer.spotify.com/dashboard/applications) as long as you have a Spotify account.
All you need to do is to create an application, and then copy & paste the client IDs and the Secret.

<img src="https://github.com/MSIA/2022-msia423-Zhu-Simon-project/blob/main/figures/spotify_1.png" alt="drawing" height="200" width="1000"/>
<img src="https://github.com/MSIA/2022-msia423-Zhu-Simon-project/blob/main/figures/spotify_2.png" alt="drawing" height="200" width="1000"/>

Now export these variables into your current shell.

```bash
export SQLALCHEMY_DATABASE_URI = ...
export SPOTIPY_CLIENT_ID = ...
export SPOTIPY_CLIENT_SECRET = ...
```

Now to run the Flask app, run: 

```bash
docker run \
-e SPOTIPY_CLIENT_ID -e SPOTIPY_CLIENT_SECRET -e SQLALCHEMY_DATABASE_URI \
-p 5000:5000 final-project-app
```

or equivalently

```bash
make app
```

You should be able to access the app at http://127.0.0.1:5000/ in your browser (Mac/Linux should also be able to access the app at http://127.0.0.1:5000/ or localhost:5000/) .

The arguments in the above command do the following: 

* The `--rm` argument kills the container once the user is finished with it.
* The `--mount` argument allows the app to access your local `data/` folder so it can read from the SQLlite database created in the prior section. 
* The `-p 5000:5000` argument maps your computer's local port 5000 to the Docker container's port 5000 so that you can view the app in your browser. If your port 5000 is already being used for someone, you can use `-p 5001:5000` (or another value in place of 5001) which maps the Docker container's port 5000 to your local port 5001.

Note: If `PORT` in `config/flaskconfig.py` is changed, this port should be changed accordingly (as should the `EXPOSE 5000` line in `dockerfiles/Dockerfile.app`)

---
# Testing

Several unit tests for the eligible functions in the `src/` folder are also provided in this repository.

---
## 0. Build the Image

```bash
docker build -f dockerfiles/Dockerfile.test -t final-project-tests .
```

or equivalently,

```bash
make image-test
```

---
## 1. Run unit tests
To run the tests, run: 

```bash
docker run final-project-tests
```

or equivalently

```bash
make tests
```

---
# Other Utilities

To clean all the filesand artifacts build along the way after running your model pipeline, run
```bash
make clean-files
```

To clean the docker containers that you have built since the beginning
```bash
make clean-containers
```

To remove the images built for this project,
```bash
make clean-images
```

Or if you wanna delete everything that you've generated during this project, run
```bash
make clean-all
```