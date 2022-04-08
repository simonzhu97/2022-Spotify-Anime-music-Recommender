# MSiA423 Anime Music Recommender
Author: Simon Zhu


# Table of Contents
* [Project Charter](#Project-Charter)
* [Directory structure ](#Directory-structure)
* [Running the app ](#Running-the-app)
	* [1. Initialize the database ](#1.-Initialize-the-database)
	* [2. Configure Flask app ](#2.-Configure-Flask-app)
	* [3. Run the Flask app ](#3.-Run-the-Flask-app)
* [Testing](#Testing)
* [Mypy](#Mypy)
* [Pylint](#Pylint)

# Project Charter
## Background
Anime is hand-drawn and computer animation originating from Japan. In Japan, anime describes all animated works, regardless of style or origin. However, outside of Japan, anime usually refers specifically to animation produced in Japan. 

Since the 1980s, this particular medium has witnessed great international success with the rise of subtitles and increasing distribution through streaming services. In fact, streaming services such as Netflix have invested quite some amount in this genre. In 2019, the annual overseas exports of Japanese animation exceeded $10 billion.

<img src="https://github.com/MSIA/2022-msia423-Zhu-Simon-project/blob/main/figures/aot.jpeg" alt="drawing" height="300" width="500"/>

## Vision
Despite rapid growth of overseas exports, stigmas about anime still linger. Take anime music as an example. A vast majority still considers anime music as cute music that might not have much meaning. Nevertheless, in the recent years, one can find traces of jazz, rock and many other elements in anime music. By exposing people to anime music that is similar to other widely-listened music, the project could bring awareness to the diversity of anime music and therefore even piques interests in the anime industry itself. Thus, the target audience of the application is those who are explorative and want to identify other sources of music that fit their taste.

## Mission
The project aims to provide a music recommender application where the app would recommend anime songs similar to the one the user inputs. The recommended songs would be pulled from a [dataset](https://www.kaggle.com/datasets/simonzhu97/popular-spotify-anime-songs) that includes recent hit anime songs on Spotify. 

In a text input field, the user will input a song they like and the app will search for this song on Spotify, extract the corresponding song features, compare them to the songs in the anime-song-dataset, and recommend the top 5 similar anime songs based on a cosine similarity measure. For example, if the user inputs __"City of Stars"__, the app might output the following table, where each row represents a recommended song. 

|      | song name  | genre | link                                     |
| ---: | :--------- | :---- | :--------------------------------------- |
|    1 | One Last Kiss | jazz | https://open.spotify.com/track/5RhWszHMSKzb7KiXk4Ae0M?si=88e08a24b2fb4c63 |
|    ... | ...  | ... | ... |

The table above could be altered as the project proceeds. For instance, other song features such as release dates and which anime it comes from could also be included in the table if time permits.

## Success Criteria

### Model performance metric
A test dataset would be created where 10 random pop songs would be chosen. Then, for each of these 10 songs, the top 5 anime songs that are most similar are hand-picked by knowledgeable people. An example test dataset is as follows.

|      | song name  | Similar to the input song "Levitating"|
| ---: | :--------- | :---- |
|    1 | One Last Kiss | F | 
|    2 | Lost in Paradise | T |
|    3 | Black Catcher | T |
|    ... | ... | ... |
|    439 | Blue Sky | F |

Prior to deployment, the recommendation system would be tested on this dataset and the performance metric would be the precision@k. That is, among the top k songs that the model recommends, how many of them are labeled as "similar to the input song"? Currently, the threshold for publishing is set at __precision@5 = 2__. That is, at least two of the five songs that the model recommends are labeled as "similar to the input song" in the test dataset. However, this threshold could be further altered according to circumstances.

### Business metric 
Since the application aims to destigmatize anime and exhibits the diversity of anime music to users, the metric would be how well the recommended music is perceived by the users. After each recommendation, the app would collect users' feedbacks (ratings out of 5, with 5 being most satisfied) on the recommended music. If the average ratings are high, then the diversity of anime music is well-delivered to the users and the goal of the project is met.

## Directory structure 

```
├── README.md                         <- You are here
├── api
│   ├── static/                       <- CSS, JS files that remain static
│   ├── templates/                    <- HTML (or other code) that is templated and changes based on a set of inputs│    
│
├── config                            <- Directory for configuration files 
│   ├── local/                        <- Directory for keeping environment variables and other local configurations that *do not sync** to Github 
│   ├── logging/                      <- Configuration of python loggers
│   ├── flaskconfig.py                <- Configurations for Flask API 
│
├── data                              <- Folder that contains data used or generated. Only the external/ and sample/ subdirectories are tracked by git. 
│   ├── external/                     <- External data sources, usually reference data,  will be synced with git
│   ├── sample/                       <- Sample data used for code development and testing, will be synced with git
│
├── deliverables/                     <- Any white papers, presentations, final work products that are presented or delivered to a stakeholder 
│
├── docs/                             <- Sphinx documentation based on Python docstrings. Optional for this project.
|
├── dockerfiles/                      <- Directory for all project-related Dockerfiles 
│   ├── Dockerfile.app                <- Dockerfile for building image to run web app
│   ├── Dockerfile.run                <- Dockerfile for building image to execute run.py  
│   ├── Dockerfile.test               <- Dockerfile for building image to run unit tests
│
├── figures/                          <- Generated graphics and figures to be used in reporting, documentation, etc
│
├── models/                           <- Trained model objects (TMOs), model predictions, and/or model summaries
│
├── notebooks/
│   ├── archive/                      <- Develop notebooks no longer being used.
│   ├── deliver/                      <- Notebooks shared with others / in final state
│   ├── develop/                      <- Current notebooks being used in development.
│   ├── template.ipynb                <- Template notebook for analysis with useful imports, helper functions, and SQLAlchemy setup. 
│
├── reference/                        <- Any reference material relevant to the project
│
├── src/                              <- Source data for the project. No executable Python files should live in this folder.  
│
├── test/                             <- Files necessary for running model tests (see documentation below) 
│
├── app.py                            <- Flask wrapper for running the web app 
├── run.py                            <- Simplifies the execution of one or more of the src scripts  
├── requirements.txt                  <- Python package dependencies 
```

## Running the app 

### 1. Initialize the database 
#### Build the image 

To build the image, run from this directory (the root of the repo): 

```bash
 docker build -f dockerfiles/Dockerfile.run -t pennylanedb .
```
#### Create the database 
To create the database in the location configured in `config.py` run: 

```bash
docker run --mount type=bind,source="$(pwd)"/data,target=/app/data/ pennylanedb create_db  --engine_string=sqlite:///data/tracks.db
```
The `--mount` argument allows the app to access your local `data/` folder and save the SQLite database there so it is available after the Docker container finishes.


#### Adding songs 
To add songs to the database:

```bash
docker run --mount type=bind,source="$(pwd)"/data,target=/app/data/ pennylanedb ingest --engine_string=sqlite:///data/tracks.db --artist=Emancipator --title="Minor Cause" --album="Dusk to Dawn"
```

#### Defining your engine string 
A SQLAlchemy database connection is defined by a string with the following format:

`dialect+driver://username:password@host:port/database`

The `+dialect` is optional and if not provided, a default is used. For a more detailed description of what `dialect` and `driver` are and how a connection is made, you can see the documentation [here](https://docs.sqlalchemy.org/en/13/core/engines.html). We will cover SQLAlchemy and connection strings in the SQLAlchemy lab session on 
##### Local SQLite database 

A local SQLite database can be created for development and local testing. It does not require a username or password and replaces the host and port with the path to the database file: 

```python
engine_string='sqlite:///data/tracks.db'

```

The three `///` denote that it is a relative path to where the code is being run (which is from the root of this directory).

You can also define the absolute path with four `////`, for example:

```python
engine_string = 'sqlite://///Users/cmawer/Repos/2022-msia423-template-repository/data/tracks.db'
```


### 2. Configure Flask app 

`config/flaskconfig.py` holds the configurations for the Flask app. It includes the following configurations:

```python
DEBUG = True  # Keep True for debugging, change to False when moving to production 
LOGGING_CONFIG = "config/logging/local.conf"  # Path to file that configures Python logger
HOST = "0.0.0.0" # the host that is running the app. 0.0.0.0 when running locally 
PORT = 5000  # What port to expose app on. Must be the same as the port exposed in dockerfiles/Dockerfile.app 
SQLALCHEMY_DATABASE_URI = 'sqlite:///data/tracks.db'  # URI (engine string) for database that contains tracks
APP_NAME = "penny-lane"
SQLALCHEMY_TRACK_MODIFICATIONS = True 
SQLALCHEMY_ECHO = False  # If true, SQL for queries made will be printed
MAX_ROWS_SHOW = 100 # Limits the number of rows returned from the database 
```

### 3. Run the Flask app 

#### Build the image 

To build the image, run from this directory (the root of the repo): 

```bash
 docker build -f dockerfiles/Dockerfile.app -t pennylaneapp .
```

This command builds the Docker image, with the tag `pennylaneapp`, based on the instructions in `dockerfiles/Dockerfile.app` and the files existing in this directory.

#### Running the app

To run the Flask app, run: 

```bash
 docker run --name test-app --mount type=bind,source="$(pwd)"/data,target=/app/data/ -p 5000:5000 pennylaneapp
```
You should be able to access the app at http://127.0.0.1:5000/ in your browser (Mac/Linux should also be able to access the app at http://127.0.0.1:5000/ or localhost:5000/) .

The arguments in the above command do the following: 

* The `--name test-app` argument names the container "test". This name can be used to kill the container once finished with it.
* The `--mount` argument allows the app to access your local `data/` folder so it can read from the SQLlite database created in the prior section. 
* The `-p 5000:5000` argument maps your computer's local port 5000 to the Docker container's port 5000 so that you can view the app in your browser. If your port 5000 is already being used for someone, you can use `-p 5001:5000` (or another value in place of 5001) which maps the Docker container's port 5000 to your local port 5001.

Note: If `PORT` in `config/flaskconfig.py` is changed, this port should be changed accordingly (as should the `EXPOSE 5000` line in `dockerfiles/Dockerfile.app`)


#### Kill the container 

Once finished with the app, you will need to kill the container. If you named the container, you can execute the following: 

```bash
docker kill test-app 
```
where `test-app` is the name given in the `docker run` command.

If you did not name the container, you can look up its name by running the following:

```bash 
docker container ls
```

The name will be provided in the right most column. 

## Testing

Run the following:

```bash
 docker build -f dockerfiles/Dockerfile.test -t pennylanetest .
```

To run the tests, run: 

```bash
 docker run pennylanetest
```

The following command will be executed within the container to run the provided unit tests under `test/`:  

```bash
python -m pytest
``` 

## Mypy

Run the following:

```bash
 docker build -f dockerfiles/Dockerfile.mypy -t pennymypy .
```

To run mypy over all files in the repo, run: 

```bash
 docker run pennymypy .
```
To allow for quick iteration, mount your entire repo so changes in Python files are detected:


```bash
 docker run --mount type=bind,source="$(pwd)"/,target=/app/ pennymypy .
```

To run mypy for a single file, run: 

```bash
 docker run pennymypy run.py
```

## Pylint

Run the following:

```bash
 docker build -f dockerfiles/Dockerfile.pylint -t pennylint .
```

To run pylint for a file, run:

```bash
 docker run pennylint run.py 
```

(or any other file name, with its path relative to where you are executing the command from)

To allow for quick iteration, mount your entire repo so changes in Python files are detected:


```bash
 docker run --mount type=bind,source="$(pwd)"/,target=/app/ pennylint run.py
```
