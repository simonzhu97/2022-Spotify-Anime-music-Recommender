"""
A configuration file containing configurations for the flask app
"""

import os
DEBUG = True
LOGGING_CONFIG = "config/logging/local.conf"
PORT = 5000
APP_NAME = "anime_song_recommender"
SQLALCHEMY_TRACK_MODIFICATIONS = True
HOST = "0.0.0.0"
SQLALCHEMY_ECHO = False  # If true, SQL for queries made will be printed
MAX_ROWS_SHOW = 10

SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')
if SQLALCHEMY_DATABASE_URI is None:
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://user:password@host:3306/msia423_db'

FEATURES = ['danceability', 'energy', 'loudness',
       'speechiness', 'acousticness', 'instrumentalness',
        'liveness','valence', 'tempo']
TARGET = "clusterId"
TOP_N = 10
CENTROIDS_PATH =  "data/intermediate/centroids.csv"
