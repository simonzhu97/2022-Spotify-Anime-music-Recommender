"""
Script running the flask app
"""
import logging.config
import sqlite3
import traceback

import pandas as pd
import spotipy
import sqlalchemy.exc
from flask import Flask, redirect, render_template, request, url_for

# For setting up the Flask-SQLAlchemy database session
from src.add_songs import SongManager, Songs
from src.search_songs import get_closest_cluster, get_song_features, get_top_n_closest_song

# Initialize the Flask application
app = Flask(__name__, template_folder="app/templates",
            static_folder="app/static")

# Configure flask app from flask_config.py
app.config.from_pyfile('config/flaskconfig.py')

# Define LOGGING_CONFIG in flask_config.py - path to config file for setting
# up the logger (e.g. config/logging/local.conf)
logging.config.fileConfig(app.config["LOGGING_CONFIG"])
logger = logging.getLogger(app.config["APP_NAME"])
logger.debug(
    'Web app should be viewable at %s:%s if docker run command maps local '
    'port to the same port as configured for the Docker container '
    'in config/flaskconfig.py (e.g. `-p 5000:5000`). Otherwise, go to the '
    'port defined on the left side of the port mapping '
    '(`i.e. -p THISPORT:5000`). If you are running from a Windows machine, '
    'go to 127.0.0.1 instead of 0.0.0.0.', app.config["HOST"], app.config["PORT"])

# Initialize the database session
song_manager = SongManager(app)


@app.route('/')
def index():
    """Main view that lists songs in the database.

    Create view into index page that uses data queried from Song database and
    inserts it into the app/templates/index.html template.

    Returns:
        Rendered html template

    """

    try:
        songs = song_manager.session.query(Songs).limit(
            app.config["MAX_ROWS_SHOW"]).all()
        logger.debug("Index page accessed")
        return render_template('index.html', songs=songs)
    except sqlite3.OperationalError as err:
        logger.error(
            "Error page returned. Not able to query local sqlite database: %s."
            " Error: %s ",
            app.config['SQLALCHEMY_DATABASE_URI'], err)
        return render_template('error.html')
    except sqlalchemy.exc.OperationalError as err:
        logger.error(
            "Error page returned. Not able to query MySQL database: %s. "
            "Error: %s ",
            app.config['SQLALCHEMY_DATABASE_URI'], err)
        return render_template('error.html')
    except:
        traceback.print_exc()
        logger.error("Not able to display tracks, error page returned")
        return render_template('error.html')


@app.route('/search', methods=['POST'])
def get_entry():
    """View that process a post with new song input
    It searches for the closest cluster of the song

    Returns:
        redirect to index page
    """

    try:
        # define the song name to be searched
        song_name = request.form['song_name']
        artist = request.form['artist']

        # search for the song -> returns df_song with all possible features
        try:
            song_features = get_song_features(song_name, artist)
        except KeyError:
            # TODO: prints the log stream to the page!
            return redirect(url_for('index'))
        except spotipy.SpotifyException:
            return redirect(url_for('index'))

        # in case there aren't any search results
        if len(song_features) == 0:
            return redirect(url_for('index'))

        centroids = pd.read_csv(app.config['CENTROIDS_PATH'])

        # get closest cluster
        cluster_id = get_closest_cluster(song_features, centroids,
                                        app.config['FEATURES'], app.config['TARGET'])
        logger.info("Song queried: %s by %s", request.form['song_name'],
                    request.form['artist'])
        logger.debug("The closest cluster is Cluster %d", cluster_id)

        # fetch all songs in that cluster

        songs = pd.read_sql_query(
            sql=song_manager.session.query(
                Songs).filter_by(clusterId=cluster_id).statement,
            con=song_manager.database.engine
        )
        
        # find the closest ones in terms of cosine_similarity
        top_songs = get_top_n_closest_song(
            song_features, songs, app.config['FEATURES'], app.config['TARGET'], app.config['TOP_N'])
        
        return render_template('search_results.html',
                               searched=f"{song_name} by {artist}",
                               songs=top_songs)

    except sqlite3.OperationalError as err:
        logger.error(
            "Error page returned. Not able to add song to local sqlite "
            "database: %s. Error: %s ",
            app.config['SQLALCHEMY_DATABASE_URI'], err)
        return render_template('error.html')
    except sqlalchemy.exc.OperationalError as err:
        logger.error(
            "Error page returned. Not able to add song to MySQL database: %s. "
            "Error: %s ",
            app.config['SQLALCHEMY_DATABASE_URI'], err)
        return render_template('error.html')


if __name__ == '__main__':
    app.run(debug=app.config["DEBUG"], port=app.config["PORT"],
            host=app.config["HOST"])