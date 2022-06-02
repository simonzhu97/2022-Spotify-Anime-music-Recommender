"""Creates, ingests data into, and enables querying of a table of
 songs for the anime_recommender app to query from and display results to the user."""

import logging.config
import sqlite3
import typing

import flask
import pandas as pd
import sqlalchemy
import sqlalchemy.orm
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base

logger = logging.getLogger(__name__)

Base: typing.Any = declarative_base()


class Songs(Base):
    """Creates a data model for the database to be set up for capturing songs
    and assigning songs cluster ids.
    """

    __tablename__ = "songs"

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    title = sqlalchemy.Column(sqlalchemy.UnicodeText(100), unique=False,
                              nullable=False)
    clusterId = sqlalchemy.Column(sqlalchemy.Integer, unique=False,
                                  nullable=False)

    danceability = sqlalchemy.Column(sqlalchemy.Float, unique=False,
                                     nullable=False)
    energy = sqlalchemy.Column(sqlalchemy.Float, unique=False,
                               nullable=False)
    loudness = sqlalchemy.Column(sqlalchemy.Float, unique=False,
                                 nullable=False)
    key = sqlalchemy.Column(sqlalchemy.Integer, unique=False,
                            nullable=False)
    speechiness = sqlalchemy.Column(sqlalchemy.Float, unique=False,
                                    nullable=False)
    acousticness = sqlalchemy.Column(sqlalchemy.Float, unique=False,
                                     nullable=False)
    instrumentalness = sqlalchemy.Column(sqlalchemy.Float, unique=False,
                                         nullable=False)
    liveness = sqlalchemy.Column(sqlalchemy.Float, unique=False,
                                 nullable=False)
    valence = sqlalchemy.Column(sqlalchemy.Float, unique=False,
                                nullable=False)
    tempo = sqlalchemy.Column(sqlalchemy.Float, unique=False,
                              nullable=False)
    duration = sqlalchemy.Column(sqlalchemy.Integer, unique=False,
                                 nullable=False)
    track_uri = sqlalchemy.Column(sqlalchemy.String(200), unique=True,
                                  nullable=False)

    def __repr__(self):
        return f"<Song {self.title} {self.track_uri}>"

    def __str__(self):
        return f"<Song {self.title} {self.track_uri}>"


class SongManager:
    """Creates a SQLAlchemy connection to the songs table.

    Args:
        app (:obj:`flask.app.Flask`): Flask app object for when connecting from
            within a Flask app. Optional.
        engine_string (str): SQLAlchemy engine string specifying which database
            to write to.
    """

    def __init__(self, app: typing.Optional[flask.app.Flask] = None,
                 engine_string: typing.Optional[str] = None):
        if app:
            self.database = SQLAlchemy(app)
            self.session = self.database.session
        elif engine_string:
            engine = sqlalchemy.create_engine(engine_string)
            session_maker = sqlalchemy.orm.sessionmaker(bind=engine)
            self.session = session_maker()
        else:
            raise ValueError(
                "Need either an engine string or a Flask app to initialize")

    def close(self) -> None:
        """Closes SQLAlchemy session

        Returns: None

        """
        self.session.close()

    def add_song(self, **kwargs) -> None:
        """Seeds an existing database with additional songs.

        Args:
            **kwargs: Other important features of the song to add to database

        Returns:
            None
        """

        session = self.session
        song = Songs(**kwargs)
        session.add(song)
        session.commit()
        try:
            title = kwargs["title"]
            logger.info("A song called %s is added to database", title)
        except KeyError:
            logger.error("The input song features do not contain 'title'")

    def add_songs_from_csv(self, data_path: str) -> None:
        """Reads songs from a csv file and add them to the database

        Arguments:
            data_path -- the path to the csv file to read
        """

        session = self.session

        # transform the dataframe to a dictionary for convenience
        data_list = pd.read_csv(data_path).to_dict(orient="records")
        persist_list = []

        # prepare data to persist into the database
        for data in data_list:
            persist_list.append(Songs(**data))

        logger.debug("%d records are prepared to be persisted",
                     len(persist_list))

        # add all songs to the database
        try:
            session.add_all(persist_list)
            session.commit()
        except sqlite3.OperationalError as err:
            logger.error(
                "Error page returned. Not able to add song to local sqlite "
                "Are you offering the right database path? Error: %s ",
                err)
        except sqlalchemy.exc.OperationalError as err:
            logger.error(
                "Error page returned. Not able to add song to MySQL database.  "
                "Please check engine string and VPN. \n Error: %s ", err)
        except sqlalchemy.exc.IntegrityError:
            my_message = ("Have you already inserted the same record into the database before? \n"
                          "This database does not allow duplicate in the input-recommendation pair")
            logger.error("%s \n The original error message is: ",
                         my_message, exc_info=True)
        else:
            logger.info("%d songs have been added to the database!",
                        len(persist_list))


def create_db(engine_string: str) -> None:
    """Create database with Songs() data model from provided engine string.

    Args:
        engine_string (str): SQLAlchemy engine string specifying which database
            to write to

    Returns: None

    """
    engine = sqlalchemy.create_engine(engine_string)

    Base.metadata.create_all(engine)
    logger.info("Database created.")
