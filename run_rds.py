import sqlalchemy as sql
import sqlalchemy.exc
from sqlalchemy.ext.declarative import declarative_base
import src.add_songs as songs
import logging
import os

engine_string = os.getenv("SQLALCHEMY_DATABASE_URI")
if engine_string is None:
    raise RuntimeError(
        "SQLALCHEMY_DATABASE_URI environment variable not set; exiting")

logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("rds_running")

Base = declarative_base()

if __name__ == "__main__":
    # set up mysql connection
    engine = sql.create_engine(engine_string)

    # test database connection
    try:
        songs.create_db(engine_string)
    except sqlalchemy.exc.OperationalError as e:
        logger.error("Could not connect to database!")
        logger.debug("Database URI: %s", )
        raise e
    
