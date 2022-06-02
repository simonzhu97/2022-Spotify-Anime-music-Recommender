"""allow users to have two operation options regarding rds instance when running this script
    1. create a database
    2. add data to a specific database
"""
import argparse
import logging

import sqlalchemy.exc

import src.song_manager as songs
from config.flaskconfig import SQLALCHEMY_DATABASE_URI

logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("rds_running")

if __name__ == "__main__":
    # Add two subparsers to create a database or to add data to a existing database
    parser = argparse.ArgumentParser(
        description="Create database or add data to a current database")
    parser.add_argument("operation", default="create_db",
                        help="This argument decides whether to create a new database"
                        "Or to add data to a current database. You can choose between `create`"
                        "or  `add_data`",
                        choices=["create", "add_data"])
    parser.add_argument("--engine_string", default=SQLALCHEMY_DATABASE_URI,
                        help="SQLAlchemy connection URI for database")
    parser.add_argument("--data_path", default="data/intermediate/clustered_songs.csv",
                        help="If use add_data, then need to provide this argument."
                        "Gives a list of songs to be added.")

    args = parser.parse_args()

    if args.operation == "create":
        # test database connection
        try:
            songs.create_db(args.engine_string)
        except sqlalchemy.exc.OperationalError as err:
            logger.error("Could not connect to database!")
            raise err
    else:
        # add data from the csv file line by line to the database
        sm = songs.SongManager(engine_string=args.engine_string)
        sm.add_songs_from_csv(data_path=args.data_path)
        sm.close()
