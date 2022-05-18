import argparse
import logging.config

from src.s3 import download_file_from_s3, upload_file_to_s3

logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("s3_running")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('action', default='upload', choices=['upload', 'download'],
                        help="The action to take."
                        "If 'upload' is given, will upload a file from the local path to the s3 path"
                        "If 'download' is given, will download a file from s3 path to the local path."
                        "Default action is to upload files")
    parser.add_argument('--s3path', default='s3://2022-msia423-zhu-simon/data/raw/anime_songs.csv',
                        help="If used, will load data")
    parser.add_argument('--local_path', default='data/raw/anime_songs.csv',
                        help="Where to load data in S3")
    args = parser.parse_args()

    if args.action == 'upload':
        upload_file_to_s3(args.local_path, args.s3path)
    else:
        download_file_from_s3(args.local_path, args.s3path)
