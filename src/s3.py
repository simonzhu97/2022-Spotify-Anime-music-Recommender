import logging
import re

import boto3
import botocore

logger = logging.getLogger(__name__)


def parse_s3(s3path: str) -> tuple[str, str]:
    """Parse the s3 path to get the names of the bucket and the path to the file of interest

    Arguments:
        s3path -- the s3 path

    Returns:
        s3bucket:   the s3 bucket name
        s3path:     the s3 path to the file of interest
    """
    # pattern of a regular s3 path
    regex = r"s3://([\w._-]+)/([\w./_-]+)"
    logger.debug("Given s3path: %s", s3path)

    matches = re.match(regex, s3path)
    s3bucket = matches.group(1)
    s3path = matches.group(2)

    return s3bucket, s3path


def upload_file_to_s3(local_path: str, s3path: str) -> None:
    """Upload file(s) from a local path to a location in S3

    Arguments:
        local_path -- The local path to the file to be uploaded
        s3path -- The path in s3 to store the file
    """
    s3bucket, s3_just_path = parse_s3(s3path)

    s_3 = boto3.resource("s3")
    bucket = s_3.Bucket(s3bucket)

    try:
        bucket.upload_file(local_path, s3_just_path)
    except botocore.exceptions.NoCredentialsError:
        logger.error(
            "Please provide AWS credentials via AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env variables.")
    except FileNotFoundError:
        logger.error("The local path does not exist.")
    else:
        logger.info("Data uploaded from %s to %s", local_path, s3path)


def download_file_from_s3(local_path: str, s3path: str) -> None:
    """Download a file to local path from s3

    Arguments:
        local_path -- the local path to store the file
        s3path -- the path on s3 where the file is stored
    """

    s3bucket, s3_just_path = parse_s3(s3path)

    s3_session = boto3.resource("s3")
    bucket = s3_session.Bucket(s3bucket)

    try:
        bucket.download_file(s3_just_path, local_path)
    except botocore.exceptions.NoCredentialsError:
        logger.error(
            "Please provide AWS credentials via AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env variables.")
    else:
        logger.info("Data downloaded from %s to %s", s3path, local_path)
