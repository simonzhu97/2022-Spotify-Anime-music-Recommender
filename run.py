"""Configures the subparsers for receiving command line arguments for each
 stage in the model pipeline and orchestrates their execution."""
import argparse
import logging.config
import os

import joblib
import pandas as pd
import yaml

from src.evaluate_model import assign_labels, assign_new_labels, evaluate
from src.model import get_model, save_model
from src.preprocessing import clean, featurize, read_from_local
from src.s3 import download_file_from_s3, upload_file_to_s3

logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("running_pipeline")

S3_BUCKET = os.getenv("S3_BUCKET")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("step", choices=["acquire", "clean", "featurize", "train", "score", "evaluate"],
                        help="which step to run")
    parser.add_argument("--input", "-i", default=None,
                        help="Path to input data")
    parser.add_argument("--model", "-m", default=None,
                        help="Path to the model to use for testing")
    parser.add_argument("--config", default="config/model.yaml",
                        help="Path to configuration file")
    parser.add_argument("--model_output", "-mo", default=None,
                        help="Path to save the generated model")
    parser.add_argument("--file_output", "-o", default=None,
                        help="Path to save output CSV or images (optional, default = None)")
    parser.add_argument("--mid_output", default=None,
                        help="Specifical to the cleaning step to save the downloaded file from s3")
    parser.add_argument("--origin_data", default=None,
                        help="specific for the train step, gives the cleaned unprocessed file")
    parser.add_argument("--scalar", default=None,
                        help="specific for the score step, gives the standard scalar used in featurize")

    args = parser.parse_args()

    # Load configuration file for parameters and tmo path
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        logger.error(
            "The path to the config file is wrong! Please check your path.")
    logger.info("Configuration file loaded from %s", args.config)

    if args.input is not None:
        # unless it"s the acquisition step, most input can be read as a dataframe
        if args.step not in ["acquire", "clean"]:
            try:
                file_in = pd.read_csv(args.input)
                logger.info("Input data loaded from %s", args.input)
            except FileNotFoundError:
                logger.error(
                    "Cannot find file at the specified input path %s", args.input)

    if args.origin_data is not None:
        try:
            origin_data = pd.read_csv(args.origin_data)
            logger.info("Input data loaded from %s", args.origin_data)
        except FileNotFoundError:
            logger.error(
                "Cannot find file at the specified input path %s", args.origin_data)

    if args.model is not None:
        try:
            model_in = joblib.load(args.model)
            logger.info("Input model loaded from %s", args.model)
        except FileNotFoundError:
            logger.error(
                "Cannot find model at the specified path %s", args.model)

    if args.scalar is not None:
        try:
            scalar_in = joblib.load(args.scalar)
            logger.info("Input scalar loaded from %s", args.scalar)
        except FileNotFoundError:
            logger.error(
                "Cannot find scalar at the specified path %s", args.scalar)

    # taking actions based on step name
    if args.step == "acquire":
        upload_file_to_s3(args.input, S3_BUCKET)
    elif args.step == "clean":
        # download file from s3 and save as an csv
        download_file_from_s3(local_path=args.mid_output, s3path=S3_BUCKET)
        file_in = read_from_local(args.mid_output)
        file_out = clean(file_in, **config["preprocessing"]["clean"])
    elif args.step == "featurize":
        file_out, model_out = featurize(
            file_in, **config["preprocessing"]["featurize"])
    elif args.step == "train":
        model_out = get_model(file_in, **config["model"]["get_model"])
        file_out = assign_labels(origin_data, model_out)
    elif args.step == "score":
        file_out = assign_new_labels(
            file_in, scalar_in, model_in, **config["evaluate_model"]["assign_new_labels"])
    else:
        file_out = evaluate(file_in)

    # define the output files
    if args.file_output is not None:
        try:
            if args.step not in ["acquire", "evaluate"]:
                file_out.to_csv(args.file_output, index=False)
            if args.step == "evaluate":
                with open(args.file_output, "w", encoding="utf-8") as f:
                    f.write(file_out)
            logger.info("Output saved to %s", args.file_output)
        except FileNotFoundError:
            logger.error(
                "The specified output path at %s does not exist", args.file_output)

    if args.model_output is not None:
        save_model(model_out, output_path=args.model_output)
