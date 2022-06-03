"""
Use a clusering model to cluster the anime songs
"""
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans

from src.preprocessing import validate_features

logger = logging.getLogger(__name__)
WRONG_INDICATOR = -10


def get_train_data(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Get the training data based on features given

    Arguments:
        df -- given preprocessed dataframe
        cols -- features to include in the K-Means algorithm

    Returns:
        dataframe for KMeans modelings
    """
    # check if the given columns are valid
    if validate_features(df, cols):
        df_in = df[cols]
        # if not all the columns provided are numeric,
        # choose only those that are numeric
        if df_in.select_dtypes(np.number).shape[1] < len(cols):
            df_in = df_in.select_dtypes(np.number)
            logger.error("Not all the columns given are numeric"
                         "KMeans require nuemric columns,"
                         "so %s are chosen", df_in.columns)
        else:
            logger.info("All the given columns are used in clustering.")
    else:
        # if not, use all the numeric columns in the dataframe
        df_in = df.select_dtypes(np.number)
        logger.error("Given features do not exists in the dataframe, %s are used instead",
                     df_in.columns)
    return df_in


def get_model(df: pd.DataFrame, cols: list[str],
              k: int,
              seed=42) -> BaseEstimator:
    """run a K-means model on a dataframe

    Arguments:
        df -- dataframe to work on
        cols -- features to include in the K-Means algorithm
        k -- number of clusters

    Keyword Arguments:
        seed -- random state seed (default: {42})

    Returns:
        a KMeans models
    """
    df_in = get_train_data(df, cols)
    # create a KMeans models
    mod = KMeans(n_clusters=k, random_state=seed).fit(df_in)
    return mod

def save_model(model: BaseEstimator, output_path: str) -> None:
    """A helper function that saves a model to a path

    Arguments:
        model -- Any model or sclaer class
        output_path -- the path to store the best model

    """
    try:
        joblib.dump(model, output_path)
        logger.info("The model is saved to %s", output_path)
    except FileNotFoundError as err:
        logger.error("Path does not exist at %s", output_path)
        raise err
