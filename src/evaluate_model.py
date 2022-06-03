"""
Assign each song a clusterId based on the optimal KMeans model
Evaluate the KMeans model results
"""
import logging

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.preprocessing import clean
from src.search_songs import get_closest_cluster

logger = logging.getLogger(__name__)

def assign_labels(df_features: pd.DataFrame, model: BaseEstimator) -> pd.DataFrame:
    """assigns labels to songs in the dataset based on optimal model result

    Arguments:
        df_features -- featurized dataset
        model -- trained KMeans model

    Returns:
        dataframe that has clusterId to it
    """
    df_new = pd.concat([df_features, pd.Series(
        model.labels_, name="clusterId")], axis=1)
    return df_new


def assign_new_labels(df_song: pd.DataFrame,
                      scalar: StandardScaler,
                      model: BaseEstimator,
                      clean_features: list[str],
                      col_mapper: dict,
                      features: list[str]) -> pd.DataFrame:
    """Assigns labels to new incoming songs based on the model

    Arguments:
        df_song -- a dataframe of given songs with features
        scalar -- the scalar used during the featurizing step
        model -- the KMeans model
        clean_features -- the features needed for data cleaning
        col-mapper -- the column mapper needed for data cleaning
        features -- a list of features to calculate cluster on

    Raises:
        TypeError -- the given model is not a KMeans model
        ValueError -- the given dataframe of songs is empty

    Returns:
        a dataframe of songs with both features and labels
    """
    # check if the model is a KMeans model
    if not isinstance(model, KMeans):
        logger.error("Invalid model! Needs to be a Kmeans model")
        raise TypeError("Invalid model")
    # check if df_song is empty:
    if len(df_song) == 0:
        logger.error("No song provided!")
        raise ValueError("No song provided.")
    # find centroids from the model, make it into a dataframe
    centroids = pd.DataFrame(model.cluster_centers_,
                             columns=model.feature_names_in_.tolist())
    # give cluster_ids
    centroids = centroids.reset_index().rename(columns={"index": "clusterId"})
    # first transform the df_song to scaled version
    df_song = clean(df_song, col_mapper, clean_features)
    df_song_scale = pd.DataFrame(scalar.transform(df_song[scalar.feature_names_in_.tolist()]),
                                 columns=scalar.feature_names_in_.tolist())
    ids = get_closest_cluster(df_song_scale, centroids, features)
    return pd.concat([df_song_scale, pd.Series(ids, name="clusterId")],
                     axis=1)


def evaluate(df_sample: pd.DataFrame) -> str:
    """Returns the silhouette score of a clustering results

    Arguments:
        df_sample -- the sample dataframe that has been assigned labels

    Raises:
        TypeError -- input not a dataframe
        KeyError -- input does not have a label column called "clusterId"

    Returns:
        the silhouette score of the clustered results
    """
    if not isinstance(df_sample, pd.DataFrame):
        logger.error("The argument needs to be a dataframe"
                     "but you have %s", type(df_sample))
        raise TypeError("Not dataframe input!")
    if "clusterId" not in df_sample.columns:
        logger.error(
            "Needs to have a clusterId label column in your dataframe")
        raise KeyError("No clusterId column")
    score = silhouette_score(df_sample.drop(
        "clusterId", axis=1), df_sample["clusterId"])
    return f"Silhoutee score: {score}"
