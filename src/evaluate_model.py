"""
Assign each song a clusterId based on the optimal KMeans model
Evaluate the KMeans model results
"""
import pandas as pd
import logging

from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


def summary_kmeans(model:BaseEstimator) -> pd.DataFrame:
    """Showcase the centroid information of the clustering model

    Arguments:
        model -- trained KMeans model

    Returns:
        Dataframe containing cluster centroid information
        if no valid model given, will output an empty dataframe
    """
    if not model:
        logger.error("There is not a valid model given.")
        return pd.DataFrame([])
    cols = model.feature_names_in_.tolist()
    df_clusters = pd.DataFrame(model.cluster_centers_, columns=cols)
    df_clusters['count'] = pd.Series(model.labels_).value_counts().sort_index()
    df_clusters['percent'] = df_clusters['count']/df_clusters['count'].sum()
    columns_ordered = ['count', 'percent'] + cols
    df_clusters = df_clusters[columns_ordered]
    return df_clusters


def assign_labels(df_features:pd.DataFrame, model: BaseEstimator) -> pd.DataFrame:
    """assigns labels to songs in the dataset based on optimal model result

    Arguments:
        df_features -- featurized dataset
        model -- trained KMeans model

    Returns:
        dataframe that has clusterId to it
    """
    df_new = pd.concat([df_features,pd.Series(model.labels_,name="clusterId")],axis=1)
    return df_new

def evaluate():
    # give a test set and see how the clustering algorithm works
    pass