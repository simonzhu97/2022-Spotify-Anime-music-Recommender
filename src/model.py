"""
Use a clusering model to cluster the anime songs
"""
import logging
from typing import Optional

import joblib
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.preprocessing import validate_features

logger = logging.getLogger(__name__)
WRONG_INDICATOR = -10

def get_train_data(df: pd.DataFrame, cols:list[str]) -> pd.DataFrame:
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

def get_models_dict(df:pd.DataFrame, cols:list[str],
                    k_range:list[int],
                    seed=42, return_list=False)->tuple[dict,Optional[list]]:
    """run several K-means models with different number of clusters K on a dataframe

    Arguments:
        df -- dataframe to work on
        cols -- features to include in the K-Means algorithm
        k_range -- range of number of clusters to try

    Keyword Arguments:
        seed -- random state seed (default: {42})
        return_list -- whether to also return a list of models (default: {False})

    Returns:
        If return_list = True, will return both a dictionary and a list of different KMeans models
        If return_list = False, will only return a dictionary of different KMeans models
    """
    df_in = get_train_data(df, cols)
    model_dict = {}
    # create different KMeans models
    for k in k_range:
        mod = KMeans(n_clusters=k, random_state=seed).fit(df_in)
        model_dict[str(k)] = mod
    res = model_dict

    if return_list:
        ls_mods = list(model_dict.values())
        res = (model_dict, ls_mods)
    return res

def plot_models_performance(df:pd.DataFrame,
                            mod_list:list[BaseEstimator],
                            k_range: list[int]) -> plt.figure:
    """
    Plot silhouette scores and inertia to identify the optimal
    number of clusters to use for the model prediction

    Arguments:
        df -- the original datafram with all the features
        mod_list -- list of KMeans models established
        k_range -- range of number of clusters to try

    Returns:
        a matplotlib figure
    """
    # check if the cols
    within_ss = [i.inertia_ for i in mod_list]
    try:
        silhouette_list = [silhouette_score(df[i.feature_names_in_], i.labels_) for i in mod_list]
    except KeyError as err:
        logger.error("The dataframe does not contain features used in the KMeans model."
                     "Please recheck the dataframe.")
        raise err

    # check if k_range corresponds with that from KMeans model
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    if len(k_range) != len(within_ss):
        logger.error("The k_range given does not align with that from the modeling process!")
        logger.warning("The x-labels of the graphs will be off.")
        k_range = list(range(WRONG_INDICATOR,len(within_ss)+WRONG_INDICATOR))
    axs[0].plot(k_range, within_ss, color='red')
    axs[1].plot(k_range, silhouette_list, color='orange')

    for i in range(2):
        axs[i].set_xlabel('number of clusters')
    for idx, name in zip([0,1,2], ['inertia', 'silhouette score']):
        axs[idx].set_ylabel(name)
    return fig

def save_model(model_dict: dict, best_num: str, output_path: str) -> BaseEstimator:
    """A helper function that saves the dict of models to a path

    Arguments:
        model_dict -- A dictionary of KMeans models (num_of_cluster: MODEL)
        best_num -- the optimal number of cluster
        output_path -- the path to store the best model
        
    Returns:
        the best model
    """
    # ensure that the best_num is string
    if isinstance(best_num,int):
        best_num = str(best_num)
    try:
        joblib.dump(model_dict[best_num], output_path)
        logger.info('The model is saved to %s', output_path)
        return model_dict[best_num]
    except FileNotFoundError as err:
        logger.error("Path does not exist at %s", output_path)
        raise err
    except KeyError as err:
        logger.error("Model with the given best_num %s has not been built yet!", best_num)
        raise err
