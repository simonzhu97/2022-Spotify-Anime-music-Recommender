""" 
Thhe module contains methods that 
    1. searches for a song's features by calling Spotify's API
    2. returns the cluster of which the centroid is closest to the song in terms of cosine similarity
    3. returns the top N closest (in terms of cosine similarity) songs to a given song in a given cluster
"""
import logging
import os

import numpy as np
import pandas as pd
import spotipy
from sklearn.metrics.pairwise import cosine_similarity
from spotipy.oauth2 import SpotifyClientCredentials

logger = logging.getLogger(__name__)

CID = os.getenv("SPOTIPY_CLIENT_ID")
SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

# establish a spotify api service

client_credentials_manager = SpotifyClientCredentials(
    client_id=CID, client_secret=SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def get_song_features(song_name: str, artist_name: str) -> pd.DataFrame:
    """Get the features for a song. If the search returns nothing,
    then returns an empty dataframe.

    Arguments:
        song_name -- the song name to be searched for
        artist_name -- the artist of the song to be searched for

    Returns:
        A dataframe containing the features of the song (one row)

    Raises:
        spotipy.exceptions.SpotifyException
        KeyError
    """
    # if the user types in both fields, we are gucci
    if song_name and artist_name:
        song_to_search = f"artist:{artist_name} track:{song_name}"
    # if either field but not both is missing, just search for whatever you have
    elif song_name and not artist_name:
        song_to_search = f"track:{song_name}"
    elif artist_name and not song_name:
        song_to_search = f"artist:{artist_name}"
    # if none is provided, raise an error
    else:
        logger.error("Please provide at least one field. Song name or artist name.")
        raise KeyError("Fields missing")
    
    # use spotipy to search for the song
    try:
        search_results = sp.search(song_to_search, limit=1, offset=0, type="track")
        logger.info("The song %s by %s has been searched", song_name, artist_name)
    except spotipy.oauth2.SpotifyOauthError as err:
        logger.error(
            "The configured client id and secret does not match. Please check them again!")
        raise err

    # in case the search results returned nothing
    if search_results['tracks']['total']>0:
        song_id = search_results['tracks']['items'][0]['id']
        res = [sp.audio_features(tracks=song_id)[0]]
        # transform list of song features to a dataframe that contains one row - pertaining to the song
        df = pd.DataFrame.from_records(res, index=[0]*len(res))
        logger.info("%d song has been returned.", len(res))
    else:
        logger.info("The search returns nothing. The song you are looking for does not currently exist in Spotify!"
                    "Please search for a new song.")
        df = pd.DataFrame([])

    return df

def valid_features(sub_features:list[str],all_features:list[str])->bool:
    """A helper function that returns sub_features only if all of them are valid

    Arguments:
        sub_features -- sub_feature lists to go through
        all_features -- the original complete list of features

    Returns:
        a boolean indicating whether the sub_features are all valid
    
    Raises:
        KeyError
    """
    if isinstance(sub_features,str):
        sub_features = [sub_features]
    # check if the features provided exist
    for fea in sub_features:
        if not (fea in all_features):
            return False
    return True

def get_closest_cluster(df_song:pd.DataFrame,
                        centroids:pd.DataFrame,
                        features:list[str],
                        target: str)->int:
    """Based on the selected features, choose the clusters with a centroid closest to the song provided.

    Arguments:
        df_song -- one row of song features
        centroids -- the datafram containing info about centroids
        features -- the features to use when calculating distance
        target -- the column name for cluster id

    Returns:
        the closest clusterId
    """
    if not (valid_features(features,df_song.columns) and valid_features(features,centroids.columns)):
        logger.error("The features selected is not an available song feature!")
        raise KeyError("Nonexisting feature")
  
    # check if the target provided exist
    if not (valid_features(target, centroids.columns)):
        logger.error("The target column provided does not exist.")
        raise KeyError("Nonexisting target column")
    
    # calculate distance between the song and each cluster
    dist = cosine_similarity(df_song[features],centroids[features])
    logger.debug(dist)
    # return the cluster with the closest centroid
    return int(centroids.loc[np.argmax(dist,axis=-1),"clusterId"])

def get_top_n_closest_song(df_song: pd.DataFrame, df_anime: pd.DataFrame,
                           features:list[str], target: str, top_n: int) -> list[dict]:
    """Select the top N closest song to the given song

    Arguments:
        df_song -- A dataframe containing the features of the song (one row)
        df_anime -- A dataframe containing all possible anime songs with cluster ids
        features -- A list of features to compute distance on
        target -- The column name standing for cluster id
        N -- number of songs to find

    Returns:
        a dataframe containing information about the found closest songs
    """
    
    if not (valid_features(features,df_song.columns) and valid_features(features,df_anime.columns)):
        logger.error("The features selected is not an available song feature!")
        raise KeyError("Nonexisting feature")
    # check if the target provided exist
    if not (valid_features(target, df_anime.columns)):
        logger.error("The target column provided does not exist.")
        raise KeyError("Nonexisting target column")

    dist = cosine_similarity(df_song[features], df_anime[features])
    inds = np.argpartition(dist,-top_n,axis=-1)
    return df_anime.loc[inds[0][-top_n:]].to_dict('records')