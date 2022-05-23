import logging
import os

import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.metrics.pairwise import euclidean_distances

logger = logging.getLogger(__name__)

CID = os.getenv("SPOTIPY_CLIENT_ID")
SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")


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
    # establish a spotify api service

    client_credentials_manager = SpotifyClientCredentials(
        client_id=CID, client_secret=SECRET)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

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
    if search_results:
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
    # check if the features provided exist
    for fea in features:
        if fea in df_song.columns and fea in centroids.columns:
            continue
        else:
            logger.error("The features selected is not an available song feature!")
            raise KeyError("Nonexisting feature")
    # check if the target provided exist
    if not (target in centroids.columns):
        logger.error("The target column provided does not exist.")
        raise KeyError("Nonexisting target column")
    
    # calculate distance between the song and each cluster
    dist = euclidean_distances(df_song[features],centroids[features])
    # return the cluster with the closest centroid
    return int(centroids.loc[np.argmin(dist,axis=-1),"clusterId"])