"""
Thhe module contains methods that
    1. searches for a song"s features by calling Spotify"s API
    2. returns the cluster of which the centroid is closest to the song in terms of cosine similarity
    3. returns the top N closest (in terms of cosine similarity) songs to a given song in a given cluster
"""
import logging
import os
from typing import Union

import numpy as np
import pandas as pd
import spotipy  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
from spotipy.oauth2 import SpotifyClientCredentials  # type: ignore

logger = logging.getLogger(__name__)


def establish_api() -> spotipy.client.Spotify:
    """Establish an Spotify API client

    Returns: an spotify api agent

    """
    cid = os.getenv("SPOTIPY_CLIENT_ID")
    secret = os.getenv("SPOTIPY_CLIENT_SECRET")

    # establish a spotify api service

    client_credentials_manager = SpotifyClientCredentials(
        client_id=cid, client_secret=secret)

    try:
        sp_api = spotipy.Spotify(
            client_credentials_manager=client_credentials_manager)
    except spotipy.oauth2.SpotifyOauthError as err:
        logger.error(
            "The configured client id and secret does not match. Please check them again!")
        raise err
    return sp_api


def search_on_spotify(query: str) -> dict:
    """A helper function that searches on Spotify with a given query

    Arguments:
        query -- the query to search on Spotify

    Returns:
        a dictionary containing search results
    """
    try:
        sp_api = establish_api()
        search_results = sp_api.search(query, limit=1, offset=0, type="track")
        logger.info("The song %s has been searched", query)
    except spotipy.oauth2.SpotifyOauthError as err:
        logger.error(
            "The configured client id and secret does not match. Please check them again!")
        raise err
    return search_results


def form_query(song_name: str, artist_name: str) -> str:
    """A helper function that combines song name and artist name into a spotify query

    Arguments:
        song_name -- the song name to be searched for
        artist_name -- the artist of the song to be searched for

    Raises:
        KeyError -- missing all fields
    Returns:
        a query string
    """
    # create two flags to indicate valid song_name or artist_name
    f_song_name = (song_name is not None) and (song_name != "")
    f_artist_name = (artist_name is not None) and (artist_name != "")
    # if the user types in both fields, we are gucci
    if f_song_name and f_artist_name:
        song_to_search = f"artist:{artist_name} track:{song_name}"
    # if either field but not both is missing, just search for whatever you have
    elif f_song_name and not f_artist_name:
        song_to_search = f"track:{song_name}"
    elif f_artist_name and not f_song_name:
        song_to_search = f"artist:{artist_name}"
    # if none is provided, raise an error
    else:
        logger.error(
            "Please provide at least one field. Song name or artist name.")
        raise KeyError("Fields missing")
    return song_to_search


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
    sp_api = establish_api()
    # form the query
    song_to_search = form_query(song_name, artist_name)

    # use spotipy to search for the song
    search_results = search_on_spotify(song_to_search)

    # in case the search results returned nothing
    if search_results["tracks"]["total"] > 0:
        song_id = search_results["tracks"]["items"][0]["id"]
        res = [sp_api.audio_features(tracks=song_id)[0]]
        # transform list of song features to a dataframe that contains one row - pertaining to the song
        df = pd.DataFrame.from_records(res, index=[0]*len(res))
        logger.info("%d song has been returned.", len(res))
    else:
        logger.info("The search returns nothing. The song you are looking for does not currently exist in Spotify!"
                    "Please search for a new song.")
        df = pd.DataFrame([])

    return df


def valid_features(sub_features: list[str], all_features: list[str]) -> bool:
    """A helper function that returns sub_features only if all of them are valid

    Arguments:
        sub_features -- sub_feature lists to go through
        all_features -- the original complete list of features

    Returns:
        a boolean indicating whether the sub_features are all valid

    Raises:
        TypeError: given list contains other than string types
    """
    if isinstance(sub_features, str):
        sub_features = [sub_features]
    if len(sub_features) == 0:
        logger.info("No features are selected for sub_features")
        return True
    if len(sub_features) > len(all_features):
        return False
    # check if the types of the two lists are identical
    if not isinstance(sub_features[0], str) or not isinstance(all_features[0], str):
        raise TypeError("Given list contains types other than string")
    # check if the features provided exist
    for fea in sub_features:
        if not fea in all_features:
            return False
    return True


def get_closest_cluster(df_song: pd.DataFrame,
                        centroids: pd.DataFrame,
                        features: list[str]) -> Union[list[int], int]:
    """Based on the selected features, choose the clusters with a centroid closest to the song provided.

    Arguments:
        df_song -- dataframe of songs including their features
        centroids -- the datafram containing info about centroids
        features -- the features to use when calculating distance

    Raises:
        KeyError -- Nonexisting features

    Returns:
        if df_song contains only one song, returns the closest clusterId
        if df_song contains many songs, returns the closest cluserId to each song in a list
    """
    if not (valid_features(features, df_song.columns) and valid_features(features, centroids.columns)):
        logger.error("The features selected is not an available song feature!")
        raise KeyError("Nonexisting feature")

    # calculate distance between the song and each cluster
    dist = cosine_similarity(df_song[features], centroids[features])
    logger.debug(dist)

    # if only one song, return the cluster with the closest centroid
    if df_song.shape[0] == 1:
        return int(centroids.loc[np.argmax(dist, axis=-1), "clusterId"])
    return centroids.loc[np.argmax(dist, axis=-1), "clusterId"].to_list()


def get_top_n_closest_song(df_song: pd.DataFrame, df_anime: pd.DataFrame,
                           features: list[str], top_n: int,
                           cluster_id: int) -> list[dict]:
    """Select the top N closest song to the given song

    Arguments:
        df_song -- A dataframe containing the features of one song
        df_anime -- A dataframe containing all possible anime songs with cluster ids
        features -- A list of features to compute distance on
        top_n -- number of songs to find
        cluster_id -- the cluster to look into

    Returns:
        a list of records containing info about the found closest songs
    """
    if not (valid_features(features, df_song.columns) and valid_features(features, df_anime.columns)):
        logger.error("The features selected is not an available song feature!")
        raise KeyError("Nonexisting feature")

    # select the rows that correspond to the given cluster id
    df_anime = df_anime[df_anime["clusterId"]
                        == cluster_id].reset_index(drop=True)
    dist = cosine_similarity(df_song[features], df_anime[features])
    inds = np.argpartition(dist, -top_n, axis=-1)

    return df_anime.loc[inds[0][-top_n:]].to_dict("records")
