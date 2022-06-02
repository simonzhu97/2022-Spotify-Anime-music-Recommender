""" Test the functions in search_songs.py
form_query
valid_features
get_closest_cluster
get_top_n_closest_song
"""

import pandas as pd
import pytest
import src.search_songs as search

# Define expected input dataframe
df_in_values = [[0.627, "怪物"],
                [0.585, "廻廻奇譚"],
                [0.561, "One Last Kiss"]]

df_in_index = list(range(3))

df_in_columns = ["danceability", "title"]

df_in = pd.DataFrame(df_in_values, index=df_in_index, columns=df_in_columns)


def test_form_query():
    """Unit test - happy path - form_query()
    """
    # Define expected output, df_true
    true_out = "artist:simon zhu track:wow wow"

    # Compute test output
    test_out = search.form_query("wow wow", "simon zhu")

    # Test that the true and test are the same
    assert true_out == test_out


def test_form_query_no_input():
    """Unit test - unhappy path - form_query()
    """
    with pytest.raises(KeyError):
        search.form_query(None, "")


def test_valid_features():
    """Unit test - happy path - valid_features()
    """
    # Define expected output, df_true
    true_out = True
    all_columns = ["danceability", "energy", "key", "loudness", "speechiness", "acousticness", "instrumentalness",
                   "liveness", "valence", "tempo"]
    sub_columns = ["danceability", "key"]

    # Compute test output
    test_out = search.valid_features(sub_columns, all_columns)

    # Test that the true and test are the same
    assert true_out == test_out


def test_valid_features_not_string():
    """Unit test - unhappy path - valid_features()
    """
    # when any of the columns for transformation does not exist
    int_in = [1, 2, 3]
    all_columns = ["danceability", "energy", "key", "loudness", "speechiness", "acousticness", "instrumentalness",
                   "liveness", "valence", "tempo", "title", "duration", "track_uri"]

    with pytest.raises(TypeError):
        search.valid_features(int_in, all_columns)


def test_get_closest_cluster():
    """Unit test - happy path - get_closest_cluster()
    """
    centroids_cols = ["clusterId", "danceability"]
    centroids = pd.DataFrame([[0, 0.627], [1, 0.573]],
                             columns=centroids_cols)
    features = ["danceability"]

    # true output value
    true_out = [0, 0, 0]
    # test output
    test_out = search.get_closest_cluster(df_in, centroids, features)

    assert true_out == test_out


def test_get_closest_cluster_bad_features():
    """Unit test - unhappy path - get_closest_cluster()
    """
    centroids_cols = ["clusterId", "danceability"]
    centroids = pd.DataFrame([[0, 0.627], [1, 0.573]],
                             columns=centroids_cols)
    features = ["key"]

    with pytest.raises(KeyError):
        search.get_closest_cluster(df_in, centroids, features)


def test_get_top_n_closest_song():
    """Unit test - happy path - get_top_n_closest_song()
    """
    anime_cols = ["danceability", "title", "clusterId"]
    df_anime = pd.DataFrame([[0.5, "Song_1", 0], [0.1, "Song_2", 0], [0.3, "Song_3", 0],
                             [0.3, "Song_4", 1]],
                            columns=anime_cols)
    features = ["danceability"]

    true_out = [{"danceability": 0.1, "title": "Song_2", "clusterId": 0},
                {"danceability": 0.3, "title": "Song_3", "clusterId": 0}]

    test_out = search.get_top_n_closest_song(
        df_in.iloc[[0]], df_anime, features, 2, 0)

    assert true_out == test_out


def test_get_top_n_closest_song_bad_features():
    """Unit test - unhappy path - get_top_n_closest_song()
    """
    anime_cols = ["danceability", "title", "clusterId"]
    df_anime = pd.DataFrame([[0.5, "Song_1", 0], [0.1, "Song_2", 0], [0.3, "Song_3", 0],
                             [0.3, "Song_4", 1]],
                            columns=anime_cols)
    features = ["key"]
    with pytest.raises(KeyError):
        search.get_top_n_closest_song(
            df_in.iloc[[0]], df_anime, features, 2, 0)
