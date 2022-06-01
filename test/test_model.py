""" Test the functions in model.py
get_train_data
get_model
"""

import pandas as pd
import pytest
import src.model as mod
from sklearn.cluster import KMeans

# Define expected input dataframe
df_in_values = [[0, 0.627, 0.824, 1, -3.419, 1, 0.118, 0.0699, 6.57e-05, 0.31, 0.774, 169.935, "audio_features",
                 "str", "str", "uri", "uri", 206000, 4, "怪物"],
                [0, 0.585, 0.91, 5, -3.703, 0, 0.0917, 0.0138, 0.0, 0.19, 0.602, 92.591, "audio_features",
                 "str", "str", "uri", "uri", 221426, 4, "廻廻奇譚"],
                [0, 0.561, 0.667, 8, -8.519, 0, 0.0652, 0.0539, 0.0, 0.0797, 0.383, 100.047, "audio_features",
                 "str", "str", "uri", "uri", 252027, 4, "One Last Kiss"]]

df_in_index = list(range(3))

df_in_columns = ["Unnamed: 0", "danceability", "energy", "key", "loudness", "mode",
                 "speechiness", "acousticness", "instrumentalness", "liveness",
                 "valence", "tempo", "type", "id", "uri", "track_href", "analysis_url",
                 "duration_ms", "time_signature", "name"]

df_in = pd.DataFrame(df_in_values, index=df_in_index, columns=df_in_columns)


def test_get_train_data():
    """Unit test - happy path - get_train_data()
    """
    # Define expected output, df_true
    df_true = pd.DataFrame([[0.627], [0.585], [0.561]],
                           columns=["danceability"])
    feature_columns = ["danceability", "name"]

    # Compute test output
    df_test = mod.get_train_data(df_in, feature_columns)

    # Test that the true and test are the same
    pd.testing.assert_frame_equal(df_true, df_test)


def test_get_train_data_not_df():
    """Unit test - unhappy path - get_train_data()
    """
    # when any of the columns for transformation does not exist
    nonexist_in = pd.Series(["1", "2"])
    feature_columns = ["danceability", "name"]

    with pytest.raises(TypeError):
        mod.get_train_data(nonexist_in, feature_columns)

def test_get_model():
    """Unit test - happy path - get_model()
    """
    #model
    feature_columns = ['danceability']
    fit = mod.get_model(df_in,feature_columns,2,42)

    assert isinstance(fit, KMeans)

def test_get_model_not_valid_data():
    """Unit test - unhappy path - get_model()
    """
    # when any of the columns for transformation does not exist
    nonexist_in = pd.Series(["1", "2"])
    feature_columns = ["danceability"]

    with pytest.raises(TypeError):
        mod.get_model(nonexist_in, feature_columns,2,42)
    