""" Test the functions in preprocessing.py
validate_features
clean
featurize
"""

import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
import src.preprocessing as prep

# Define expected input dataframe
df_in_values = [[0,0.627,0.824,1,-3.419,1,0.118,0.0699,6.57e-05,0.31,0.774,169.935,"audio_features",
                 "str","str","uri","uri",206000,4,"怪物"],
                [0,0.585,0.91,5,-3.703,0,0.0917,0.0138,0.0,0.19,0.602,92.591,"audio_features",
                 "str","str","uri","uri",221426,4,"廻廻奇譚"],
                [0,0.561,0.667,8,-8.519,0,0.0652,0.0539,0.0,0.0797,0.383,100.047,"audio_features",
                 "str","str","uri","uri",252027,4,"One Last Kiss"]]

df_in_index = list(range(3))

df_in_columns = ["Unnamed: 0", "danceability", "energy", "key", "loudness", "mode",
       "speechiness", "acousticness", "instrumentalness", "liveness",
       "valence", "tempo", "type", "id", "uri", "track_href", "analysis_url",
       "duration_ms", "time_signature", "name"]

df_in = pd.DataFrame(df_in_values, index=df_in_index, columns=df_in_columns)


def test_validate_features():
    """Unit test - happy path - validate_features()
    """
    # Define expected output, df_true
    true_out = True
    feature_columns = ["danceability","energy","key","loudness","speechiness","acousticness","instrumentalness",
                       "liveness","valence","tempo"]

    # Compute test output
    test_out = prep.validate_features(df_in,feature_columns)

    # Test that the true and test are the same
    assert true_out == test_out

def test_validate_features_not_df():
    """Unit test - unhappy path - validate_features()
    """
    # when any of the columns for transformation does not exist
    nonexist_in = pd.Series(["1","2"])
    feature_columns = ["danceability","energy","key","loudness","speechiness","acousticness","instrumentalness",
                       "liveness","valence","tempo","title","duration","track_uri"]
    
    with pytest.raises(TypeError):
        prep.validate_features(nonexist_in,feature_columns)

def test_clean():
    """Unit test - happy path - clean()
    """
    # Define expected output, df_true
    df_true = pd.DataFrame([[0.627,"怪物"],[0.585,"廻廻奇譚"],[0.561,"One Last Kiss"]],
                           columns=["danceability","title"],
                           index=[0,1,2])
    feature_columns = ["danceability","title"]
    col_mapper = {"name":"title", "duration_ms":"duration", "uri":"track_uri"}

    # Compute test output
    df_test = prep.clean(df_in,col_mapper=col_mapper,features=feature_columns)

    # Test that the true and test are the same
    pd.testing.assert_frame_equal(df_true, df_test)

def test_clean_col_map_not_dict():
    """Unit test - unhappy path - clean()
    """
    # Define expected output, df_true
    df_true = df_in
    feature_columns = ["danceability","title"]
    col_mapper = ["something"]

    # Compute test output
    df_test = prep.clean(df_in,col_mapper,feature_columns)

    # Test that the true and test are the same
    pd.testing.assert_frame_equal(df_true, df_test)

def test_featurize():
    """Unit test - happy path - featurize()
    """
    # Define expected output, df_true
    df_true = pd.DataFrame([[1.31982404,"怪物"],[-0.21997067,"廻廻奇譚"],[-1.09985336,"One Last Kiss"]],
                           columns=["danceability","name"],
                           index=[0,1,2])
    feature_columns = ["danceability","name"]

    std_scale = StandardScaler()
    std_scale.fit(pd.DataFrame([[0.627],[0.585],[0.561]],
                               columns=["danceability"]))

    # Compute test output
    df_test,std_test = prep.featurize(df_in,features=feature_columns)

    # Test that the true and test are the same
    pd.testing.assert_frame_equal(df_true, df_test)
    assert std_test.feature_names_in_ == np.array(["danceability"])
    assert std_test.mean_ == np.array([0.5910000000000001])
    assert std_test.scale_ == np.array([0.027276363393971693])


def test_featurize_not_df():
    """Unit test - unhappy path - featurize()
    """
    # Define expected output, df_true
    nonexist_in = pd.Series(["1","2"])
    feature_columns = ["danceability","title"]

    with pytest.raises(TypeError):
        prep.featurize(nonexist_in,feature_columns)