""" Test the functions in evaluate_model.py
evaluate()
assign_labels()
assign_new_labels()
"""

import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.evaluate_model import assign_labels, assign_new_labels, evaluate

# Define expected input dataframe
df_in_values = [[0.627, 1],
                [0.585, 1],
                [0.561, 0]]

df_in_index = list(range(3))

df_in_columns = ["danceability", "clusterId"]

df_in = pd.DataFrame(df_in_values, index=df_in_index, columns=df_in_columns)


def test_assign_labels():
    """Unit test - happy path - assign_labels()
    """
    # define input values
    df_features = pd.DataFrame([[0.627], [0.585], [0.561]],
                               columns=["danceability"])
    model = KMeans(n_clusters=2, random_state=42)
    model.fit(df_features)

    # define true output
    df_true = pd.DataFrame([[0.627, 1], [0.585, 0], [0.561, 0]],
                           columns=["danceability", "clusterId"])

    # test output
    df_test = assign_labels(df_features, model)
    # check_dtypes=False to avoid problems of int32 and int64
    pd.testing.assert_frame_equal(df_true, df_test, check_dtype=False)


def test_assign_labels_unmatch_columns():
    """Unit test - unhappy path - assign_labels()
    """
    # define input values
    df_features = pd.DataFrame([[0.627], [0.585], [0.561]],
                               columns=["key"])
    model = KMeans(n_clusters=2, random_state=42)
    model.fit(df_in)

    with pytest.raises(KeyError):
        assign_labels(df_features, model)


def test_assign_new_labels():
    """Unit test - happy path - assign_new_labels()
    """
    # define input values
    df_data = pd.DataFrame([[0.627], [0.585], [0.561]],
                           columns=["danceability"])
    # define scalar to use
    scalar = StandardScaler()
    scalar.fit(df_data)
    # define the KMeans model
    model = KMeans(n_clusters=2, random_state=42)
    model.fit(pd.DataFrame(scalar.transform(df_data),
                           columns=["danceability"]))

    # define other input features
    clean_features = ["danceability"]
    features = ["danceability"]
    col_mapper = {}
    df_songs = pd.DataFrame([[0.1], [0.9], [1.0]],
                            columns=["danceability"])

    # define true output
    df_true = pd.DataFrame([[-18.001, 0], [11.328, 1], [14.995, 1]],
                           columns=["danceability", "clusterId"])

    # test output
    df_test = assign_new_labels(
        df_songs, scalar, model, clean_features, col_mapper, features)
    # test if df_true and df_test are the same
    pd.testing.assert_frame_equal(
        df_true, df_test, check_dtype=False, atol=1e-3, rtol=1e-3)


def test_assign_new_labels_no_songs():
    """Unit test - unhappy path - assign_new_labels()
    """
    # define input values
    df_data = pd.DataFrame([[0.627], [0.585], [0.561]],
                           columns=["danceability"])
    # define scalar to use
    scalar = StandardScaler()
    scalar.fit(df_data)
    # define the KMeans model
    model = KMeans(n_clusters=2, random_state=42)
    model.fit(pd.DataFrame(scalar.transform(df_data),
                           columns=["danceability"]))

    # define other input features
    clean_features = ["danceability"]
    features = ["danceability"]
    col_mapper = {}
    df_songs = pd.DataFrame([])

    with pytest.raises(ValueError):
        assign_new_labels(df_songs, scalar, model,
                          clean_features, col_mapper, features)


def test_evaluate():
    """Unit test - happy path - evaluate()
    """
    # Define expected output, df_true
    true_out = "Silhoutee score: -0.021645021645016176"

    # Compute test output
    test_out = evaluate(df_in)

    # Test that the true and test are the same
    assert true_out == test_out


def test_evaluate_not_df():
    """Unit test - unhappy path - evaluate()
    """
    df_bad = pd.DataFrame([[0.627, 1],
                           [0.585, 1],
                           [0.561, 0]], columns=["danceability", "label"])

    with pytest.raises(KeyError):
        evaluate(df_bad)
