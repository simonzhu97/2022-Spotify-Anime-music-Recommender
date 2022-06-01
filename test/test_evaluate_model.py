""" Test the functions in evaluate_model.py
evaluate()
"""

import pandas as pd
import pytest
from src.evaluate_model import evaluate

# Define expected input dataframe
df_in_values = [[0.627, 1],
                [0.585, 1],
                [0.561, 0]]

df_in_index = list(range(3))

df_in_columns = ["danceability", "clusterId"]

df_in = pd.DataFrame(df_in_values, index=df_in_index, columns=df_in_columns)


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
                [0.561, 0]],columns=['danceability','label'])

    with pytest.raises(KeyError):
        evaluate(df_bad)