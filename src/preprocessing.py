"""
Clean the raw data downloaded from s3 -- choose relevant features for clustering
Featurize the cleaned data - standard scaling

"""
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def read_from_local(path: str, **kwargs) -> pd.DataFrame:
    """Read a csv file from a local path

    Arguments:
        path -- input path to the csv file

    Returns:
        a pandas dataframe
    """
    try:
        res = pd.read_csv(path, **kwargs)
        logger.info("CSV file read from %s", path)
    except pd.errors.ParserError as err:
        logger.error("The given file path does not contain a csv file")
        raise err
    except FileNotFoundError as err:
        logger.error("No file exists at the given path %s", path)
        raise err
    except Exception as err:
        logger.error(err)

    return res


def validate_features(df: pd.DataFrame, features: list[str]) -> bool:
    """A helper function that checks whether the given features exist in the dataframe

    Arguments:
        df -- the given dataframe
        features -- the features to check

    Returns:
        True/False
    """
    for fea in features:
        if fea not in df.columns:
            return False
    return True


def clean(df_in: pd.DataFrame, col_mapper: dict, cols_to_drop: list[str]) -> pd.DataFrame:
    """Clean the raw dataset by changing column names and drop some unnecessary columns

    Arguments:
        df -- raw data in the dataframe format
        col_mapper -- a dictionary containing column names to change
        cols_to_drop -- list of columns to drop from the dataframe

    Returns:
        a cleaned dataframe
    """
    try:
        df_new = df_in.drop(cols_to_drop, axis=1)
    except KeyError as err:
        logger.error("The column to drop does not exist")
        raise err
    # ensure the col_mapper is indeed a dictionary
    if isinstance(col_mapper, dict):
        df_new = df_new.rename(columns=col_mapper)
    else:
        logger.error("col_mapper needs to be a dictionary, but now it is a %s",
                     type(col_mapper))
        logger.error("No columns are renamed.")
    return df_new


def featurize(df_in: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Generate features from the cleaned dataset

    Arguments:
        df_in -- the cleaned dataframe
        features -- list of features to include in the final dataframe

    Returns:
        a dataframe with scaled columns
    """
    # check if features exist`
    if validate_features(df_in, features):
        df_in = df_in[features]
    else:
        logger.error(
            "The given features %s do not exist in the dataframe", features)
        logger.error("The original whole dataframe is selected.")

    # conduct scaling on numerical columns
    std_scale = StandardScaler()
    df_num = df_in.select_dtypes(include=np.number)
    df_rest = df_in.select_dtypes(exclude=np.number)
    df_scale = pd.DataFrame(std_scale.fit_transform(df_num),
                            columns=df_num.columns)
    # combine numerical and non-numerical columns
    df_fin = pd.concat([df_scale, df_rest], axis=1)

    return df_fin
