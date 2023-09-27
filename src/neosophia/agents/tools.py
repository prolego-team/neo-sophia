""" Tools the Agent can use """
import json
import sqlite3

from typing import Any, Dict, List, Union

import pandas as pd

from pandasql import load_births, load_meat, sqldf

from neosophia.agents.utils import strip_quotes


def iloc(
        df: pd.DataFrame,
        start: int,
        end: int) -> Union[pd.DataFrame, pd.Series]:
    """
    This function returns a subset of a DataFrame using integer-based indexing.

    Args:
        df (pd.DataFrame): The DataFrame to be sliced.
        start (int): The starting index of the subset.
        end (int): The ending index of the subset.

    Returns:
        return_arg_1 (Any): The subset of the DataFrame from start to end.
    """
    return df.iloc[start:end]


def get_dataframe_length(df: pd.DataFrame) -> int:
    """
    Gets the length of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame for which to get the length.

    Returns:
        length (int): The number of rows in the DataFrame.
    """
    return df.shape[0]


def merge_dataframes(
        left: Union[pd.DataFrame, pd.Series],
        right: Union[pd.DataFrame, pd.Series],
        how: str='inner',
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        sort=False):

    """
    Args:
        left (pd.DataFrame or pd.Series):
        right (pd.DataFrame or pd.Series): Object to merge with.
        how (str): {‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’}, default
        ‘inner’.  Type of merge to be performed.
            - left: use only keys from left frame, similar to a SQL left outer
              join; preserve key order.
            - right: use only keys from right frame, similar to a SQL right
              outer join; preserve key order.
            - outer: use union of keys from both frames, similar to a SQL full
              outer join; sort keys lexicographically.
            - inner: use intersection of keys from both frames, similar to a
              SQL inner join; preserve the order of the left keys.
            - cross: creates the cartesian product from both frames, preserves
              the order of the left keys.
        on (str): Column or index level names to join on. These must
        be found in both DataFrames. If on is None and not merging on indexes
        then this defaults to the intersection of the columns in both
        DataFrames.
        left_on (str): Column or index level names to
        join on in the left DataFrame. Can also be an array or list of arrays
        of the length of the left DataFrame. These arrays are treated as if
        they are columns.
        right_on (str): Column or index level names to
        join on in the right DataFrame. Can also be an array or list of arrays
        of the length of the right DataFrame. These arrays are treated as if
        they are columns.
        left_index (bool): Use the index from the left DataFrame as the join
        key(s). If it is a MultiIndex, the number of keys in the other
        DataFrame (either the index or a number of columns) must match the
        number of levels.
        right_index (bool): Use the index from the right DataFrame as the join
        key. Same caveats as left_index.
        sort (bool): Sort the join keys lexicographically in the result
        DataFrame. If False, the order of the join keys depends on the join
        type (how keyword).
    """
    return pd.merge(
        left,
        right,
        how=how,
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        sort=sort)


def execute_pandas_query(
        query: str, **kwargs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Executes a SQL query against an arbitrary number of Pandas DataFrames. The
    kwargs passed in must match exactly to the table names in the query.

    Example call:
        query_str = "SELECT * FROM table1, table2;"
        execute_pandas_query(query_str, table1=table1, table2=table2)

    Args:
        query (str): The SQLite query to be executed.
        **kwargs: Keyword arguments containing Pandas DataFrames

    Returns:
        result (pd.DataFrame): The resulting dataframe after executing the
        query.
    """
    if query.lower().startswith('alter table'):
        if 'add column' in query.lower():
            column_details = query.split('ADD COLUMN')[1].strip()
            column_name, column_type = [
                x.strip() for x in column_details.split()
            ]

            if column_type.lower() == 'int':
                dataframe[column_name] = 0
            elif column_type.lower() == 'float':
                dataframe[column_name] = 0.0
            elif column_type.lower() == 'text' or column_type.lower() == 'string':
                dataframe[column_name] = ''
            else:
                raise ValueError(f'Unsupported column type: {column_type}')

            return dataframe
        else:
            raise ValueError(
                'Only ADD COLUMN operation is supported in ALTER TABLE')
    else:
        query = strip_quotes(query)
        return sqldf(query, kwargs)


def get_max_values(df: pd.DataFrame) -> pd.Series:
    """
    Get the maximum value of each column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.Series: A series containing the maximum value of each column.
    """
    return df.max()


def get_min_values(df: pd.DataFrame) -> pd.Series:
    """
    Get the maximum value of each column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.Series: A series containing the maximum value of each column.
    """
    return df.min()


def get_std(df: pd.DataFrame) -> pd.Series:
    """
    Get the standard deviation

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.Series: A series containing the standard deviation
    """
    axis = 0
    numeric_only = True
    return df.std(axis=axis, numeric_only=numeric_only)


def dataframe_concat(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        concat_axis: int = 0) -> pd.DataFrame:
    """
    Concatenates two dataframes along the specified axis.

    Args:
        df1 (pd.DataFrame): The first dataframe to be concatenated.
        df2 (pd.DataFrame): The second dataframe to be concatenated.
        concat_axis (int): The axis along which the dataframes will be
        concatenated. 0 indicates row-wise (default) and 1 indicates
        column-wise.

    Returns:
        result_df (pd.DataFrame): A new dataframe that results from the
        concatenation of df1 and df2 along the specified axis.
    """
    result_df = pd.concat([df1, df2], axis=concat_axis)
    return result_df


def dataframe_intersection(
        column1_name: str,
        column2_name: str,
        join_type: str,
        df1: pd.DataFrame,
        df2: pd.DataFrame) -> pd.DataFrame:
    """
    Performs an intersection operation between two dataframes based on a
    specified column name and join type.

    Args:
        column1_name (str): The column name in the first dataframe on which the
        join operation will be performed.
        column2_name (str): The column name in the second dataframe on which
        the join operation will be performed.
        join_type (str): The type of join operation to be performed on the two
        dataframes. This can be any valid pandas join type ('left', 'right',
        'outer', 'inner').
        df1 (pd.DataFrame): The first dataframe to be joined.
        df2 (pd.DataFrame): The second dataframe to be joined.

    Returns:
        result_df (pd.DataFrame): A new dataframe that results from the join of
        df1 and df2 based on the specified column names and join type.
    """

    args = [df1, df2]

    # Start with the first dataframe
    result_df = args[0]

    # Iteratively join with the other dataframes
    for idx, df in enumerate(args[1:]):
        result_df = result_df.join(
            df.set_index(column2_name),
            how=join_type,
            on=column1_name,
            rsuffix=str(idx + 1))

    return result_df

