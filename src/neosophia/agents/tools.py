""" Tools the Agent can use """
import json
import sqlite3

from typing import Any, Dict, List

import pandas as pd

from pandasql import load_meat, sqldf


def _execute_pandas_query(
        query: str, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Executes a SQL query against a Pandas DataFrame. In the SQL command refer
    to the dataframe as table named `df_table`.

    Args:
        query (str): The SQL query to be executed.
        dataframe: pd.DataFrame: A dataframe to run the query on

    Returns:
        result (pd.DataFrame): The resulting dataframe after executing the
        query.
    """
    if query[0] == "'" or query[0] == '"':
        query = query[1:-1]
    return sqldf(query, {'df_table': dataframe})


def execute_pandas_query(
        query: str, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Executes a SQL query against a Pandas DataFrame. In the SQL command refer
    to the dataframe as table named `df_table`.

    Args:
        query (str): The SQL query to be executed.
        dataframe: pd.DataFrame: A dataframe to run the query on

    Returns:
        result (pd.DataFrame): The resulting dataframe after executing the
        query.
    """
    if query.lower().startswith('alter table'):
        if 'add column' in query.lower():
            column_details = query.split('ADD COLUMN')[1].strip()
            column_name, column_type = [x.strip() for x in column_details.split()]

            if column_type.lower() == 'int':
                dataframe[column_name] = 0
            elif column_type.lower() == 'float':
                dataframe[column_name] = 0.0
            elif column_type.lower() == 'text' or column_type.lower() == 'string':
                dataframe[column_name] = ''
            else:
                raise ValueError('Unsupported column type')

            return dataframe
        else:
            raise ValueError('Only ADD COLUMN operation is supported in ALTER TABLE')
    else:
        if query[0] == "'" or query[0] == '"':
            query = query[1:-1]
        return sqldf(query, {'df_table': dataframe})


def create_dictionary(dict_string):
    """ """
    return json.loads(dict_string)


def execute_query(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    """
    This function executes a given SQL query on a specified sqlite3 database
    connection and returns the results as a pandas DataFrame.

    Args:
        conn (sqlite3.Connection): The connection to the sqlite3 database where
        the query will be executed.
        query (str): The SQL query to be executed. If the query is enclosed in
        quotes, they will be stripped before execution.

    Returns:
        result (pandas.DataFrame): The result of the executed SQL query
        returned as a pandas DataFrame.
    """

    if query[0] == "'" or query[0] == '"':
        query = query[1:-1]
    return pd.read_sql_query(query, conn)


def execute_query_with_variables(
        conn: sqlite3.Connection,
        query: str,
        variables: str) -> pd.DataFrame:
    """
    This function executes a SQL query with variables and returns the result as
    a pandas DataFrame.

    query = "SELECT guid FROM products WHERE account_number = :account_number"
    variables = {'account_number': 1251}
    result = execute_query_with_variables(conn, query, variables)

    Args:
        conn (sqlite3.Connection): A connection object representing the
        database connection.
        query (str): The SQL query to execute.
        variables (Dict[str, Any]): A dictionary containing the variables to be
        used in the query.

    Returns:
        result (pd.DataFrame): A pandas DataFrame representing the result of
        the executed query
    """
    variables = json.loads(variables)
    if query[0] == "'" or query[0] == '"':
        query = query[1:-1]
    return pd.read_sql_query(query, conn, params=variables)


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

