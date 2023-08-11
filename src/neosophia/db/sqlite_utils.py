""" Tools for interacting with SQLite """
import csv
import sqlite3

from typing import Any, List

import pandas as pd


def get_conn(db_file: str) -> sqlite3.Connection:
    return sqlite3.connect(db_file)


def execute_query(conn: sqlite3.Connection, query: str) -> List[Any]:
    """ Executes an SQL query """
    cursor = conn.cursor()
    print(f'QUERY:|{query}|')
    if query[0] == "'" or query[0] == '"':
        query = query[1:-1]
    print(f'QUERY:|{query}|')
    return cursor.execute(query).fetchall()


def get_table_schema(
        conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    """Get a description of a table into a pandas dataframe."""
    query = f"PRAGMA table_info({table_name});"
    return pd.read_sql_query(query, conn)


def get_db_creation_sql(conn: sqlite3.Connection) -> str:
    """Construct a description of the DB schema for the LLM by retrieving the
    CREATE commands used to create the tables."""
    cursor = conn.cursor()

    query = "SELECT sql FROM sqlite_master WHERE type='table'"
    results = cursor.execute(query).fetchall()
    results = [col[0] for col in results]
    schema_description = '\n'.join(results)

    return schema_description


def get_tables_from_db(conn: sqlite3.Connection) -> str:
    """ Get a list of table names from the database """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [x[0] for x in cursor.fetchall()]


def create_database_from_csv(conn: sqlite3.Connection, csv_file: str, table_name: str) -> None:
    """Create database table from a CSV file."""

    # Connect to the SQLite database
    cursor = conn.cursor()

    # Read the CSV file and insert data into the database
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)

        # Create the table in the database based on the CSV header
        columns = ", ".join(header)
        drop_table_query = f'DROP TABLE IF EXISTS {table_name}'
        cursor.execute(drop_table_query)

        create_table_query = f'CREATE TABLE {table_name} ({columns})'
        cursor.execute(create_table_query)

        # Insert rows into the table
        insert_query = f"INSERT INTO {table_name} VALUES ({', '.join(['?'] * len(header))})"
        for row in reader:
            cursor.execute(insert_query, row)

    # Commit the changes
    conn.commit()

    print("Database created and data inserted successfully.")
