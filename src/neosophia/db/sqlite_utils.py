""" Tools for interacting with SQLite """
import csv

import pandas as pd


def get_table_schema(conn, table_name):
    """ """
    cursor = conn.cursor()
    query = f"PRAGMA table_info({table_name});"
    return pd.read_sql_query(query, conn)


def create_database_from_csv(conn, csv_file, db_file, table_name):
    """ """

    # Connect to the SQLite database
    cursor = conn.cursor()

    # Read the CSV file and insert data into the database
    with open(csv_file, mode="r") as file:
        reader = csv.reader(file)
        header = next(reader)

        # Create the table in the database based on the CSV header
        columns = ", ".join(header)
        drop_table_query = f"DROP TABLE IF EXISTS {table_name}"
        cursor.execute(drop_table_query)

        create_table_query = f"CREATE TABLE {table_name} ({columns})"
        cursor.execute(create_table_query)

        # Insert rows into the table
        insert_query = f"INSERT INTO {table_name} VALUES ({', '.join(['?'] * len(header))})"
        for row in reader:
            cursor.execute(insert_query, row)

    # Commit the changes
    conn.commit()

    print("Database created and data inserted successfully.")

