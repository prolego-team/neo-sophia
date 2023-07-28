"""
A collection of tools that LLMs can use to solve problems.

A tool is a function with a standard description.
"""
from collections.abc import Callable
import sqlite3


def make_sqlite_query_tool(db_connection: sqlite3.Connection) -> Callable:
    """Run a sqlite query against a given database connection."""


    description = 'Run a sqlite query against a databse.'
    params = {
        'query': {
            'description': 'The query to run',
            'typ': str,
            'required': True
        }
    }


    def sqlite_tool(query: str) -> str:
        cursor = db_connection.cursor()
        try:
            results = str(cursor.execute(query).fetchall())
        except Exception as error:
            results = f'Query failed: {error}'
        return results

    return sqlite_tool, (description, params)
