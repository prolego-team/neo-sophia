"""
A collection of tools that LLMs can use to solve problems.

A tool is a function with a standard description.
"""
from collections.abc import Callable
import sqlite3

from ..db.sqlite_utils import get_db_creation_sql


def make_sqlite_query_tool(db_connection: sqlite3.Connection) -> Callable:
    """Run a sqlite query against a given database connection."""

    description = 'Function for running a sqlite query against a databse.'
    params = {
        'query': {
            'description': 'The query to run',
            'typ': str,
            'required': True
        }
    }

    def sqlite_tool(query: str):
        cursor = db_connection.cursor()
        results = cursor.execute(query).fetchall()
        return results

    return sqlite_tool, (description, params)


conn = sqlite3.connect('synthbank.db')
print(get_db_creation_sql(conn))
tool = make_sqlite_query_tool(conn)
