"""
A collection of tools that LLMs can use to solve problems.

A tool is a function with a standard description.
"""
from collections.abc import Callable
import sqlite3

from ..db.sqlite_utils import get_db_creation_sql


# class SQLiteQueryTool:
#     """Run a sqlite query against a given database connection."""

#     def __init__(self, db_connection: sqlite3.Connection):
#         self._connection = db_connection

#         self._description = \
#         'Function for running a sqlite query against a databse.'
#         self._params = {
#             'query': {
#                 'description': 'The query to run',
#                 'typ': str,
#                 'required': True
#             }
#         }

#     def __call__(self, query: str) -> list[tuple]:
#         """Run the query against the database connection.
        
#         Returns a list of tuples"""
#         cursor = self._connection.cursor()
#         results = cursor.execute(query).fetchall()
#         return results


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
# tool1 = SQLiteQueryTool(conn)
tool2 = make_sqlite_query_tool(conn)
