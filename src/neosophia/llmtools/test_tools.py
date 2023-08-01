import sqlite3

from neosophia.llmtools import tools

def test_make_sqlite_query_tool():
    db_conn = sqlite3.connect(':memory:')

    sqlite_tool, (description, params) = tools.make_sqlite_query_tool(db_conn)
    assert isinstance(description, str)
    assert isinstance(params, dict)
    assert callable(sqlite_tool)

    results = sqlite_tool('SELECT * FROM table;')
    assert results.startswith('Query failed')

    cur = db_conn.cursor()

    cur.execute('CREATE TABLE customers(guid NUMERIC PRIMARY KEY, name TEXT, dob TEXT)')
    data = [(0, 'John Doe', '1983-03-04'),
            (1, 'Jane Doe', '1984-06-14')]
    cur.executemany('INSERT INTO customers VALUES(?, ?, ?)', data)
    db_conn.commit()

    results = sqlite_tool('SELECT * FROM customers;')
    assert isinstance(results, str)
    assert results=="[(0, 'John Doe', '1983-03-04'), (1, 'Jane Doe', '1984-06-14')]"
