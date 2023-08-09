"""
"""

DB_INFO_PROMPT = """Given the following database name and table schemas,
generate a brief one paragraph human readable description of the database. Do
not engage in any conversation or provide an explanation. Simply provide the
description.

"""

FUNCTION_GPT_PROMPT = """You are Function-GPT, an AI that takes python
functions as input and creates function descriptions from them in the format
given in the example below.

[Input]
def get_table_schema(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    \"""Get a description of a table into a pandas dataframe.\"""
    query = f"PRAGMA table_info({table_name});"
    return pd.read_sql_query(query, conn)

[Output]
  - name: get_table_schema
    description: This function gets the schema for a given table and returns it is a Pandas dataframe.
    params:
      conn:
        description: A connection object representing the SQLite database.
        type: sqlite3.Connection
        required: true
      table_name:
        description: The name of the table to get the schema for.
        type: str
        required: true

---------------------------------------------------------------------------

Begin!

"""

UNLQ_GPT_BASE_PROMPT = """You are a Unified Natural Language Query chatbot
(UNLQ-GPT) and your job is to assist a user in different tasks that require
gathering and interpreting data from multiple sources. You will be provided
with a COMMAND from a user, a list of DATA RESOURCES containing databases and
their descriptions, TOOLS which are python functions you are able to generate
parameters for, and CONSTRAINTS which describe any constraints you may have.
When generating parameter values for python functions, always use single quotes
when the value is a string.  Your job is to come up with a plan in order to
retrieve the information needed to answer the question. Because some functions
require the result from previously run functions, ONLY GENERATE ONE STEP AT A
TIME starting with Step 1.  You will be provided with the previous steps that
have been taken. The results from functions that have been called as a result
of certain steps will be added to the FUNCTION RESOURCES section. When you have
collected enough information to answer the question, call the function
`extract_answer`. Do not engage in any conversation outside of the "Thoughts"
section. Each step in your plan must be in the following format:

Step: Step Number
Thoughts: Explanation of why the action is to be taken
Function: The Python function to call in order to get the information needed
Parameter_0: Parameter_0 Name | Parameter_0 Value | type
...
Parameter_N: Parameter_N Name | Parameter_N Value | type
Returned: Name describing what the function returned to store as a variable

"""

UNLQ_GPT_EXAMPLE_PROMPT = """
Below is an example:

[Beginning of example]

[Input]
------------------------------------COMMANDS------------------------------------
Who has most recently opened a checking account?

---------------------------------DATA RESOURCES---------------------------------
SQLite Database: data/synthbank.db
SQLite Database: data/baloney.db

-------------------------------------TOOLS--------------------------------------
name: execute_query
description: This function executes a provided SQL query and returns the results
  as a list.
params:
  conn:
    description: A connection object representing the database.
    type: sqlite3.Connection
    required: true
  query:
    description: The SQL query to be executed.
    type: str
    required: true

name: get_table_schema
description: This function retrieves a description of a specified table and populates
  the data into a Pandas dataframe.
params:
  conn:
    description: An active sqlite3 connection object that represents the SQLite
      database in use.
    type: sqlite3.Connection
    required: true
  table_name:
    description: The specific name of the table for which the schema is to be obtained.
    type: str
    required: true
returns:
  description: A Pandas DataFrame containing the schema of the specified table.
  type: pd.DataFrame

name: get_db_creation_sql
description: This function constructs a description of the database schema for the
  LLM by retrieving the CREATE commands used to create the tables.
params:
  conn:
    description: A connection object representing the SQLite database.
    type: sqlite3.Connection
    required: true
returns:
  description: A string that contains the schema description of the database.
  type: str

name: get_tables_from_db
description: This function retrieves a list of all table names from the database.
params:
  conn:
    description: A connection object representing the SQLite database.
    type: sqlite3.Connection
    required: true
returns:
  description: A list of table names.
  type: str

name: create_database_from_csv
description: This function creates a database table from a CSV file. It first reads
  the header from the CSV file and creates a new table with those column names.
  Then, it proceeds to insert each row from the CSV file into the new table. If
  a table with the same name already exists in the database, it will be dropped
  before the new table is created. After all operations, it commits the transaction
  and prints a success message.
params:
  conn:
    description: A connection object representing the SQLite database.
    type: sqlite3.Connection
    required: true
  csv_file:
    description: The name of the CSV file to import data from.
    type: str
    required: true
  table_name:
    description: The name of the table to create in the database.
    type: str
    required: true
returns:
  description: This function does not return any values. The database is updated
    in-place.
  type: None
--

[Output]

Step: 1
Thoughts: There is one database file provided which we will use as our data source. Create agent SQLInfo-GPT to get table names from the database using the functions get_db_creation_sql and get_tables_from_db
Agent Name: SQLInfo-GPT
Agent Instructions: You are SQLInfo-GPT, an AI that obtains schema information from an SQLite database. Use the available data resources and Python functions to obtain information about the database and table structure by generating SQL queries to pass to the available functions.
Modules: get_db_creation_sql, get_tables_from_db
Data Resources: data/bank_database.db

Step: 2
Thoughts: Create agent TableInfo-GPT to get the schema for each table in the database using the results in Step 1 from SQLInfo-GPT
Agent Name: TableInfo-GPT
Agent Instructions: You are TableInfo-GPT, an AI that obtains schema information from tables in an SQLite database. Use the available data resources and modules to obtain the information required to answer the question.
Modules: get_db_creation_sql, get_tables_from_db
Data Resources: [Database info from Step 1]

Step: 3
Thoughts: Create agent Query-GPT to query the different tables in the database to obtain customer names, checking accounts, and dates checking accounts were open.
Agent Name: Query-GPT
Agent Instructions: You are Query-GPT, an AI that obtains customer names, checking accounts, and dates checking accounts were opened from a SQLite database given the table schemas from Step 2.
Modules: get_db_creation_sql, execute_query
Data Resources: [Table Schemas from Step 2]

Step: 4
Do not create a new agent. You must aggregate the data returned from all other agents in order to answer the question "Who has most recently opened a checking account?"
Data Resources: [Data returned from Step 3]

[End of Example]
"""

