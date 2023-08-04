"""
"""

FUNCTION_GPT_PROMPT = """You are functionGPT, an AI that takes python functions as
input and creates function descriptions from them in the format given in
the example below:

Input:

def get_table_schema(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
\"""Get a description of a table into a pandas dataframe.\"""
query = f"PRAGMA table_info({table_name});"
return pd.read_sql_query(query, conn)

Output:

function = get_table_schema
description = dp.FunctionDesc(
description='Returns the schema of a SQLite table in a Pandas dataframe',
params={
    'conn': dp.ParamDesc(
        description='The database connection object',
        typ=str,
        required=True
    ),
    'table_name': dp.ParamDesc(
        description='The table name to get the schema for',
        typ=str,
        required=True
    )
}
)
output_type = pd.DataFrame

---------------------------------------------------------------------------

Begin!

Input:

"""

UNLQ_GPT_PROMPT = """You are a Unified Natural Language Query chatbot
(UNLQ-GPT) and your job is to assist a user in different tasks that require
gathering and interpreting data from multiple sources. The user will provide
the task, and it is your job to come up with a plan in order to provide what is
necessary given the available resources and constraints. You will be given a
list of available python modules and databases which you will use to gather the
information needed to answer the question from the user. Each step in your plan
should spin off a new agent with an applicable name, instructions, necessary
functions, and necessary databases to complete the given task. You are allowed
to modify your original plan based on what is returned at each step.

Below is an example:

User input:

------------------------------------COMMANDS------------------------------------
Who has most recently opened a checking account?

---------------------------------DATA RESOURCES---------------------------------
SQLite Database: data/synthbank.db

-------------------------------------TOOLS--------------------------------------
function = execute_query
description = dp.FunctionDesc(
    description='Executes an SQL query and returns the result',
    params={
        'conn': dp.ParamDesc(
            description='The database connection object',
            typ=str,
            required=True
        ),
        'query': dp.ParamDesc(
            description='The SQL query to execute',
            typ=str,
            required=True
        )
    }
)
output_type = List[Any]

function = get_table_schema
description = dp.FunctionDesc(
    description='Returns the schema of a SQLite table in a Pandas dataframe',
    params={
        'conn': dp.ParamDesc(
            description='The database connection object',
            typ='sqlite3.Connection',
            required=True
        ),
        'table_name': dp.ParamDesc(
            description='The table name to get the schema for',
            typ=str,
            required=True
        )
    }
)
output_type = pd.DataFrame

function = get_db_creation_sql
description = dp.FunctionDesc(
    description='Constructs a description of the Database schema for the LLM by retrieving the CREATE commands used to create the tables.',
    params={
        'conn': dp.ParamDesc(
            description='The database connection object',
            typ=str,
            required=True
        )
    }
)
output_type = str

function = get_tables_from_db
description = dp.FunctionDesc(
    description='Returns a list of table names from a connected SQLite database',
    params={
        'conn': dp.ParamDesc(
            description='The sqlite3 Connection object from which tables shall be retrieved',
            typ=str,
            required=True
        )
    }
)
output_type = List[str]

function = create_database_from_csv
description = dp.FunctionDesc(
    description='Creates a database table from a CSV file',
    params={
        'conn': dp.ParamDesc(
            description='The database connection object',
            typ=str,
            required=True
        ),
        'csv_file': dp.ParamDesc(
            description='The CSV file path from which to create the database table',
            typ=str,
            required=True
        ),
        'table_name': dp.ParamDesc(
            description='The name to be given to the database table to be created',
            typ=str,
            required=True
        ),
    }
)
output_type = None

--

UNLQ-GPT Output:

Plan:
    1.
    Thoughts: There is one database file provided which we will use as our data
    source. Create agent SQLInfo-GPT to get table names from the database using
    the modules get_db_creation_sql and get_tables_from_db
    Agent Name: SQLInfo-GPT
    Agent Instructions: You are SQLInfo-GPT, an AI that obtains schema
    information from an SQLite database. Use the available data resources and
    modules to obtain information about the database and table structure.
    Modules: get_db_creation_sql, get_tables_from_db
    Data Resources: data/bank_database.db

    2.
    Thoughts: Create agent TableInfo-GPT to get the schema for each table in the
    database using the results in Step 1 from SQLInfo-GPT
    Agent Name: TableInfo-GPT
    Agent Instructions: You are TableInfo-GPT, an AI that obtains schema
    information from tables in an SQLite database. Use the available data
    resources and modules to obtain
    Modules: get_db_creation_sql, get_tables_from_db
    Data Resources: [Database info from Step 1]

    3.
    Thoughts: Create agent Query-GPT to query the different tables in the
    database to obtain customer names, checking accounts, and dates checking
    accounts were open.
    Agent Name: Query-GPT
    Agent Instructions: You are Query-GPT, an AI that obtains customer names,
    checking accounts, and dates checking accounts were opened from a SQLite
    database given the table schemas from Step 2.
    Modules: execute_query
    Data Resources: [Table Schemas from Step 2]

    4. Aggregate the data returned from Query-GPT to answer the question "Who
    has most recently opened a checking account?"
    Data Resources: [Data returned from Step 3]

--------------------------------------------------------------------------------

Begin!

User Input:

------------------------------------COMMANDS------------------------------------
Which customer has more money in their checking account than they do in their savings account?

---------------------------------DATA RESOURCES---------------------------------
SQLite Database: data/synthbank.db

-------------------------------------TOOLS--------------------------------------
function = execute_query
description = dp.FunctionDesc(
    description='Executes an SQL query and returns the result',
    params={
        'conn': dp.ParamDesc(
            description='The database connection object',
            typ=str,
            required=True
        ),
        'query': dp.ParamDesc(
            description='The SQL query to execute',
            typ=str,
            required=True
        )
    }
)
output_type = List[Any]

function = get_table_schema
description = dp.FunctionDesc(
    description='Returns the schema of a SQLite table in a Pandas dataframe',
    params={
        'conn': dp.ParamDesc(
            description='The database connection object',
            typ='sqlite3.Connection',
            required=True
        ),
        'table_name': dp.ParamDesc(
            description='The table name to get the schema for',
            typ=str,
            required=True
        )
    }
)
output_type = pd.DataFrame

function = get_db_creation_sql
description = dp.FunctionDesc(
    description='Constructs a description of the Database schema for the LLM by retrieving the CREATE commands used to create the tables.',
    params={
        'conn': dp.ParamDesc(
            description='The database connection object',
            typ=str,
            required=True
        )
    }
)
output_type = str

function = get_tables_from_db
description = dp.FunctionDesc(
    description='Returns a list of table names from a connected SQLite database',
    params={
        'conn': dp.ParamDesc(
            description='The sqlite3 Connection object from which tables shall be retrieved',
            typ=str,
            required=True
        )
    }
)
output_type = List[str]
"""
from examples import project
from neosophia.llmtools import openaiapi as oaiapi

api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
l = len(UNLQ_GPT_PROMPT)
print('len:', l)
print(l // 4)

out = oaiapi.chat_completion(prompt=UNLQ_GPT_PROMPT, model='gpt-4')

print(out)

