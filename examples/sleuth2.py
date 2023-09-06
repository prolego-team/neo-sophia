"""
Example of using an LLM to chat with a database.
"""

import sys
import logging
import sqlite3
from pathlib import Path

import gradio as gr

from neosophia.llmtools import openaiapi as openai, tools, dispatch
from neosophia.agents.react_chat import make_react_agent
from neosophia.db.sqlite_utils import get_table_schema

from examples import project

# === Basic setup ===================================================
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logging.getLogger('agent').setLevel(logging.DEBUG)
log = logging.getLogger('agent')
# ===================================================================

GAMEDIR = Path(project.DATASETS_DIR_PATH) / Path('sim1')
# EXPENSE_DATABASE = GAMEDIR / 'expense_database.db'
# EMPLOYEE_DATABASE = GAMEDIR / 'employee_database.db'
# MARKET_TRENDS_DATABASE = GAMEDIR
# COMPETITOR_ANALYSIS_DATABASE = GAMEDIR / 'competition.db'
# CUSTOMER_FEEDBACK_DATABASE = GAMEDIR / 'customer_feedback.db'
# DATABASES = {
#     'expense_database': EXPENSE_DATABASE,
#     'employee_database': EMPLOYEE_DATABASE,
#     'competition': COMPETITOR_ANALYSIS_DATABASE,
#     'customer_feedback': CUSTOMER_FEEDBACK_DATABASE,
# }
FINANCE_DATABASE = GAMEDIR / 'Company_B_finance.db'
COMPETITION_DATABSE = GAMEDIR / 'Company_B_competition.db'
DATABASES = {
    'finance_database': (FINANCE_DATABASE, 'This database describes the company financial operation.'),
    'competitor_database': (COMPETITION_DATABSE, 'This database describes activities of the company\'s competitors.')
}
PRIMARY_ASSET = """
Profit and Loss for Company
Jan 1, 2022 through Dec 31, 2022
--------------------------------------------------------------------
                          |      Q1 |      Q2 |      Q3 |      Q4 |
--------------------------------------------------------------------
Revenue
Product sales             |   17791 |   22115 |   17165 |   10804 |
--------------------------------------------------------------------
Costs
Cost of goods sold        |   14820 |   18489 |   14273 |    9047 |
--------------------------------------------------------------------
GROSS PROFIT              |    2971 |    3626 |    2892 |    1756 |
--------------------------------------------------------------------
Operating Expenses
HR Expenditure            |    1485 |    1505 |    1499 |    1515 |
Marketing Expenditure     |      59 |      61 |      60 |      61 |
Operations Expenditure    |      30 |      29 |      31 |      29 |
R&D Expenditure           |      60 |      60 |      59 |      60 |
--------------------------------------------------------------------
OPERATING INCOME          |    1337 |    1971 |    1243 |      91 |
--------------------------------------------------------------------
Taxes                     |     241 |     355 |     224 |      16 |
--------------------------------------------------------------------
NET PROFIT                |    1096 |    1616 |    1019 |      75 |
"""

DEFAULT_QUESTION = 'Does anything look unusual in the report?'
MAX_LLM_CALLS_PER_INTERACTION = 10

FUNCTION_DESCRIPTIONS = dispatch.convert_function_descs({
    'query_database': dispatch.FunctionDesc(
        description='Query a company sqlite database.',
        params={
            'database': dispatch.ParamDesc(
                description='The databse in which to run the query.',
                typ=str,
                required=True
            ),
            'query': dispatch.ParamDesc(
                description='A sqlite query to run against the databse.',
                typ=str,
                required=True
            )
        }
    )
})


SYSTEM_MESSAGE = (
    "You are a business analyst for Tectonic Tech, Inc.  You have the ability to run sqlite queries "
    "against the company's databses to collect information for the user.\n\n "
    "Answer the user's questions as best as you can, and prioritize using data from one of the databases."
    "You do not need permission to run a query against a databse, you can just do it, although you should "
    "check to see if you already have part or all of the answer from your interaction history!\n\n"
    "If you are unable to answer a question, suggest some alternative questions that you can answer.  Your"
    "goal is to be helpful."
)


def get_schema_description(db_connections, db_descriptions) -> str:
    """
    Construct a description of the DB schema for the LLM by retrieving the
    CREATE commands used to create the tables.
    """
    schema_description = f"There are {len(db_connections)} databases  Here are their names and schemas:\n"
    for i,db in enumerate(db_connections.keys()):
        schema_description += f'Database {i+1} name: {db}\n'
        schema_description += db_descriptions[db] + '\n'
        cur = db_connections[db].cursor()
        tables = cur.execute("select name from sqlite_master where type='table';").fetchall()
        tables = [col[0] for col in tables]
        schema_description += f'This database has {len(tables)} tables:\n\n'
        for table in tables:
            schema_description += f'Table `{table}:`\n'
            schema_description += str(get_table_schema(db_connections[db], table)) + '\n\n'
        schema_description += '\n'

    return schema_description


def format_message(message):
    """Convert a message into plain text"""
    text = f'{message.role.capitalize()}:\n'
    text += message.content
    text += f'\n\n_<name={message.name}, function_call={message.function_call}>_\n'
    return text


def concat_chat_history(chat_history: list[tuple[str,str]]) -> str:
    """Stringify a chat history."""
    message_list = [msg[0] if msg[0] else msg[1] for msg in chat_history]
    return '\n\n'.join(message_list)


def main():
    """Setup and run gradio app."""

    # Setup
    key = openai.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    openai.set_api_key(key)

    # Get a model
    model = openai.start_chat('gpt-4-0613')
    # model = openai.start_chat('gpt-3.5-turbo-0613')

    # Connect to the DB and get the table names
    log.debug('Getting the DB information.')
    db_connections = {db: sqlite3.connect(DATABASES[db][0]) for db in DATABASES}
    db_description = {db: DATABASES[db][1] for db in DATABASES}
    schema_description = get_schema_description(db_connections, db_description)

    for _,db in db_connections.items():
        db.close()

    # Setup the agent
    system_message = SYSTEM_MESSAGE
    system_message += schema_description
    print(system_message)

    def new_question_wrapper():
        return ''

    def agent_wrapper(question, status, chat_history):

        # Build the functions that the agent can use
        db_connections = {}
        for db in DATABASES:
            db_connections[db] = sqlite3.connect(DATABASES[db][0])

        def query_database(database: str, query: str) -> str:
            tool, _ = tools.make_sqlite_query_tool(db_connections[database])
            return tool(query)

        functions = {
            'query_database': query_database
        }

        # Prepare the agent
        if len(chat_history)>1:
            extra_context = 'Here is a summary of our conversation so far:\n\n'
            extra_context += concat_chat_history(chat_history[1:])
        else:
            extra_context = None

        agent = make_react_agent(
            system_message,
            model,
            FUNCTION_DESCRIPTIONS,
            functions,
            MAX_LLM_CALLS_PER_INTERACTION,
            extra_context
        )

        question_context = (
            f'The following question is about this report:\n' +
            PRIMARY_ASSET + '\n'
        )
        question = question_context + question
        for message in agent(question):
            if message.role=='user':
                status = 'User agent asked a question or provided feedback to the LLM.  Awaiting LLM response...'
                update = [format_message(message), None]
            elif message.role=='function':
                status = 'A query was run against the database.  Awaiting LLM response...'
                update = [format_message(message), None]
            elif message.role=='assistant':
                if 'Final Answer:' in message.content:
                    status = 'The final answer has been determined.'
                else:
                    status = 'The assistant responded.  Awaiting LLM next response...'
                update = [None, format_message(message)]
            chat_history.append(update)
            yield status, chat_history

        for _,connection in db_connections.items():
            connection.close()

    def answer_wrapper(chat_history):
        final_message = chat_history[-1][1]
        if final_message is None:
            response = 'It appears the agent couldn\'t answer the questions.'
        elif 'Final Answer:' in final_message:
            response = final_message.split('Final Answer:')[1].strip()
            response = response.split('_<name')[0].strip()
        else:
            response = 'It appears the agent couldn\'t answer the questions.'

        return response

    with gr.Blocks() as demo:
        gr.Markdown('# Chat About Techtonic Tech, Inc\'s P&L')
        gr.Markdown('```\n' + PRIMARY_ASSET + '```')

        question = gr.Textbox(
            value=DEFAULT_QUESTION, label='Ask a question')
        with gr.Row():
            with gr.Column():
                ask_button = gr.Button('Ask')
            with gr.Column():
                clear_button = gr.ClearButton()
        status = gr.Textbox(
            value='Status will appear here', label='Agent status', interactive=False
        )
        final_answer = gr.Textbox(value='', label='Answer', interactive=False)

        chatbot = gr.Chatbot(label='Chatbot message log')

        gr.Markdown(
            'You have access to two databases:\n'
            '\n'
            '1. A finance database with details on revenue, product costs and operating expenses.\n'
            '2. A competition database that estimates the number of products sold and the marketing spend of your competitor.\n\n'
            'Example questions:\n'
            '\n'
            '1. Is there anything unusual in the report?\n'
            '2. Why did my revenue decrease in the second half of the year?\n'
            '3. Is the cost of good sold a factor in declining revenue?\n'
        )

        clear_button.add([question, chatbot])

        question.submit(
            new_question_wrapper,
            outputs=final_answer
        ).then(
            agent_wrapper,
            [question, status, chatbot],
            [status, chatbot]
        ).then(answer_wrapper, chatbot, final_answer)

        ask_button.click(
            new_question_wrapper,
            outputs=final_answer
        ).then(
            agent_wrapper,
            [question, status, chatbot],
            [status, chatbot]
        ).then(answer_wrapper, chatbot, final_answer)

        clear_button.click(lambda: None, None, [chatbot, status, final_answer], queue=False)

    demo.queue()
    demo.launch()


if __name__=='__main__':
    main()
