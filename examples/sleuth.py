"""
Example of using an LLM to chat with a database.
"""

import os
import sys
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import gradio as gr

from neosophia.llmtools import openaiapi as openai, tools, dispatch
from neosophia.agents.react_chat import make_react_agent
from neosophia.agents.helpers import check_question
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

GAMEDIR = Path(project.DATASETS_DIR_PATH) / Path('game1')
EXPENSE_DATABASE = GAMEDIR / 'expense_database.db'
EMPLOYEE_DATABASE = GAMEDIR / 'employee_database.db'
MARKET_TRENDS_DATABASE = GAMEDIR
COMPETITOR_ANALYSIS_DATABASE = GAMEDIR / 'competition.db'
CUSTOMER_FEEDBACK_DATABASE = GAMEDIR / 'customer_feedback.db'
DATABASES = {
    'expense_database': {
        'file': EXPENSE_DATABASE,
        'connection': None
    },
    'employee_database': {
        'file': EMPLOYEE_DATABASE,
        'connection': None
    },
    'competition': {
        'file': COMPETITOR_ANALYSIS_DATABASE,
        'connection': None
    },
    'customer_feedback': {
        'file': CUSTOMER_FEEDBACK_DATABASE,
        'connection': None
    }
}
PRIMARY_ASSET = """================================================
Profit and Loss Statement for Tectonic Tech Inc.
Fiscal Year: 1st Jan 2021 - 31st Dec 2021
================================================

--------------------------------------------------------------------------
              |   Q1   |   Q2   |   Q3   |  Q4   | Total
--------------------------------------------------------------------------
Revenue
Product Sales      | 1.0M | 1.05M | 1.1025M | 1.1576M | 4.31M
--------------------------------------------------------------------------
Costs
Cost of Goods Sold | 0.3M | 0.3M | 0.22M | 0.22M | 1.02M
--------------------------------------------------------------------------
Gross Profit       | 0.7M | 0.75M | 0.8825M | 0.9376M | 3.27M
--------------------------------------------------------------------------
Operating Expenses
R&D                   | 0.15M | 0.17M | 0.19M | 0.21M | 0.73M
Marketing             | 0.10M | 0.10M | 0.12M | 0.13M | 0.45M
Admin Expenses        | 0.05M | 0.05M | 0.06M | 0.06M | 0.22M
Employee Compensation | 0.20M | 0.21M | 0.23M | 0.35M | 0.99M
--------------------------------------------------------------------------
Operating Income      | 0.2M | 0.22M | 0.2825M | 0.1876M | 0.89M
--------------------------------------------------------------------------
Other Income/Expense  | 0.01M | 0.01M | -0.02M | -0.02M | -0.02M
--------------------------------------------------------------------------
Net Income Before Taxes | 0.21M | 0.23M | 0.2625M | 0.1676M | 0.868M
Taxes                   | 0.033M | 0.037M | 0.041M | 0.05M | 0.163M
--------------------------------------------------------------------------
Net Profit              | 0.18M |  0.193M | 0.2215M | 0.1176M | 0.705M
--------------------------------------------------------------------------

Note: All amounts are represented in millions.

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


def get_system_message() -> str:
    """Get the system message."""
    return (
        "You are a business analyst for Tectonic Tech, Inc.  You have the ability to run sqlite queries "
        "against the company's databses to collect information for the user.\n\n "
        "Answer the user's questions as best as you can.  Only use the functions you have been "
        "provided with.\n\n"
        "Before running a query check to see if you already have part or all of the answer from "
        "your interaction history!\n\n"
    )


def get_schema_description(databases) -> str:
    """
    Construct a description of the DB schema for the LLM by retrieving the
    CREATE commands used to create the tables.
    """
    schema_description = f"There are {len(databases)} databases  Here are their names and schemas:\n"
    for i,db in enumerate(databases.keys()):
        schema_description += f'{i+1}. {db}\n'
        cur = databases[db]['connection'].cursor()
        tables = cur.execute("select name from sqlite_master where type='table';").fetchall()
        tables = [col[0] for col in tables]
        for table in tables:
            schema_description += f'Table `{table}:`\n'
            schema_description += str(get_table_schema(databases[db]['connection'], table)) + '\n'
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

    # Connect to the DB and get the table names
    log.debug('Getting the DB information.')
    for db in DATABASES:
        DATABASES[db]['connection'] = sqlite3.connect(DATABASES[db]['file'])
    schema_description = get_schema_description(DATABASES)

    for db in DATABASES:
        DATABASES[db]['connection'].close()
        DATABASES[db]['connection'] = None

    # Setup the agent
    system_message = get_system_message()
    system_message += schema_description
    print(system_message)

    def new_question_wrapper():
        return ''

    def agent_wrapper(question, status, chat_history):

        # Check the reasonableness of the question
        # response = check_question(
        #     question,
        #     schema_description + additional_check,
        #     model,
        #     FUNCTION_DESCRIPTIONS
        # )
        # chat_history.append([None, response])

        # if "This is a reasonable question" in response:
        #     yield 'Checked that the question is answerable', chat_history
        # else:
        #     response = 'Final Answer: ' + response
        #     chat_history[-1][1] = response
        #     yield 'Could not answer question', chat_history
        #     return


        # Build the functions that the agent can use
        for db in DATABASES:
            DATABASES[db]['connection'] = sqlite3.connect(DATABASES[db]['file'])
        databases = {db: DATABASES[db]['connection'] for db in DATABASES}
        def query_database(database: str, query: str) -> str:
            tool, _ = tools.make_sqlite_query_tool(databases[database])
            return tool(query)

        functions = {
            'query_database': query_database
        }

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

        for db in DATABASES:
            DATABASES[db]['connection'].close()
            DATABASES[db]['connection'] = None

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
        gr.Markdown(
            '```\n'
            '--------------------------------------------------------------------------\n'
            '                        |   Q1  |     Q2   |    Q3   |    Q4   | Total\n'
            '--------------------------------------------------------------------------\n'
            'Revenue\n'
            'Product Sales           | 1.0M  |    1.05M | 1.1025M | 1.1576M | 4.31M\n'
            '--------------------------------------------------------------------------\n'
            'Costs\n'
            'Cost of Goods Sold      |  0.3M |    0.3M  |   0.22M |   0.22M | 1.02M\n'
            '--------------------------------------------------------------------------\n'
            'Gross Profit            |  0.7M |    0.75M | 0.8825M | 0.9376M | 3.27M\n'
            '--------------------------------------------------------------------------\n'
            'Operating Expenses\n'
            'R&D                     | 0.15M |    0.17M |   0.19M |   0.21M | 0.73M\n'
            'Marketing               | 0.10M |    0.10M |   0.12M |   0.13M | 0.45M\n'
            'Admin Expenses          | 0.05M |    0.05M |   0.06M |   0.06M | 0.22M\n'
            'Employee Compensation   | 0.20M |    0.21M |   0.23M |   0.35M | 0.99M\n'
            '--------------------------------------------------------------------------\n'
            'Operating Income        |  0.2M |   0.22M |  0.2825M | 0.1876M | 0.89M\n'
            '--------------------------------------------------------------------------\n'
            'Other Income/Expense    | 0.01M |   0.01M |   -0.02M |  -0.02M | -0.02M\n'
            '--------------------------------------------------------------------------\n'
            'Net Income Before Taxes | 0.21M  |   0.23M | 0.2625M | 0.1676M | 0.868M\n'
            'Taxes                   | 0.033M |  0.037M |  0.041M |   0.05M | 0.163M\n'
            '--------------------------------------------------------------------------\n'
            'Net Profit              | 0.18M  |  0.193M | 0.2215M | 0.1176M | 0.705M\n'
            '--------------------------------------------------------------------------\n'
            '```'
        )

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
            'You have access to four databases:\n'
            '\n'
            '1. Expense database with high-level expenses per category.\n'
            '2. Employee compensation database with the cost of employee taxes, salaries and bonuses.\n'
            '3. Competition database with competitor\'s rough financial and market data for the year.\n'
            '4. Customer feedback database with customer feedback by category.'
        )

        clear_button.add([question, chatbot])

        question.submit(
            new_question_wrapper,
            outputs=final_answer) \
        .then(
            agent_wrapper,
            [question, status, chatbot],
            [status, chatbot]) \
        .then(answer_wrapper, chatbot, final_answer)

        ask_button.click(
            new_question_wrapper,
            outputs=final_answer) \
        .then(
            agent_wrapper,
            [question, status, chatbot],
            [status, chatbot]) \
        .then(answer_wrapper, chatbot, final_answer)

        clear_button.click(lambda: None, None, [chatbot, status, final_answer], queue=False)

    demo.queue()
    demo.launch()


if __name__=='__main__':
    main()
