"""
Example of using an LLM to chat with a database.
"""

import os
import sys
import logging
import sqlite3
from datetime import datetime

import gradio as gr

from neosophia.llmtools import openaiapi as openai, tools
from neosophia.agents.react import make_react_agent
from neosophia.db.sqlite_utils import get_db_creation_sql

# === Basic setup ===================================================
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logging.getLogger('agent').setLevel(logging.DEBUG)
log = logging.getLogger('agent')
# ===================================================================

DATABASE = 'data/synthbank.db'
DEFAULT_QUESTION = 'Who has most recently opened a checking account?'

def format_message(message):
    """Convert a message into plain text"""
    text = f'{message.role.capitalize()}:\n'
    text += message.content
    text += f'\n\n_<name={message.name}, function_call={message.function_call}>_\n'
    return text


def summarize_interaction(messages: list[openai.Message]):
    """Generate a transcription of an agent/LLM interaction."""
    responses = []
    for message in messages[1:]:
        text = format_message(message)
        responses.append(text)

    return responses


def main():
    """Setup and run gradio app."""

    # Setup
    openai.set_api_key(os.getenv('OPENAI_API_KEY'))

    # Get a model
    model = openai.start_chat('gpt-4-0613')

    # Connect to the DB and get the table names
    log.debug('Getting the DB information.')
    db_connection = sqlite3.connect(DATABASE)

    function_descriptions = [
        {
            'name': 'query_database',
            'description': 'Query the bank sqlite database.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'A sqlite query to run against the bank databse.'
                    },
                },
                'required': ['sqlite_query']
            }
        }
    ]

    # Construct a description of the DB schema for the LLM by retrieving the
    # CREATE commands used to create the tables.
    schema_description = (
        "Each customer has one or more products at the bank.  Each product has a globally unique "
        "account number.  Each customer has a globally unique guid identifier.  The customer guids "
        "and the product account numbers are related in the 'products' database table.\n\n"
        "The bank's database tables were created using the following commands.\n"
    )
    schema_description += get_db_creation_sql(db_connection)
    db_connection.close()

    # Setup the agent
    system_message = (
        "You are an assistant for a retail bank.  You have the ability to run sqlite queries "
        "against the bank's databse to collect information for the user.  Answer the user's "
        "questions as best as you can.  Only use the functions you have been provided with.\n\n"
        f"Today's date is {datetime.today()}."
    )
    system_message += schema_description

    def user_wrapper(question, chat_history):
        return '', chat_history + [[question, None]]

    def agent_wrapper(chat_history):

        question = chat_history[-1][0]

        # Build the functions that the agent can use
        db_connection = sqlite3.connect(DATABASE)
        query_database, _ = tools.make_sqlite_query_tool(db_connection)
        functions = {
            'query_database': query_database
        }

        agent = make_react_agent(
            system_message, model, function_descriptions, functions)

        for message in agent(question):
            chat_history.append([None, format_message(message)])
            yield chat_history

        db_connection.close()

    with gr.Blocks() as demo:
        gr.Markdown('# Chat With a Bank Database')
        gr.Markdown(
            "You can use the Chatbot below to answer questions about SynthBank's "
            "database.  SynthBank is (fake) bank, and the database contains information "
            "about it's customers and their accounts.\n"
            "## About the bank's database:\n"
            "The database contains the name and date of birth for each of the banks "
            "customers.  Each customer may have one or more associated products:\n"
            "savings account, checking account, mortgage or auto loan.\n\n"
            "For each account the database stores the account open date and the interest rate "
            "if applicable.  Account balances and transactions are _not_ stored in the database.\n"
            "## Example questions:\n"
            "Here are a few questions you could ask:\n"
            "- Who most recently opened a checking account?\n"
            "- How many people have opened a savings account in the last year?\n"
            "- How many products does the person who most recently opened a mortgage have?\n"
            "- Which customer has the highest interest rate on their credit card, and what "
            "is that interest rate?\n"
            "## Interacting with the Chatbot:\n"
            'When the Chatbot thinks it has your answer it will respond with "Final Answer:".\n\n'
            "Sometimes the Chatbot will struggle to get the right answer.  It is programmed "
            "to try again if it makes a mistake up to a limit of 10 tries.  It's reasoning "
            "process and database interactions will be printed in the chat dialogue so that "
            "you can follow along.\n\n"
            'Sometimes the Chatbot will make things up (or "hallucinate").  The code tries '
            "to catch the situations, but if anything looks unusual you can try repeating the "
            "question to make sure you get the same response again."
        )

        chatbot = gr.Chatbot()
        question = gr.Textbox(
            value=DEFAULT_QUESTION, label='Ask a question about the data')

        with gr.Row():
            with gr.Column():
                ask_button = gr.Button('Ask')
            with gr.Column():
                clear_button = gr.ClearButton([question, chatbot])

        question.submit(
            user_wrapper,
            inputs=[question, chatbot],
            outputs=[question, chatbot],
            queue=False).then(agent_wrapper, chatbot, chatbot)

        ask_button.click(
            user_wrapper,
            inputs=[question, chatbot],
            outputs=[question, chatbot],
            queue=False).then(agent_wrapper, chatbot, chatbot)

        clear_button.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    demo.launch()


if __name__=='__main__':
    main()
