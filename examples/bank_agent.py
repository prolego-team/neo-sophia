import os
import sys
import logging
import sqlite3

import gradio as gr

from neosophia.llmtools import openaiapi as openai, tools
from neosophia.agents.react import make_react_agent, make_simple_react_agent
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
    text = f'{message.role.capitalize()}:\n'
    text += message.content
    text += f'\n\n<name={message.name}, function_call={message.function_call}>\n'
    return text


def summarize_interaction(messages: list[openai.Message]):
    """Generate a transcription of an agent/LLM interaction."""

    responses = []
    for i, message in enumerate(messages[1:]):
        text = format_message(message)
        responses.append(text)

    return responses


def main():
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
        "questions as best as you can.  Only use the functions you have been provided with."
    )
    system_message += schema_description

    def user_wrapper(question, chat_history):
        return '', chat_history + [[question, None]]

    import time

    def agent_wrapper(chat_history):

        question = chat_history[-1][0]

        chat_history[-1][1] = 'Calling api...'
        yield chat_history

        # Build the functions that the agent can use
        db_connection = sqlite3.connect(DATABASE)
        query_database, _ = tools.make_sqlite_query_tool(db_connection)
        functions = {
            'query_database': query_database
        }

        agent = make_react_agent(
            system_message, model, function_descriptions, functions)

        messages = agent(question)

        for response in summarize_interaction(messages):
            chat_history.append([None, response])
            time.sleep(0.05)
            yield chat_history

        # final_answer = True
        # for message in messages:
        #     if 'i cannot construct a final answer' in message.content.lower():
        #         final_answer = False
        #         break

        # if not final_answer:
        #     agent = make_react_agent(
        #         system_message,
        #         model,
        #         function_descriptions,
        #         functions
        #     )
        #     messages = agent(question)

        #     for response in summarize_interaction(messages):
        #         chat_history.append([None, response])
        #         time.sleep(0.05)
        #         yield chat_history

        db_connection.close()

    with gr.Blocks() as demo:
        gr.Markdown('# Chat a Bank Database')

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
