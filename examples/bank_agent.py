"""
Example of using an LLM to chat with a database.
"""

import sys
import logging
import sqlite3
from datetime import datetime

import gradio as gr

from neosophia.llmtools import openaiapi as openai, tools
from neosophia.agents.react_chat import make_react_agent
from neosophia.agents.helpers import check_question
from neosophia.db.sqlite_utils import get_db_creation_sql

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

DATABASE = 'data/synthbank.db'
DEFAULT_QUESTION = 'Who has most recently opened a checking account?'
MAX_LLM_CALLS_PER_INTERACTION = 10

FUNCTION_DESCRIPTIONS = [
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


def get_system_message(today) -> str:
    """Get the system message."""
    return (
        "You are an assistant for a retail bank.  You have the ability to run sqlite queries "
        "against the bank's databse to collect information for the user.  Answer the user's "
        "questions as best as you can.  Only use the functions you have been provided with.\n\n"
        f"Today's date is {today}."
    )


def get_schema_description(db_connection) -> str:
    """
    Construct a description of the DB schema for the LLM by retrieving the
    CREATE commands used to create the tables.
    """
    schema_description = (
        "Each customer has one or more products at the bank.  Each product has a globally unique "
        "account number.  Each customer has a globally unique guid identifier.  The customer guids "
        "and the product account numbers are related in the 'products' database table.\n\n"
        "The bank's database tables were created using the following commands.\n"
    )
    schema_description += get_db_creation_sql(db_connection)
    return schema_description


def format_message(message):
    """Convert a message into plain text"""
    text = f'{message.role.capitalize()}:\n'
    text += message.content
    text += f'\n\n_<name={message.name}, function_call={message.function_call}>_\n'
    return text


def main():
    """Setup and run gradio app."""

    # Setup
    # openai.set_api_key(os.getenv('OPENAI_API_KEY'))
    api_key = openai.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    openai.set_api_key(api_key)

    # Get a model
    model = openai.start_chat('gpt-4-0613')

    # Connect to the DB and get the table names
    log.debug('Getting the DB information.')
    db_connection = sqlite3.connect(DATABASE)

    schema_description = get_schema_description(db_connection)
    db_connection.close()

    # Setup the agent
    system_message = get_system_message(datetime.today())
    system_message += schema_description

    def new_question_wrapper():
        return ''

    def agent_wrapper(question, status, chat_history):

        # Check the reasonableness of the question
        response = check_question(
            question,
            schema_description,
            model,
            FUNCTION_DESCRIPTIONS
        )
        chat_history.append([None, response])

        if "This is a reasonable question" in response:
            yield 'Checked that the question is answerable', chat_history
        else:
            response = 'Final Answer: ' + response
            chat_history[-1][1] = response
            yield 'Could not answer question', chat_history
            return


        # Build the functions that the agent can use
        db_connection = sqlite3.connect(DATABASE)

        query_database, _ = tools.make_sqlite_query_tool(db_connection)
        functions = {
            'query_database': query_database
        }

        agent = make_react_agent(
            system_message,
            model,
            FUNCTION_DESCRIPTIONS,
            functions,
            MAX_LLM_CALLS_PER_INTERACTION,
        )

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

        db_connection.close()

    def answer_wrapper(chat_history):
        final_message = chat_history[-1][1]
        if final_message is None:
            response = 'It appears the agent couldn\'t answer the questions.'
        elif 'Final Answer:' in final_message:
            response = final_message.split('Final Answer:')[1].strip()
            response = response.split('\n')[0].strip()
        else:
            response = 'It appears the agent couldn\'t answer the questions.'

        return response

    with gr.Blocks() as demo:
        gr.Markdown('# Chat With a Bank Database')

        question = gr.Textbox(
            value=DEFAULT_QUESTION, label='Ask a question about the data')
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
            "## About the database:\n"
            "The database contains the name and date of birth for each of the banks "
            "customers.  Each customer may have one or more associated products:\n"
            "savings account, checking account, mortgage or auto loan.\n\n"
            "For each account the database stores the account open date and the interest rate "
            "if applicable.  Account balances and transactions are _not_ stored in the database.\n\n"
            "The database has eight tables:\n"
            "- The `customers` table has three fields: `guid`, `name` and `dob`\n"
            "- The `credit_scores` table has two fields: `guid`, `credit_score`\n"
            "- The `products` table has three fields: `account_number`, `product_name` and `guid`\n"
            "- The `auto_loan` table has four fields: `loan_amount`, `interest_rate`, `account_open_date`, `account_number`\n"
            "- The `mortgage` table has four fields: `loan_amount`, `interest_rate`, `account_open_date`, `account_number`\n"
            "- The `checking_account` table has three fields: `account_open_date`, `account_number` and `account_balance`\n"
            "- The `savings_account` table has four fields: `interest_rate`, `account_open_date`, `account_number` and `account_balance`\n"
            "- The `credit_card` table has four fields: `credit_limit`, `interest_rate`, `account_open_date`, `account_number`\n"
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
