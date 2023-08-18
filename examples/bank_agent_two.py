"""
Example of using an LLM to chat with a database.
"""

import os
import sys
import logging
import sqlite3
from datetime import datetime

import gradio as gr

from neosophia.llmtools import openaiapi as openai, tools, dispatch
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

CUSTOMER_DATABASE = os.path.join(project.DATASETS_DIR_PATH, 'synthbank.db')
TRANSACTION_DATABASE = os.path.join(project.DATASETS_DIR_PATH, 'transactions.db')
DEFAULT_QUESTION = 'How many accounts does this customer have with the bank?'
MAX_LLM_CALLS_PER_INTERACTION = 10

FUNCTION_DESCRIPTIONS = dispatch.convert_function_descs({
    'query_database': dispatch.FunctionDesc(
        description='Query the bank sqlite database.',
        params={
            'database': dispatch.ParamDesc(
                description='The databse in which to run the query.',
                typ=str,
                required=True
            ),
            'query': dispatch.ParamDesc(
                description='A sqlite query to run against the bank databse.',
                typ=str,
                required=True
            )
        }
    )
})


def get_system_message() -> str:
    """Get the system message."""
    return (
        "You are an assistant for a retail bank.  You have the ability to run sqlite queries "
        "against the bank's databse to collect information for the user.\n\n "
        "Answer the user's questions as best as you can.  Only use the functions you have been "
        "provided with.\n\n"
        "Before running a query check to see if you already have part or all of the answer from "
        "your interaction history!\n\n"
        f"Today's date is {datetime.today()}."
    )


def get_schema_description(customer_db_connection, transaction_db_connection) -> str:
    """
    Construct a description of the DB schema for the LLM by retrieving the
    CREATE commands used to create the tables.
    """
    schema_description = (
        "There are two databases:\n"
        "1. `customer_database` contains information on customers and the products they hold. "
        "Each customer has one or more products at the bank.  Each product has a globally unique "
        "account number.  Each customer has a globally unique guid identifier.  The customer guids "
        "and the product account numbers are related in the 'products' database table.\n"
    )
    schema_description += "The bank has the following products:\n"
    schema_description += '"Standard credit card", "FlexSave Savings Account", "Standard mortgage", "PrestigeSave Premium Savings Account", "PremierAccess Checking Account", "Standard auto loan", "EasyAccess Checking Account"\n'
    schema_description += "The customer database tables were created using the following commands.\n"
    schema_description += get_db_creation_sql(customer_db_connection)
    schema_description += (
        "\n\n"
        "2. `transaction_database` contains all transactions for all accounts.  The accounts are "
        "identified by account numbers that can be looked up in the `customer_database`. A "
        "transaction consists of a date, description, withdrawal amount and deposit amount.\n"
        "The transaction database table was created using the following command.\n"
    )
    schema_description += get_db_creation_sql(transaction_db_connection)

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
    customer_db_connection = sqlite3.connect(CUSTOMER_DATABASE)
    transaction_db_connection = sqlite3.connect(TRANSACTION_DATABASE)

    schema_description = get_schema_description(
        customer_db_connection, transaction_db_connection
    )

    cursor = customer_db_connection.cursor()
    customer_list = cursor.execute('SELECT DISTINCT name FROM customers;').fetchall()
    customer_list = [name[0] for name in customer_list]

    customer_db_connection.close()
    transaction_db_connection.close()

    # Setup the agent
    system_message = get_system_message()
    system_message += schema_description

    def new_question_wrapper():
        return ''

    def agent_wrapper(customer_name, question, status, chat_history):

        # Check the reasonableness of the question
        additional_check = (
            f'\n\nQuestions should only pertain to the customer {customer_name}.'
        )
        response = check_question(
            question,
            schema_description + additional_check,
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
        customer_db_connection = sqlite3.connect(CUSTOMER_DATABASE)
        transaction_db_connection = sqlite3.connect(TRANSACTION_DATABASE)
        databases = {
            'customer_database': customer_db_connection,
            'transaction_database': transaction_db_connection
        }
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
            f'The following question is about {customer_name}.  '
            'Only answer questions about this customer.\n'
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

        customer_db_connection.close()
        transaction_db_connection.close()

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
        gr.Markdown('# Chat About a Bank Customer')

        customer_name = gr.Dropdown(customer_list, label='Customer name')
        question = gr.Textbox(
            value=DEFAULT_QUESTION, label='Ask a question about this customer')
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
            "## About the databases:\n"
            "There are two databses.\n\n"
            "The first database contains the name and date of birth for each of the banks "
            "customers.  Each customer may have one or more associated products:\n"
            "savings account, checking account, mortgage or auto loan.\n\n"
            "For each account the database stores the account open date and the interest rate "
            "if applicable.  Account balances and transactions are _not_ stored in this database.\n\n"
            "The database has eight tables:\n"
            "- The `customers` table has three fields: `guid`, `name` and `dob`\n"
            "- The `credit_scores` table has two fields: `guid`, `credit_score`\n"
            "- The `products` table has three fields: `account_number`, `product_name` and `guid`\n"
            "- The `auto_loan` table has four fields: `loan_amount`, `interest_rate`, `account_open_date`, `account_number`\n"
            "- The `mortgage` table has four fields: `loan_amount`, `interest_rate`, `account_open_date`, `account_number`\n"
            "- The `checking_account` table has three fields: `account_open_date`, `account_number` and `account_balance`\n"
            "- The `savings_account` table has four fields: `interest_rate`, `account_open_date`, `account_number` and `account_balance`\n"
            "- The `credit_card` table has four fields: `credit_limit`, `interest_rate`, `account_open_date`, `account_number`\n\n"
            "The second database contains a global list of all transactions.  This database contains a "
            "single table, `transactions`, that has four fields: `account_number`, `date`, "
            "`description`, `withdrawal` and `deposit`.  The `withdrawal` and `deposit` fields are "
            "dollar amounts and are mutually exclusive.\n\n"
            "## Example questions:\n"
            "Here are a few questions you could ask:\n"
            "- How many accounts does this customer have?\n"
            "- List this customer's accounts?\n"
            "- What were this customers last 10 checking account transactions?\n"
            "- What was this customer's largest transaction?\n"
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
            [customer_name, question, status, chatbot],
            [status, chatbot]) \
        .then(answer_wrapper, chatbot, final_answer)

        ask_button.click(
            new_question_wrapper,
            outputs=final_answer) \
        .then(
            agent_wrapper,
            [customer_name, question, status, chatbot],
            [status, chatbot]) \
        .then(answer_wrapper, chatbot, final_answer)

        clear_button.click(lambda: None, None, [chatbot, status, final_answer], queue=False)

    demo.queue()
    demo.launch()


if __name__=='__main__':
    main()
