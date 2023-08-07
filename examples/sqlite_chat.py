"""
Script to interact with a SQLite database using natural language commands
"""
import os
import json
import sqlite3

from typing import List, Optional, Tuple

import click
import gradio as gr
import pandas as pd

import neosophia.db.sqlite_utils as sql_utils

from examples import project
from neosophia.llmtools import openaiapi as oaiapi

opj = os.path.join

OPENAI_LLM_MODEL_NAME = 'gpt-4'
TABLE_NAME = 'data'
DEFAULT_QUESTION = 'Which customer has the most money?'
EXP_QUERY_TEMPLATE = json.dumps(
    {
        'explanation': '[explain what the query does]',
        'query': '```[sql query]```'
    }, indent=4
)


def setup(csv_file: str) -> Tuple[str, pd.DataFrame]:
    """ Makes initial connection to the database and gets the table schema """
    db_file = opj(project.DATASETS_DIR_PATH, 'bank_database.db')
    conn = sqlite3.connect(db_file)
    sql_utils.create_database_from_csv(conn, csv_file, TABLE_NAME)
    schema = sql_utils.get_table_schema(conn, TABLE_NAME)
    return db_file, schema


def get_error_prompt(
        schema: str,
        table_name: str,
        question: str,
        query: str,
        sql_error_message: str) -> str:
    """ Given an SQL error after a bad query, this writes a prompt to fix it """

    sql_error_dict = {
        'Question': question,
        'Query': query,
        'SQL Error': str(sql_error_message)
    }

    prompt = get_table_prompt(schema, table_name)
    prompt += 'Given the followng question, query, and sql error, fix the query.'
    prompt += json.dumps(sql_error_dict)
    prompt += '\nReturn your answer by filling out the following template:\n'
    prompt += EXP_QUERY_TEMPLATE
    return prompt


def extract_query_from_response(response: str) -> Tuple[str, Optional[str]]:
    """
    Extracts the explanation and SQL query out of the response from the model
    """

    print('Response_1:', response, '\n')

    if '{' in response and '}' in response:
        # Sometimes there's more explanation before/after the returned JSON
        response = '{' + response.split('{')[1].split('}')[0] + '}'
        print('Response_2:', response, '\n')
        response = response.replace('```json', '```')
        response = response.strip().rstrip().replace('\n', '')
        print('Response_3:', response, '\n')
        response = json.loads(response)
        return response['explanation'], response['query'].replace('```', '')
    else:
        # No JSON returned, so probably just an explanation for why it can't
        # complete it
        return response, None


def get_table_prompt(schema: str, table_name: str) -> str:
    """ Writes a prompt to show information about the table """
    prompt = 'Below is the information for an SQLite table.'
    prompt += f'Table name: {table_name}\n'
    prompt += f'Table schema:\n{schema}\n\n'
    prompt += '\n------------------------------------------------------\n'
    return prompt


def get_user_agent_prompt(schema: str, table_name: str, question: str) -> str:
    """ Writes the prompt to ask the question given by the user """
    prompt = get_table_prompt(schema, table_name)
    prompt += 'Below is a question input from a user. '
    prompt += 'Generate an SQL query that pulls the necessary data to answer the question.\n\n'
    prompt += f'Question: {question}\n\n'
    prompt += 'Return your answer by filling out the following template:\n'
    prompt += EXP_QUERY_TEMPLATE

    return prompt


def get_question_result(
        question: str, query: str, explanation: str, result: str) -> str:
    """
    Formats the question, query, explanation, and result into a JSON type string
    """
    return json.dumps(
        {
            'question': question,
            'query': query,
            'explanation': explanation,
            'result': result
        }, indent=4
    )


def get_db_agent_prompt(
        schema: str,
        table_name: str,
        question: str,
        query: str,
        explanation: str,
        result: pd.DataFrame) -> str:
    """
    Writes the prompt to answer the question the user had given the result from
    the query to the database
    """
    prompt = get_table_prompt(schema, table_name)
    prompt += 'Below is a question, SQL query, explanation, and the result from executing the query. '
    prompt += 'Use these pieces of information to answer the question.\n\n'
    prompt += get_question_result(
        question, query, explanation, result.to_string())
    return prompt


@click.command()
@click.option(
    '--csv_file', '-c',
    default=f'{project.DATASETS_DIR_PATH}/bank_customers.csv')
def main(csv_file: str):
    """Main program."""

    key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    oaiapi.set_api_key(key)

    db_file, schema = setup(csv_file)

    def respond(question: str, chat_history: List[Tuple[str, str]]) -> Tuple:
        """Respond to a chat message."""

        conn = sqlite3.connect(db_file)

        user_prompt = get_user_agent_prompt(
            schema.to_string(), TABLE_NAME, question)

        ua_response = oaiapi.chat_completion(
            prompt=user_prompt,
            model=OPENAI_LLM_MODEL_NAME)
        explanation, query = extract_query_from_response(ua_response)

        if query is None:
            conn.close()
            return '', chat_history, None, '', explanation

        success = False
        for _ in range(5):
            try:
                query_result = pd.read_sql_query(query, conn)
                success = True
                break
            except Exception as sql_error_message:

                sql_error_prompt = get_error_prompt(
                    schema.to_string(), TABLE_NAME, question, query, str(sql_error_message))

                response = oaiapi.chat_completion(
                    prompt=sql_error_prompt,
                    model=OPENAI_LLM_MODEL_NAME)
                explanation, query = extract_query_from_response(response)
                if query is None:
                    conn.close()
                    return '', chat_history, None, '', explanation

        if success:
            db_res_prompt = get_db_agent_prompt(
                schema.to_string(), TABLE_NAME, question, query, explanation, query_result)

            chat_response = oaiapi.chat_completion(
                prompt=db_res_prompt,
                model=OPENAI_LLM_MODEL_NAME)

            chat_history.append((question, chat_response))

        else:
            query_result = ''

        conn.close()
        return '', chat_history, query_result, query, explanation

    conn = sqlite3.connect(db_file)

    with gr.Blocks() as demo:
        gr.Markdown('# Chat with your database demo')
        initial_query = 'select * from data;'
        initial_explanation = (
            'Interact with the data with natural language questions or commands. Some examples:\n\n' +
            'What fraction of customers are millennials?\n\n' +
            'Describe the data.\n\n' +
            'What are the unique states that customers live in?\n\n' +
            'What is the average of total assets across customers?\n\n'
        )
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown('## Query Result')
                dataframe = gr.Dataframe(
                    value=pd.read_sql_query(initial_query, conn))
            with gr.Column():
                gr.Markdown('## Schema')
                schema_box = gr.Dataframe(value=schema)

        query_text_box = gr.Textbox(value=initial_query, label='Last Query')
        explanation_text_box = gr.Textbox(
            value=initial_explanation, label='Explanation',
            lines=5,
            max_lines=20
        )
        chatbot = gr.Chatbot()
        question = gr.Textbox(
            value=DEFAULT_QUESTION, label='Ask a question about the data')

        with gr.Row():
            with gr.Column():
                ask_button = gr.Button('Ask')
            with gr.Column():
                clear = gr.ClearButton([question, chatbot])

        question.submit(
            respond,
            [question, chatbot],
            [question, chatbot, dataframe, query_text_box, explanation_text_box])

        ask_button.click(
            respond,
            [question, chatbot],
            [question, chatbot, dataframe, query_text_box, explanation_text_box])

    demo.launch()


if __name__ == '__main__':
    main()

