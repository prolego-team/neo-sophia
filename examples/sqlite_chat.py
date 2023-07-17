"""
"""
import os
import re
import csv
import pprint
import sqlite3
import readline

import click
import gradio as gr
import pandas as pd

from examples import project
from neosophia.llmtools import openaiapi as oaiapi

OPENAI_LLM_MODEL_NAME = 'gpt-4'

opj = os.path.join

TABLE_NAME = 'data'


def create_database_from_csv(conn, csv_file, db_file):
    """ """

    # Connect to the SQLite database
    cursor = conn.cursor()

    # Read the CSV file and insert data into the database
    with open(csv_file, mode="r") as file:
        reader = csv.reader(file)
        header = next(reader)

        # Create the table in the database based on the CSV header
        columns = ", ".join(header)
        drop_table_query = f"DROP TABLE IF EXISTS {TABLE_NAME}"
        cursor.execute(drop_table_query)

        create_table_query = f"CREATE TABLE {TABLE_NAME} ({columns})"
        cursor.execute(create_table_query)

        # Insert rows into the table
        insert_query = f"INSERT INTO {TABLE_NAME} VALUES ({', '.join(['?'] * len(header))})"
        for row in reader:
            cursor.execute(insert_query, row)

    # Commit the changes
    conn.commit()

    print("Database created and data inserted successfully.")


def get_schema(conn):
    """ """
    cursor = conn.cursor()
    query = f"PRAGMA table_info({TABLE_NAME});"
    return pd.read_sql_query(query, conn).to_string()


@click.command()
@click.option(
    '--csv_file', '-c',
    default=f'{project.DATASETS_DIR_PATH}/bank_customers.csv')
def main(csv_file):
    """ """

    # TODO
    # - Error checking
    # - More context
    # - Multiple tables/queries
    # - Speculate what the user wants based on their failed queries

    db_file = opj(project.DATASETS_DIR_PATH, 'bank_database.db')
    conn = sqlite3.connect(db_file)
    create_database_from_csv(conn, csv_file, db_file)

    schema = get_schema(conn)

    user_agent_bp = 'Given the following SQLite schema and question, write a SQL query to retrieve data that answers the question. Return the answer in the following format, and do not include "sql" or "SQL" as part of the SQL query:'
    user_agent_bp += 'Explanation: [explain query]'
    user_agent_bp += '```[sql query]```'
    user_agent_bp += '\n------------------------------------------------------\n'

    schema_prompt = f'TABLE NAME: {TABLE_NAME}\n'
    schema_prompt += f'TABLE SCHEMA:\n'
    schema_prompt += schema + '\n\n'

    user_agent_bp += schema_prompt

    first_response = oaiapi.chat_completion(
        prompt=user_agent_bp,
        model=OPENAI_LLM_MODEL_NAME)

    db_agent_bp = 'Given the following schema, question, SQL query, and result from executing the query, answer the question.\n\n'


    def respond(question, chat_history):

        conn = sqlite3.connect(db_file)
        user_prompt = user_agent_bp + question

        response = oaiapi.chat_completion(
            prompt=user_prompt,
            model=OPENAI_LLM_MODEL_NAME)

        explanation = response.split('```')[0]

        pattern = r"```(SQLite|SQLITE|sqlite|sql|SQL|\n|)((.|\n)*)```"
        matches = re.search(pattern, response, re.DOTALL)
        if matches:
            query = ' '.join(matches.group(2).split('\n')).rstrip().strip()
        else:
            query = ' '.join(response.split('query:')[1:]).rstrip().strip()

        print('\n----------------RESPONSE_START----------------\n')
        print(response)
        print('\n----------------RESPONSE_END----------------\n')

        print('\n----------------QUERY_START----------------\n')
        print(query)
        print('\n----------------QUERY_END----------------\n')
        print('\n===========================================================\n')

        if query == '':
            chat_history.append((question, response))
            return "", chat_history, None, ''

        res = pd.read_sql_query(query, conn)

        db_res_prompt = db_agent_bp + schema_prompt
        db_res_prompt += 'question: ' + question + '\n'
        db_res_prompt += 'SQL query: ' + query + '\n'
        db_res_prompt += 'SQL result: ' + res.to_string()

        response = oaiapi.chat_completion(
            prompt=db_res_prompt,
            model=OPENAI_LLM_MODEL_NAME)

        chat_history.append((question, response))

        conn.close()

        return "", chat_history, res, query, explanation

    with gr.Blocks() as demo:
        gr.Markdown('# VaultVibe')
        dataframe = gr.Dataframe()
        query_text_box = gr.Textbox(label='Last Query')
        explanation_text_box = gr.Textbox(label='Explanation')
        chatbot = gr.Chatbot()
        question = gr.Textbox()
        clear = gr.ClearButton([question, chatbot])

        question.submit(
            respond,
            [question, chatbot],
            [question, chatbot, dataframe, query_text_box, explanation_text_box])

    demo.launch(share=True)



if __name__ == '__main__':
    main()

