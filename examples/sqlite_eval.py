"""
Evaluate the SQLLite question-answering system.
"""

import sqlite3
from typing import List, Optional, Tuple, Callable, Any

import click
import pandas as pd

from neosophia.llmtools import openaiapi as oaiapi

from examples import project
from examples import sqlite_chat as sc


OPENAI_LLM_MODEL_NAME = 'gpt-4'


def question_answer(
        question: str,
        db_file: str,
        schema: str
    ) -> Optional[str]:

    """
    Ask a question, get answer. This is basically copied from
    the respond fucntion in sqlite_chat with minimal changes
    """

    conn = sqlite3.connect(db_file)

    user_prompt = sc.get_user_agent_prompt(schema, sc.TABLE_NAME, question)

    ua_response = oaiapi.chat_completion(
        prompt=user_prompt,
        model=OPENAI_LLM_MODEL_NAME)
    explanation, query = sc.extract_query_from_response(ua_response)

    if query is None:
        conn.close()
        return None

    success = False
    for _ in range(5):
        try:
            query_result = pd.read_sql_query(query, conn)
            success = True
            break
        except Exception as sql_error_message:

            sql_error_prompt = sc.get_error_prompt(
                schema, sc.TABLE_NAME, question, query, str(sql_error_message))

            response = oaiapi.chat_completion(
                prompt=sql_error_prompt,
                model=OPENAI_LLM_MODEL_NAME)
            explanation, query = sc.extract_query_from_response(response)
            if query is None:
                conn.close()
                return None

    if success:
        db_res_prompt = sc.get_db_agent_prompt(
            schema, sc.TABLE_NAME, question, query, explanation, query_result)

        chat_response = oaiapi.chat_completion(
            prompt=db_res_prompt,
            model=OPENAI_LLM_MODEL_NAME)
    else:
        chat_response = None

    return chat_response



def evaluate(qa_func: Callable, tests: List[Tuple[str, Any]]):

    bleurt_results = []
    numeric_results = []

    from neosophia.llmtools import bleurt

    model_and_tokenizer = bleurt.load_bleurt(
        model_name=bleurt.BLEURT_20_D12,
        cache_dir_path=project.MODELS_DIR_PATH
    )

    import re

    for question, expected_answer in tests:
        answer = qa_func(question)

        if isinstance(expected_answer, str):

            scores = bleurt.compare(
                *model_and_tokenizer,
                references=[answer],
                candidates=[expected_answer]
            )
            score = scores.item()
            bleurt_results.append((
                question, answer, expected_answer, score
            ))

        else:
            # extract a numeric response and compare

            values = re.findall('[\d,]+\.?\d*', answer)
            if values:
                value = float(values[0])
            else:
                value = None

            # TODO: some kind of score?
            numeric_results.append((
                question, answer, value, expected_answer
            ))

    print('BLEURT results:')
    for x in bleurt_results:
        print(x)

    print('Numeric results:')
    for x in numeric_results:
        print(x)


@click.command()
@click.option(
    '--csv_file', '-c',
    default=f'{project.DATASETS_DIR_PATH}/bank_customers.csv')
def main(csv_file: str):
    """Main program."""

    key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    oaiapi.set_api_key(key)

    db_file, schema = sc.setup(csv_file)

    qa_func = lambda x: question_answer(x, db_file, schema)

    tests = [
        (
            'How many customers are millennials?',
            # 'Three customers are millennials.'
            'Ten customers are millennials'  # incorrect
        ),
        (
            'Which customer has the most money?',
            'The customer with the most money Oscar White, with $58000.0.'
        ),
        (
            'What is the average amount in a Roth IRA?',
            27400.275
        )
    ]

    evaluate(qa_func, tests)


if __name__ == '__main__':
    main()
