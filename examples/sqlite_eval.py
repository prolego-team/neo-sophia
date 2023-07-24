"""
Evaluate the SQLLite question-answering system.
"""

import sqlite3
from typing import List, Optional, Tuple, Callable, Any

import click
import numpy as np
import pandas as pd

from neosophia.llmtools import openaiapi as oaiapi

from examples import project
from examples import sqlite_chat as sc

from neosophia.llmtools.util import Colors


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


def evaluate(
        qa_func: Callable,
        tests: List[Tuple[str, Any]],
        bleurt_thresh: float,
        delta_thresh: float
        ) -> Tuple[int, int]:
    """evaluate model and print results"""

    bleurt_results = []
    numeric_results = []

    from neosophia.llmtools import bleurt

    model_and_tokenizer = bleurt.load_bleurt(
        model_name=bleurt.BLEURT_20_D12,
        cache_dir_path=project.MODELS_DIR_PATH
    )

    import re

    import tqdm

    for question, expected_answer in tqdm.tqdm(tests):
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
            values = re.findall(r'[\d,]+\.?\d*', answer)
            try:
                value = values[0]
                value = value.replace(',', '')
                value = float(value)
                delta = value - expected_answer
            except:
                value = None
                delta = np.nan

            # TODO: some kind of score?
            numeric_results.append((
                question, answer, value, expected_answer, delta
            ))

    correct = 0
    missing = 0

    print('BLEURT results:')
    for idx, (question, answer, expected_answer, score) in enumerate(bleurt_results):
        print(f'{idx}. {question}')
        if score > bleurt_thresh:
            correct += 1
            color = Colors.GREEN
        else:
            color = Colors.RED
        print(f'\t{color}`{answer}` vs `{expected_answer}` (expected) {round(score, 3)}{Colors.RESET}')

    print('Numeric results:')
    for idx, (question, answer, value, expected_answer, delta) in enumerate(numeric_results):
        # Not sure what the right way to do a binary result here is
        if delta is not None and np.abs(delta) < delta_thresh:
            correct += 1
            color = Colors.GREEN
        else:
            if delta is None:
                missing += 1
            color = Colors.RED
        print(f'{idx}. {question}')
        print(f'\t{color}`{answer}` vs `{expected_answer}` (expected) {round(delta, 3)}{Colors.RESET}')

    return correct, missing


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
            'Three customers are millennials.'
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

    correct, missing = evaluate(
        qa_func, tests, 0.35, 1.0)

    print('correct:', correct, '/', len(tests), '=', round((correct * 1.0) / len(tests), 3))
    print('missing:', missing, '/', len(tests), '=', round((missing * 1.0) / len(tests), 3))


if __name__ == '__main__':
    main()
