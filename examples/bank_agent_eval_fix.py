"""
Rerun evaluations on a CSV result.
"""

from typing import Callable
import os
import csv

import pandas as pd

from examples import bank_agent_eval as bav


def main():
    """Main program."""

    input_dir_path = 'eval_20231024_093904'
    output_dir_path = input_dir_path + '_fixed'
    output_file_prefix = 'eval'
    input_file_path = os.path.join(input_dir_path, f'{output_file_prefix}.csv')

    # read in previous results
    rows = []
    with open(input_file_path, 'r') as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        header = next(reader)
        print('header:', header)
        for row in reader:
            rows.append(row)

    # recalculate whether each answer is correct and write back out
    os.makedirs(output_dir_path, exist_ok=True)
    output_file_path = os.path.join(output_dir_path, f'{output_file_prefix}.csv')
    with open(output_file_path, 'w') as csv_file:
        csv_file.write(','.join(header) + '\n')
        for row in rows:
            system, question, run, calls, time, missing, correct, answer = row
            if answer is not None:
                answer = answer.replace('%%%', '\n').replace('``', '"')
            eval_func = find_eval_func(question)
            correct = eval_func(answer)

            # assemble new line using the correct flag
            if answer is not None:
                answer = answer.replace('\n', '%%%').replace('"', '``')
            line = [
                f'"{system}"',
                f'"{question}"',
                int(run),
                int(calls),
                float(time),
                missing == 'True',
                str(correct),
                f'"{answer}"'
            ]
            line = [str(x) for x in line]
            line = ','.join(line) + '\n'
            csv_file.write(line)

    # read csv back in with pandas to aggregate

    df = pd.read_csv(output_file_path)

    df_system_question = df.groupby(['system', 'question']).agg(
        func={'calls': 'mean', 'time': 'mean', 'missing': 'mean', 'correct': 'mean'})
    df_system = df.groupby(['system']).agg(
        func={'calls': 'mean', 'time': 'mean', 'missing': 'mean', 'correct': 'mean'})

    df_system_question.to_csv(
        os.path.join(
            output_dir_path,
            f'{output_file_prefix}_system_question.csv'
        ),
        float_format='{:.3f}'.format
    )
    df_system.to_csv(
        os.path.join(
            output_dir_path,
            f'{output_file_prefix}_system.csv',
        ),
        float_format='{:.3f}'.format
    )

    print('wrote aggregate CSVs')


def find_eval_func(question: str) -> Callable:
    for x, y in bav.QS_AND_EVALS:
        if x == question:
            return y


if __name__ == '__main__':
    main()
