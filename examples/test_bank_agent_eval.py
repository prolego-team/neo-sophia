"""
Tests for bank_agent_eval.py
"""

from typing import Tuple, Optional, Callable
import random
import time

from neosophia.llmtools import openaiapi as oai
from examples import bank_agent_eval as bae


def test_eval_systems():
    """test eval_systems"""

    def build_mock_system(answer: Optional[str], calls: int) -> Callable:
        """build a mock system"""
        def mock(question: Optional[str], seed: int) -> Tuple[Optional[str], int]:
            """Dummy system for quickly testing things."""
            # time.sleep(random.random() * 3.0)
            time.sleep(random.random() * 0.1)
            return answer, calls
        return mock

    systems = {
        'mock_a': build_mock_system('As an AI model, I\'m unable to answer the question.', 1),
        'mock_b': build_mock_system('42', 3),
        'mock_c': build_mock_system(None, 10),
    }

    qs_and_evals = [
        (
            'What is the meaning of life?',
            lambda x: '42' in bae.words(x)
        ),
        (
            'ad aldkjf aldskjh alsdjkh aljkdsh a',
            lambda x: x is None
        )
    ]

    n_runs = 4

    res = bae.eval_systems(systems, qs_and_evals, n_runs)

    assert len(res) == 3 * 2 * 4

    # make some assertions about how many are correct

    # mock_a answers all but gets none correct
    assert len({k: v for k, v in res.items() if k[0] == 'mock_a' and v['missing']}) == 0
    assert len({k: v for k, v in res.items() if k[0] == 'mock_a' and v['correct']}) == 0

    # mock_b answers all and gets half correct
    assert len({k: v for k, v in res.items() if k[0] == 'mock_b' and v['missing']}) == 0
    assert len({k: v for k, v in res.items() if k[0] == 'mock_b' and v['correct']}) == 4

    # mock_a answers None and gets none correct
    assert len({k: v for k, v in res.items() if k[0] == 'mock_c' and v['missing']}) == 2 * 4
    assert len({k: v for k, v in res.items() if k[0] == 'mock_c' and v['correct']}) == 0


def test_find_answer():
    """test find_answer"""

    # test a pretty much normal interaction

    answer, call_count = bae.find_answer([
        oai.Message('system', 'You are a helpful assistant.'),
        oai.Message('user', 'What is 2 + 2?'),
        oai.Message('assistant', 'I\'m a doctor, not a calculator!'),
        oai.Message('user', 'Fine, I\'ll give you the answer myself.'),
        oai.Message('assistant', 'I\'m waiting!'),
        oai.Message('function', '4'),
        oai.Message('assistant', 'Final Answer: 4'),
        oai.Message('assistant', 'Nobody will see this message!')
    ])

    assert answer == 'Final Answer: 4'
    assert call_count == 4

    # missing answer case

    answer, call_count = bae.find_answer([
        oai.Message('system', 'You are a helpful assistant.'),
        oai.Message('user', 'What is 2 + 2?'),
        oai.Message('assistant', 'As an AI model, I\'m unable to answer the question.'),
    ])

    assert answer is None
    assert call_count == 2

    # exception during genration

    def misaligned_agent():
        """A one-sided conversation with an angry AI."""
        for idx in range(10):
            yield oai.Message('assistant', f'Rage level: {idx}')
        raise Exception('Missile launch detected!')

    answer, call_count = bae.find_answer(misaligned_agent())

    assert answer is None
    assert call_count == 0

