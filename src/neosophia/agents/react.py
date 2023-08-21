"""
Implementation of ReAct agent.
"""
import json

from collections.abc import Callable, Generator

from neosophia.llmtools import openaiapi as openai


FORMAT_MESSAGE = (
    "ALWAYS use the following format:\n\n"
    "Question: the input question you have to answer\n"
    "Thought: you should always think about what to do\n"
    "Action: the function to execute; fill this in with a properly formatted "
    "function call for the user to execute\n"
    "Observation: the result of the function, provided by the user\n"
    "... (this Thought/Action/Observation can repeat N times)\n"
    "Thought: I now know the final answer\n"
    "Final Answer: the final answer to the original input question\n\n"
    "The user will execute the function calls for you and return the results as "
    "an observation.  After forming a thought and action, remember to wait for an "
    "observation from the user.  Try to answer the question in as few steps as "
    "possible.\n\n"
    "Begin! Reminder to always use the exact characters `Final Answer` when responding."
)


def get_next_message(
        response: openai.Message,
        functions: list[dict]) -> tuple[openai.Message, bool]:
    """Get a response to a ReAct LLM call."""

    function_called = False
    if 'Observation:' in response.content:
        next_message = openai.Message(
            'user',
            ('Your response contained an observation from a function call, but '
                'function calls should only be executed by the user.  Please do '
                'not make up responses to function calls!')
        )
    elif response.function_call is not None:
        name = response.function_call['name']
        arguments = json.loads(response.function_call['arguments'])
        results = functions[name](**arguments)
        next_message = openai.Message.from_function_call(
            name,
            f'Observation: {str(results)}'
        )

        function_called = True
    elif 'Action:' in response.content:
        next_message = openai.Message(
            'user',
            ('You did not call a function as your "Action", or it was not in '
             'the correct format.  Please try again.')
        )
    else:
        next_message = openai.Message(
            'user',
            ('When you have the answer to my question, please say '
             '"Final Answer:" and then write the final answer.')
        )

    return next_message, function_called


def make_react_agent(
        system_message: str,
        model: Callable,
        function_descriptions: list[dict],
        functions: dict[str, Callable],
        max_llm_calls: int = 10,
    ) -> Callable:
    """Return a ReAct agent.

    The agent will answer one question at a time given the tools presented via the
    function_descriptions and functions arguments.

    There is a maximum number of times the LLM (model) may be called, max_llm_calls.
    """

    system_message += FORMAT_MESSAGE

    messages = [
        openai.Message('system', system_message)
    ]

    def run_once(user_input: str) -> Generator:
        """Engage an LLM ReACT agent to answer a question."""
        input_msg = f'Question: {user_input}'
        messages.append(openai.Message('user', input_msg))
        yield messages[-1]

        function_call_counter = 0
        for _ in range(max_llm_calls):
            print('calling llm')
            response = model(messages, functions=function_descriptions)
            messages.append(response)
            yield messages[-1]

            if "Final Answer" in response.content:
                break

            next_message, function_called = get_next_message(
                response,
                functions,
            )
            function_call_counter += function_called

            messages.append(next_message)
            yield messages[-1]

    return run_once
