"""
Implementation of ReAct agent.
"""

from collections.abc import Callable
import json

from ..llmtools import openaiapi as openai


def make_react_agent(
        system_message: str,
        model: Callable,
        function_descriptions: dict[str, Callable],
        functions: list[dict],
        max_llm_calls: int | None = 10
    ) -> Callable:
    """Return a ReAct agent.
    
    The agent will answer one question at a time given the tools presented via the
    function_descriptions and functions arguments.
    
    There is a maximum number of times the LLM (model) may be called, max_llm_calls.
    """

    format_message = (
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
        "observation from the user.\n\n"
        "Begin! Reminder to always use the exact characters `Final Answer` when responding."
    )

    system_message += format_message

    messages = [
        openai.Message('system', system_message)
    ]

    def run_once(user_input: str) -> list[openai.Message]:
        """Engage an LLM ReACT agent to answer a question."""
        input_msg = f'Question: {user_input}'
        messages.append(openai.Message('user', input_msg))

        function_call_counter = 0
        for _ in range(max_llm_calls):
            print('calling llm')
            response = model(messages, functions=function_descriptions)
            messages.append(response)
            print(response)

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

                function_call_counter += 1
            else:
                next_message = openai.Message(
                    'user', 
                    ('You did not call a function as your "Action", or it was not in '
                     'the correct format.  Please try again.')
                )

            if ("Final Answer" in response.content) and (function_call_counter>0):
                break

            messages.append(next_message)

        return messages

    return run_once


"""
Implementation of ReAct agent.
"""

from collections.abc import Callable
import json

from ..llmtools import openaiapi as openai


def make_simple_react_agent(
        system_message: str,
        model: Callable,
        function_descriptions: dict[str, Callable],
        functions: list[dict],
        max_llm_calls: int | None = 2
    ) -> Callable:
    """Return a ReAct agent.
    
    The agent will answer one question at a time given the tools presented via the
    function_descriptions and functions arguments.
    
    There is a maximum number of times the LLM (model) may be called, max_llm_calls.
    """

    format_message = (
        "ALWAYS use the following format:\n\n"
        "Question: the input question you have to answer\n"
        "Thought: you should always think about what to do\n"
        "Action: the function to execute; fill this in with a properly formatted "
        "function call for the user to execute\n"
        "Observation: the result of the function, provided by the user\n"
        "Final Answer: the final answer to the original input question\n\n"
        "The user will execute the function call for you and return the results as "
        "an observation.  After forming a thought and action, remember to wait for an "
        "observation from the user.\n\n"
        "If you are not able to answer the question with one and only one function call "
        "then say, 'I cannot construct a Final Answer for this task' and explain why.\n\n"
        "Begin! Reminder to always use the exact characters `Final Answer` when responding."
    )

    system_message += format_message

    messages = [
        openai.Message('system', system_message)
    ]

    def run_once(user_input: str) -> list[openai.Message]:
        """Engage an LLM ReACT agent to answer a question."""
        input_msg = f'Question: {user_input}'
        messages.append(openai.Message('user', input_msg))

        function_call_counter = 0
        for _ in range(max_llm_calls):
            print('calling llm')
            response = model(messages, functions=function_descriptions)
            messages.append(response)
            print(response)

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

                function_call_counter += 1
            else:
                next_message = openai.Message(
                    'user', 
                    ('You did not call a function as your "Action", or it was not in '
                     'the correct format.  Please try again.')
                )

            if "Final Answer" in response.content:
                break

            messages.append(next_message)

        return messages

    return run_once
