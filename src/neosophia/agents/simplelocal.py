"""
Simple agent with function calling designed for use with a local LLM.
"""

from typing import List, Callable, Generator

import llama_cpp

from neosophia.llmtools import dispatch as dp
from neosophia.llmtools import openaiapi as openai
from neosophia.agents import react


MAX_TOKENS = 4096


FORMAT_MESSAGE = (
    "When the user asks a question, think about what to do before responding. "
    "Share your thoughts with the user so they understand what you are doing. "
    "You can use a function call to get additional information from the user. "
    "When you have the final answer say, \"Final Answer: \" followed by the "
    "resposne to the user's question."
)


def make_simple_agent(
        system_message: str,
        model: Callable,
        function_descriptions: dict[str, dp.FunctionDesc],
        functions: dict[str, Callable],
        max_llm_calls: int,
        simple_formatting: bool
        ) -> Callable:
    """"""

    # system_message += '\n\n' + FORMAT_MESSAGE

    messages = [
        openai.Message('user', system_message)
    ]

    print('SYSTEM MESSAGE:')
    print(system_message)
    print()

    def run_once(user_input: str) -> Generator:
        """Engage an LLM ReACT agent to answer a question."""

        # input_msg = f'Question: {user_input}'
        input_msg = dp.dispatch_prompt(
            question=user_input,
            functions=function_descriptions
        )

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

            next_message, function_called = react.get_next_message(
                response,
                functions,
                simple_formatting
            )
            if next_message.role == 'function':
                next_message.role = 'user'
            function_call_counter += function_called

            messages.append(next_message)
            yield messages[-1]

    return run_once


def build_llama_wrapper(
        llama_model: llama_cpp.Llama
        ) -> Callable:

    # https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama.py

    def run(
            messages: List[openai.Message],
            functions: dict[str, dp.FunctionDesc]
            ) -> openai.Message:
        """do it"""

        # TODO: we may have to do a custom prompt

        try:
            result = llama_model.create_chat_completion(
                messages=[
                    llama_cpp.ChatCompletionMessage(role=x.role, content=x.content)
                    for x in messages
                ],
                temperature=0.7,
                repeat_penalty=1.1,
                max_tokens=MAX_TOKENS
            )

            response = result['choices'][0]['message']['content']
            print('RESPONSE:')
            print(response)
            print('-' * 50)

            if 'FUNCTION:' in response:
                function_call = dp.parse_dispatch_response(response, functions)
                # OpenAI format
            else:
                function_call = None
        except Exception as e:
            print(str(e))
            response = None
            function_call = None

        if function_call is not None:
            function_call = {
                'name': function_call[0],
                'arguments': function_call[1]
            }

        # construct a response message

        return openai.Message(
            role='assistant',
            content=response if response is not None else '',
            function_call=function_call
        )

    return run
