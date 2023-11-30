"""
Simple agent with function calling designed for use with a local LLM.
"""

from typing import List, Callable, Generator, Optional
import json

import llama_cpp

from neosophia.llmtools import dispatch as dp
from neosophia.llmtools import openaiapi as openai
from neosophia.agents import react_chat


LLAMA2_MAX_TOKENS = 4096


FORMAT_MESSAGE = (
    "When the user asks a question, think about what to do before responding. "
    
    # Try to prevent chattiness or the model trying get more user input.
    # "Share your thoughts with the user so they understand what you are doing. "
    "Briefly share your thoughts, but do not engage in conversation. "
    
    "You can use a function call to get additional information from the user. "
    "The user will run the function and give you the answer with an \"Observation:\" prefix. "
        
    "Do not show example function output. "
    
    # Original "Final Answer" instructions
    # "When you have the final answer say, \"Final Answer: \" followed by the response to the user's question. "   
    
    # May help some models that can't follow format instructions so well.
    # "Do not say \"Final Answer\" without getting an observation from the user."  # !!!! !!!! !!!
    
    "When you have gathered enough observations from the user to answer the question, "
    "say, \"Final Answer: \" followed by the answer. Do not make up an answer."
)

CUSTOM_DISPATCH_PROMPT_PREFIX = (
    # This allows adapting the custom dispatch prompt for use with an agent.
    'Find information to help answer the question by choosing a single function ' +
    'and generating parameters for the function based on the function descriptions below. ' +
    'Do not ask the user questions. ' +
    'Do not ask the user to add columns or tables to the database. ' +

    # Reminders about the expected output format.
    'Do not write code or expressions. Follow the output format EXACTLY! ' +
    'Only choose one function and generate one set of parameters! ' +
    'Make sure to put each parameter on a separate line prefixed by "PARAMETER:"'
)


def make_simple_agent(
        system_message: str,
        dp_message: str,
        model: Callable,
        function_descriptions: dict[str, dp.FunctionDesc],
        functions: dict[str, Callable],
        max_llm_calls: int,
        debug_func: Optional[Callable]
        ) -> Callable:
    """Simple agent using a local LLM based on Justin's react agent simple mode."""

    # I'm following Justin's agent code here.
    # One improvement would be to move this
    # message initialization stuff inside `run_once` and then
    # `run_once` could be reused.

    system_message += '\n\n' + FORMAT_MESSAGE
    messages = [
        openai.Message('system', system_message)
    ]

    print('SYSTEM MESSAGE:')
    print(system_message)
    print('-' * 50)
    print()

    def run_once(user_input: str) -> Generator:
        """Engage an LLM ReACT agent to answer a question."""

        dp.DISPATCH_PROMPT_PREFIX = dp_message
        input_msg = dp.dispatch_prompt(
            question=user_input,
            functions=function_descriptions
        )

        messages.append(openai.Message('user', input_msg))
        yield messages[-1]

        function_call_counter = 0
        step = 0
        for _ in range(max_llm_calls):
            print('calling llm')
            response = model(messages, functions=function_descriptions)
            if debug_func is not None:
                # optionally save LLM input and output for debugging
                debug_func(step, messages, response)

            messages.append(response)
            yield messages[-1]

            # need to check "final answer" in lowercase since llama is not good
            # at following formatting directions
            # ignore "final answer" if there is a function call
            if 'final answer' in response.content.lower() and response.function_call is None:
                break

            try:
                next_message, function_called = react_chat.get_next_message(
                    response,
                    functions
                )
                # Reminders to follow function call / final answer format after
                # a successful function call.
                next_message.content += (
                    '\n\nIf you need to fix the function call or make a new one ' +
                    'use the format described above EXACTLY. ' +
                    'If you have enough information for the final answer, ' +
                    'say \"Final Answer:\" followed by the answer to the user\'s question ' +
                    'instead of generating a function call. ' +
                    'Do not ask the user questions, give the user commands, ' +
                    'or ask the user to fix the database or queries.'
                )
                function_call_counter += function_called
            except:
                # Reminders about the custom dispatch format if the function call fails.
                next_message = openai.Message(
                    'user',
                    'Error calling function. \n'
                    'Describe the function and parameters in this format EXACTLY!\n' +
                    'EXPLANATION: [brief explanation of your answer]\n' +
                    'FUNCTION: [function_name] \n' +
                    'PARAMETER: [name_0]=[value_0] \n' +
                    'PARAMETER: [name_1]=[value_1] \n' +
                    '...\n\n' +
                    'Make sure to put each parameter on a separate line prefixed by "PARAMETER:"'
                )

            messages.append(next_message)
            yield messages[-1]

            step += 1

    return run_once


def build_llama2_wrapper(
        llama_model: llama_cpp.Llama,
        messages_to_prompt: Callable,
        stop: List[str]
        ) -> Callable:
    """
    Wrap a llama_cpp model to take messages and function descriptions as input
    and return a message object.
    """

    def run(
            messages: List[openai.Message],
            functions: dict[str, dp.FunctionDesc]
            ) -> openai.Message:
        """do it"""

        try:
            # build the prompt manually
            prompt = messages_to_prompt(messages)

            output = llama_model(
                prompt=prompt,
                temperature=0.7,
                repeat_penalty=1.1,
                max_tokens=LLAMA2_MAX_TOKENS,
                stop=stop
            )
            response = output['choices'][0]['text']

            print('RESPONSE:')
            print(response)
            print('-' * 50)
            print()

            if dp.FUNC_PREFIX_LOWER in response.lower():
                function_call = dp.parse_dispatch_response(response, functions)
            else:
                function_call = None

        except Exception as e:
            print('Exception:', str(e))
            response = None
            function_call = None

        # build the OpenAI function call result format, expected by the functions in `react_chat`
        if function_call is not None:
            function_call = {
                'name': function_call[0],
                'arguments': json.dumps(function_call[1])
            }

        # construct a response message

        return openai.Message(
            role='assistant',
            content=response if response is not None else '',
            function_call=function_call
        )

    return run


def build_debug_func(prefix: str, promptformat: Callable) -> Callable:
    """
    Build a function for debugging that formats
    messages / response and writes to file.
    """
    def call(step: int, messages: List[openai.Message], response: openai.Message):
        """do it"""
        with open(prefix + f'_step_{step}.txt', 'w') as fp:
            fp.write('~~~~ ~~~~ ~~~~ ~~~~ PROMPT ~~~~ ~~~~ ~~~~ ~~~~\n')
            fp.write(promptformat(messages))
            fp.write('\n~~~~ ~~~~ ~~~~ ~~~~ RESPONSE ~~~~ ~~~~ ~~~~ ~~~~\n')
            fp.write(response.content)
    return call
