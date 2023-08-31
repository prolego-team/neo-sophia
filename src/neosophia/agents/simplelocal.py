"""
Simple agent with function calling designed for use with a local LLM.
"""

from typing import List, Callable, Generator
import json

import llama_cpp

from neosophia.llmtools import dispatch as dp
from neosophia.llmtools import openaiapi as openai
from neosophia.agents import react_chat


LLAMA2_MAX_TOKENS = 4096


FORMAT_MESSAGE = (
    "When the user asks a question, think about what to do before responding. "
    # "Share your thoughts with the user so they understand what you are doing. "
    "Briefly share your thoughts, but do not engage in conversation. "
    "You can use a function call to get additional information from the user. "
    "The user will run the function and give you the answer with an \"Observation:\" prefix. "
    "Do not run functions yourself. "
    "When you have the final answer say, \"Final Answer: \" followed by the "
    "response to the user's question."
)

CUSTOM_DISPATCH_PROMPT_PREFIX = (
    'Find information to help answer the question by choosing a single function ' +
    'and generating parameters for the function based on the function descriptions below. ' +
    'Do not ask the user questions.'
)


def make_simple_agent(
        system_message: str,
        model: Callable,
        function_descriptions: dict[str, dp.FunctionDesc],
        functions: dict[str, Callable],
        max_llm_calls: int
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
    print()

    def run_once(user_input: str) -> Generator:
        """Engage an LLM ReACT agent to answer a question."""

        dp.DISPATCH_PROMPT_PREFIX = CUSTOM_DISPATCH_PROMPT_PREFIX
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

            # llama2 tends to capitalize "Final Answer", maybe due to
            # capitals in dispatch prompt? So we'll do the check
            # in lowercase
            if 'final answer' in response.content.lower():
                break

            next_message, function_called = react_chat.get_next_message(
                response,
                functions
            )
            function_call_counter += function_called

            messages.append(next_message)
            yield messages[-1]

    return run_once


def build_llama2_wrapper(
        llama_model: llama_cpp.Llama
        ) -> Callable:
    """
    Wrap a llama_cpp model to take messages and function descriptions as input
    and return a message.
    """

    def run(
            messages: List[openai.Message],
            functions: dict[str, dp.FunctionDesc]
            ) -> openai.Message:
        """do it"""

        try:
            if False:
                # use the chat completion interface, doesn't work well
                # I believe the llama2 chat models need special formatting
                # https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama.py
                # vs.
                # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212
                result = llama_model.create_chat_completion(
                    messages=[
                        llama_cpp.ChatCompletionMessage(role=x.role, content=x.content)
                        for x in messages
                    ],
                    temperature=0.7,
                    repeat_penalty=1.1,
                    max_tokens=LLAMA2_MAX_TOKENS
                )
                response = result['choices'][0]['message']['content']
            else:
                # build the prompt manually
                prompt = messages_to_llama2_prompt(messages)

                hash_str = str(hash(prompt))
                prompt_file_path = f'prompt_{hash_str}.txt'
                with open(prompt_file_path, 'w') as fp:
                    fp.write(prompt)

                output = llama_model(
                    prompt=prompt,
                    temperature=0.7,
                    # repeat_penalty=1.1,
                    repeat_penalty=1.2,
                    max_tokens=LLAMA2_MAX_TOKENS,
                    stop=['[INST', '[Inst', '### ', '<|im_end|>', '<|im_start|>']
                )
                response = output['choices'][0]['text']

            print('RESPONSE:')
            print(response)
            print('-' * 50)

            if 'FUNCTION:' in response:
                function_call = dp.parse_dispatch_response(response, functions)
            else:
                function_call = None

        except Exception as e:
            print('Exception:', str(e))
            response = None
            function_call = None

        # build the OpenAI function call result format
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


def messages_to_llama2_prompt(messages: List[openai.Message]) -> str:
    """
    Convert a list of messages representin a chat conversation
    to a llama2 prompt.
    """

    # Reference on how it's officially done:
    # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212
    # What I've implemented isn't quite the same, there's some
    # additional special tokenization stuff going on in the original method.
    # But this should be close.

    # assumes the first messages are system and user
    assert len(messages) > 1, 'not enough messages'
    assert messages[0].role == 'system', 'wrong role for first message'
    assert messages[1].role == 'user', 'wrong role for second message'

    # special case for putting sys tags inside INST
    messages_text = [
        f'[INST] <<SYS>>\n{messages[0].content}\n<</SYS>>\n\n{messages[1].content} [/INST]'
    ]

    for message in messages[2:]:
        content = message.content.strip()
        if message.role != 'assistant':
            content = f'[INST] {content} [/INST]'
        messages_text.append(content)
    res = '\n\n'.join(messages_text)

    return res
