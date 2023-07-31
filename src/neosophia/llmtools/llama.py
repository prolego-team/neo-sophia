"""
Wrappers for running llama models.
"""

from typing import Optional

import llama_cpp


def load_llama2(model_file_path, context_tokens: int) -> llama_cpp.Llama:
    """Load llama 2 using llama-cpp-python"""
    res = llama_cpp.Llama(
        model_path=model_file_path,
        n_gpu_layers=10000,
        n_ctx=context_tokens
    )
    return res


def llama2_text(
        model: llama_cpp.Llama,
        text: str,
        max_tokens: int) -> Optional[str]:
    try:
        print('~~~~ PROMPT ~~~~ ~~~~ ~~~~')
        print(text)

        output = model(
            prompt=f'[INST]{text}[/INST]',
            temperature=0.7,
            repeat_penalty=1.1,
            max_tokens=max_tokens
        )
        answer = output['choices'][0]['text']
        print('~~~~ ANSWER ~~~~ ~~~~ ~~~~')
        print(answer)
        print('~~~~ ~~~~ ~~~~ ~~~~')
    except:
        answer = None

    return answer
