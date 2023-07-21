"""
Llama 2 stuff using llama-cpp-python.
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
            # prompt='[INST]What is the capital of France?[/INST]',
            # prompt='[INST]What is the difference between SO(3) and Spin(3)?[/INST]',
            prompt=f'[INST]{text}[/INST]',
            temperature=0.7,
            repeat_penalty=1.1,
            max_tokens=max_tokens
        )
        print(output)
        answer = output['choices'][0]['text']
    except:
        answer = None

    return answer
