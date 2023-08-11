from collections.abc import Callable

from neosophia.llmtools import openaiapi as openai

def check_question(question: str, context: str, model: Callable, functions_descriptions: list[dict]) -> str:

    system_message = (
        "You are a helpful AI assistant.  Your job is to check the reasonableness "
        "of user questions.  If the user question can be answered given the tools "
        "available say, \"This is a reasonable question.\"  If the user question "
        "cannot be answered then provide some feedback to the user that may improve "
        "their question.\n\n"
        f"Here is the context for the question:\n{context}"
    )
    user_question = f'Question: {question}'
    messages = [
        openai.Message('system', system_message),
        openai.Message('user', user_question)
    ]

    response = model(messages, functions=functions_descriptions)

    return response.content


def critique_transcript(transcript: str, model: Callable, functions_descriptions: list[dict]) -> str:
    system_message = (
        "You are a helpful AI assistant.  Your job is to check the accuracy of a "
        "user-AI interaction.  If the interaction looks accurate say, \"This "
        "appears accurate.\"  If the interaction appears to have flaws then "
        "respond with suggestions on how to correct the mistakes."
    )
    user_question = f"Here is the transcript of the interaction:\n{transcript}"
    messages = [
        openai.Message('system', system_message),
        openai.Message('user', user_question)
    ]

    response = model(messages, functions=functions_descriptions)

    return response.content
