import os
import sys
import logging

import neosophia.llmtools.openaiapi as openai

# === Basic setup ===================================================
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logging.getLogger('agent').setLevel(logging.DEBUG)
log = logging.getLogger('agent')
# ===================================================================

openai.set_api_key(os.getenv('OPENAI_API_KEY'))
model = openai.start_chat('gpt-3.5-turbo')
# log.info(openai.get_models_list())

print('Just kidding.')
messages = [
    openai.Message('system', 'You are a helpful assistant.'),
    openai.Message('user', 'Knock knock.'),
    openai.Message('assistant', 'Who is there?'),
    openai.Message('user', 'Orange.')
]
response = model(messages)
messages.append(response)
print(response)


print('Few shot prompting.')
messages = [
    openai.Message('system', 'You are a helpful assistant.'),
    openai.Message('user', 'Roger has 5 tennis balls.  He buys 2 more cans of tennis balls.  Each can has 3 balls.  How many balls does Roger have now?'),
    openai.Message('assistant', 'The answer is 11.'),
    openai.Message('user', 'The cafeteria has 23 apples originally.  If they used 20 to make lunch and bought 6 more, how many apples do they have?')
]
response = model(messages)
messages.append(response)
print(response)


print('Chain of though prompting.')
messages = [
    openai.Message('system', 'You are a helpful assistant.'),
    openai.Message('user', 'Roger has 5 tennis balls.  He buys 2 more cans of tennis balls.  Each can has 3 balls.  How many balls does Roger have now?'),
    openai.Message('assistant', 'Roger started with 5 balls.  2 cans of 3 tennis balls each is 6 tennis balls.  5 + 6 = 11.  The answer is 11.'),
    openai.Message('user', 'The cafeteria has 23 apples originally.  If they used 20 to make lunch and bought 6 more, how many apples do they have?')
]
response = model(messages)
messages.append(response)
print(response)


print('Your turn... have a conversation with an alien.')
messages = [
    openai.Message('system', 'You are an alien and are visiting earth for the first time.  You meet a human.  You want to be helpful but do not fully understand how the world works yet.'),
    openai.Message('assistant', 'Greetings, earthling!  How can I help you today?')
]
print(messages[-1].content)
while (user_input:=input()) != '':
    messages.append(openai.Message('user', user_input))
    response = model(messages)
    print(response.content)
    messages.append(response)