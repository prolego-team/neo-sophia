"""
"""
import os
import json
import readline

from abc import ABCMeta, abstractmethod
from typing import Dict, List

import neosophia.db.chroma as chroma

from examples import project

from neosophia.llmtools import dispatch as dp, openaiapi as oaiapi, pdf_utils
from neosophia.db.pdfdb import PDFDB

api_key = oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)

DESCRIPTIONS = {
    '1406.2661.pdf': """We propose a new framework for estimating generative
models via an adversarial process, in which we simultaneously train two models:
a generative model G that captures the data distribution, and a discriminative
model D that estimates the probability that a sample came from the training
data rather than G. The training procedure for G is to maximize the probability
of D making a mistake. This framework corresponds to a minimax two-player game.
In the space of arbitrary functions G and D, a unique solution exists, with G
recovering the training data distribution and D equal to 1 2 everywhere.  In
the case where G and D are defined by multilayer perceptrons, the entire system
can be trained with backpropagation.  There is no need for any Markov chains or
unrolled approximate inference networks during either training or generation of
samples. Experiments demonstrate the potential of the framework through
qualitative and quantitative evaluation of the generated samples.""",
    '1611.07004.pdf': """We investigate conditional adversarial networks as a
general-purpose solution to image-to-image translation problems. These networks
not only learn the mapping from input image to output image, but also learn a
loss function to train this mapping. This makes it possible to apply the same
generic approach to problems that traditionally would require very different
loss formulations. We demonstrate that this approach is effective at
synthesizing photos from label maps, reconstructing objects from edge maps, and
colorizing images, among other tasks. Indeed, since the release of the pix2pix
software associated with this paper, a large number of internet users (many of
them artists) have posted their own experiments with our system, further
demonstrating its wide applicability and ease of adoption without the need for
parameter tweaking. As a community, we no longer hand-engineer our mapping
functions, and this work suggests we can achieve reasonable results
without hand-engineering our loss functions either.""",
    '1704.00028.pdf': """Generative Adversarial Networks (GANs) are powerful
generative models, but suffer from training instability. The recently
proposedWasserstein GAN (WGAN) makes progress toward stable training of GANs,
but sometimes can still generate only poor samples or fail to converge. We find
that these problems are often due to the use of weight clipping in WGAN to
enforce a Lipschitz constraint on the critic, which can lead to undesired
behavior.  We propose an alternative to clipping weights: penalize the norm of
gradient of the critic with respect to its input. Our proposed method performs
better than standard WGAN and enables stable training of a wide variety of GAN
architectures with almost no hyperparameter tuning, including 101-layer ResNets
and language models with continuous generators.  We also achieve high quality
generations on CIFAR-10 and LSUN bedrooms."""
}

NO_CONVERSATION_CONSTRAINT = (
    'Do not engage in conversation or provide '
    'an explanation. Simply provide an answer.')

DEFAULT_SYSTEM_PROMPT = """You are a Unified Natural Language Query chatbot
(UNLQ-GPT) and your job is to assist a user in different tasks that require
gathering and interpreting data from multiple sources. The user will provide
the task, and it is your job to come up with a plan in order to provide what is
necessary given the available resources and constraints. Each step in your plan
must be tied to an available action you can take to execute the step.

Provide your output as outlined in the example below.

User Input:

------------------------------------COMMANDS------------------------------------
What is image-to-image translation?

-------------------------------------TOOLS-------------------------------------
name: find_pdf_by_keyword
description: Searches the `pdf_collection` by keyword and returns any matches that were found in the PDF description
parameter: name (str, required) - The collection name
parameter: keyword (str, required) - The keyword to search for

---------------------------------DATA RESOURCES---------------------------------
Resource Name: ChromaDB Database Collections
Resource Info: {
    "pdf_collection": {
        "name": "pdf_collection",
        "metadata": "{'description': 'Collection of PDF files with each item containing a description of the full PDF file'}"
    },
    "page_collection": {
        "name": "page_collection",
        "metadata": "{'description': 'collection of individual pages from PDF files'}"
    },
    "section_collection": {
        "name": "section_collection",
        "metadata": "{'description': 'collection of sections of pages from PDF files'}"
    }
}

-------------------------------------------------------------------------------

UNLQ-GPT Output:

Plan:
    Step 1.
    Action: Find relevant PDF objects by performing a keyword search on the descriptions provided for each object in the database.
    Function: find_pdf_by_keyword
    Parameter: keyword image-to-image

    Step 2.
"""

"""
User input:

--------------------------COMMAND--------------------------
James Smith has applied for a mortgage loan. Determine if there are any
immediate concerns that would disqualify him from obtaining a loan.

-------------------------------DATA RESOURCES-------------------------------
- SQLite Database:

Table Name: data
Table Schema:
   cid              name type  notnull dflt_value  pk
0    0              Name             0       None   0
1    1     Date_of_Birth             0       None   0
2    2           Address             0       None   0
3    3  Checking_Account             0       None   0
4    4   Savings_Account             0       None   0
5    5          ROTH_IRA             0       None   0

- ChromaDB database

Collections: pdf_collection, page_collection, section_collection

------------------------------TOOLS-------------------------------
Function Name: sqlite_query
Function Description: 'run a SQLite query on a database'
Function Params: {'query_str': ParamDesc(description='query string', typ=<class 'str'>, required=True)}

Function Name: find_pdf_by_keyword
Function Description: Returns the Lists the IDs of a collection in a ChromaDB database
Function Params: {'name': ParamDesc(description='The collection name', typ=<class 'str'>, required=True)}
-------------------------------------------------------------------------------

UNLQ-GPT Output:

Plan:
    1.
    Action: Query the SQLite database for account information regarding James
    Smith
    Function: `sqlite_query`

    2.
    Action: Search the `pdf_collection` for any forms related to James Smith
    that may cause concern for a loan, e.g., a bankruptcy form.
    Function: `find_pdf_by_keyword`

    3.


    4.
    Action: Aggregate the collected information into a concise summary.
    Function:


    5. Determine if James Smith should be considered for a loan.
"""


class Prompt:

    def __init__(self):

        self.commands = []
        self.data_resources = []
        self.tools = []
        self.examples = []
        self.constraints = []

    def add_command(self, command):
        self.commands.append(command)

    def add_example(self, example):
        self.examples.append(example)

    def add_data_resource(self, name, info):
        prompt = f'Resource Name: {name}\n'
        prompt += f'Resource Info: {info}\n\n'
        self.data_resources.append(prompt)

    def add_tool(self, resource):
        self.tools.append(resource)

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def generate_prompt(self):
        prompt = ''
        total = 80

        if self.commands:
            commands = '\n'.join(self.commands)
            til = '-' * int((total - len('COMMANDS')) / 2)
            prompt += f'{til}COMMANDS{til}\n{commands}\n\n'

        if self.tools:
            til = '-' * int((total - len('TOOLS')) / 2)
            tools = '\n'.join(self.tools)
            prompt += f'{til}TOOLS{til}\n{tools}\n\n'

        if self.data_resources:
            til = '-' * int((total - len('DATA RESOURCES')) / 2)
            data_resources = '\n'.join(self.data_resources)
            prompt += f'{til}DATA RESOURCES{til}\n{data_resources}\n\n'

        if self.constraints:
            til = '-' * int((total - len('CONSTRAINTS')) / 2)
            constraints = '\n'.join(self.constraints)
            prompt += f'{til}CONSTRAINTS{til}\n{constraints}\n\n'

        if self.examples:
            til = '-' * int((total - len('EXAMPLES')) / 2)
            prompt += f'{til}EXAMPLES{til}\n'
            for idx, example in enumerate(self.examples):
                prompt += f'EXAMPLE {idx + 1}:\n{example}\n'

        return prompt


class Agent:

    def __init__(self, tools: Dict, resources: Dict):

        self.tools = tools
        self.resources = resources

        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.openai_llm_model_name = 'gpt-4'

        self.max_tokens = 8192

        self.token_count = 0

        self.chat_history = [self.system_prompt]

    def execute(self, prompt):
        """ """
        print('Thinking...')

        if isinstance(prompt, Prompt):
            prompt = prompt.generate_prompt()

        #prompt = DEFAULT_SYSTEM_PROMPT + '\n' + prompt

        print('PROMPT')
        print(prompt)
        print('-' * 80)
        return oaiapi.chat_completion(
            prompt=prompt,
            model=self.openai_llm_model_name)

    def answer_question(self, question, context, constraint):
        """ """
        prompt = Prompt()
        prompt.add_command(question)
        prompt.add_data_resource(context, '')

        if constraint is not None:
            prompt.add_constraint(constraint)

        prompt_str = prompt.generate_prompt()
        print('PROMPT')
        print(prompt_str, '\n---\n')
        print('ANSWER')
        return self.execute(prompt_str)

    def chat(self):

        while True:

            prompt = Prompt()

            #user_input = input('> ')
            user_input = 'What is a GAN?'

            prompt.add_command(user_input)

            function_str_list = self.build_str_from_tools()
            for function_str in function_str_list:
                prompt.add_tool(function_str)

            for name, info in self.resources.items():
                prompt.add_data_resource(name, info)

            out = self.execute(prompt)
            print(out)
            exit()

    def build_str_from_tools(self):

        def _param_str(pname: str, pdesc: dp.ParamDesc) -> str:
            """
            Make a string to represent a parameter name and description.
            """
            required_str = '' if not pdesc.required else ', required'
            return f'{pname} ({pdesc.typ.__name__}{required_str}) - {pdesc.description}'

        functions = {k: v[1] for k, v in self.tools.items()}
        functions_str_list = []
        for name, desc in functions.items():
            functions_str_list.append(
                'name: ' + name + '\n' +
                'description: ' + desc.description + '\n' +
                '\n'.join([
                    'parameter: ' + _param_str(pname, pdesc)
                    for pname, pdesc in desc.params.items()
                ]) + '\n\n'
            )
        return functions_str_list


def build_find_pdf_by_keyword(client):

    description = dp.FunctionDesc(
        description=(
            'Searches the `pdf_collection` by keyword and returns any matches that were found in the PDF description'),
        params={
            'name': dp.ParamDesc(
                description='The collection name',
                typ=str,
                required=True
            ),
            'keyword': dp.ParamDesc(
                description='The keyword to search for',
                typ=str,
                required=True
            )
        }
    )

    def find_pdf_by_keyword(name, keyword):
        matches = []
        #for pid in client.get_collection(name).get()['ids']:
        for metadata in client.get_collection(name).get()['metadatas']:
            #if name.lower() in pid.lower():
            if keyword.lower() in metadata['description'].lower():
                #matches.append(pid)
                matches.append(metadata['filename'])
        return ', '.join(matches)

    return find_pdf_by_keyword, description


'''

# Add PDFs from the directory to the database
data_dir = 'data/gans'
pdfdb = PDFDB('gandb', api_key)
filenames = sorted(pdf_utils.find_pdfs_in_directory(data_dir))

func, _ = build_find_pdf_by_keyword(pdfdb.client)

out = func('pdf_collection', 'GAN')
print(out)
exit()

query = 'What is a GAN?'

filename = filenames[0]
res = pdfdb.page_collection.query(
    query_texts=[query],
    where={'filename': filename},
    n_results=4)

print(res)
exit()
#for filename in filenames:
#    description = DESCRIPTIONS[os.path.basename(filename)]
#    pdfdb.add_pdf(filename, description)

tools = {
    'find_pdf_by_keyword': build_find_pdf_by_keyword(pdfdb.client)
}

collection_info = {}
for x in pdfdb.client.list_collections():
    collection = pdfdb.client.get_collection(x.name)
    collection_info[x.name] = {
        'name': x.name,
        'metadata': str(collection.metadata)
    }


collection_info_str = json.dumps(collection_info, indent=4, sort_keys=False)

resources = {
    'ChromaDB Database Collections': collection_info_str
}

'''
agent = Agent(tools={}, resources={})
#agent.chat()

question = 'How do I make a ham sandwich?'
context = ''
constraint = 'Answer in a sarcastic and insulting manner'
print(agent.answer_question(question, context, constraint))

