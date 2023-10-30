""" Gradio application to interact with the Agent """
import os

import yaml
import click
import gradio as gr
import pandas as pd

import neosophia.agents.utils as autils
import neosophia.agents.system_prompts as sp

from examples import project

from neosophia.db import sqlite_utils as sql_utils
from neosophia.llmtools import openaiapi as oaiapi
from neosophia.agents.agent import Agent
from neosophia.agents.data_classes import Variable
from neosophia.agents.system_prompts import PARAM_AGENT_BP, TOOL_AGENT_BP

opj = os.path.join


def generate_tool_summaries(workspace_dir, tool_descriptions):
    tool_summaries = []
    summary_filepath = opj(workspace_dir, 'tool_summaries.yaml')
    if os.path.exists(summary_filepath):
        with open(summary_filepath) as f:
            tool_summaries = yaml.safe_load(f) or []

    # Extracting names from tool_summaries for easier lookup
    names = [x['name'] for x in tool_summaries]

    # Generate summaries of the tools for viewing
    for name, info in tool_descriptions.items():
        if name not in names:
            print(f'Generating summary for {name}...')
            description = f'name: {name}\n'
            description += str(info['description']) + '\n'
            description += 'params:\n' + str(info['params']) + '\n'
            summary = autils.summarize_text(
                description, num_sentences='1-2', num_words='40')
            tool_summaries.append({'name': name, 'description': summary})

    with open(summary_filepath, 'w') as file:
        yaml.dump(tool_summaries, file, default_flow_style=False, sort_keys=False)

    return tool_summaries


@click.command()
@click.option('--toggle', '-t', is_flag=True, help='Toggle variables')
def main(toggle):
    """ main """
    print('\n')

    oaiapi.set_api_key(oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH))

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create a workspace for the agent to save things
    workspace_dir = autils.create_workspace_dir(config)

    # Load and/or generate tools
    tools_filepath = opj(workspace_dir, config['Agent']['tools_filename'])
    tool_descriptions = autils.setup_and_load_yaml(tools_filepath, 'tools')
    tool_summaries = generate_tool_summaries(workspace_dir, tool_descriptions)
    tools = autils.setup_tools(config['Tools'], tool_descriptions)
    autils.save_tools_to_yaml(tools, tools_filepath)

    # Dictionary to store all variables the Agent has access to
    variables = {}

    # Put all data from databases into Pandas DataFrame Variables
    for db_info in config['Resources']['SQLite']:
        conn = sql_utils.get_conn(db_info['path'])
        name = db_info['name']
        tables = sql_utils.get_tables_from_db(conn)

        for table in tables:

            # Don't include system tables
            if table in ['sqlite_master', 'sqlite_sequence']:
                continue

            data = sql_utils.execute_query_pd(conn, f'SELECT * FROM {table};')

            description = f'All data from the {table} {table} in database {name}\n'
            variable = Variable(
                name=f'{table}_data',
                value=data,
                description=description)

            variables[f'{table}_data'] = variable

    def variables_to_dataframe(variables, only_visible):
        df_variables = {}
        for variable in variables.values():
            if only_visible and not variable.visible:
                continue
            df_variables.setdefault('Name', []).append(variable.name)
            df_variables.setdefault('Value', []).append(str(variable.value))
            df_variables.setdefault(
                'Description', []).append(variable.description)
        return pd.DataFrame(df_variables)

    def new_question_wrapper():
        return ''

    def agent_wrapper(command, chat_history):

        agent = Agent(
            workspace_dir,
            TOOL_AGENT_BP,
            PARAM_AGENT_BP,
            tools,
            variables,
            toggle=toggle)

        all_variables_df = pd.DataFrame()
        visible_variables_df = pd.DataFrame()
        tools_df = pd.DataFrame(tool_summaries)
        answer = None

        chat_history.append([command, None])

        # Yield before any computation so tools show up
        yield (
            all_variables_df,
            visible_variables_df,
            tools_df,
            answer,
            chat_history
        )

        for _ in agent.interact(command):
            all_variables_df = variables_to_dataframe(agent.variables, False)
            visible_variables_df = variables_to_dataframe(agent.variables, True)
            output = agent.gradio_output.replace(sp.NO_PYTHON, '')
            chat_history.append([None, output])
            yield (
                all_variables_df,
                visible_variables_df,
                tools_df,
                agent.answer,
                chat_history
            )

    with gr.Blocks() as demo:
        gr.Markdown('# Agent Example')

        with gr.Row():
            with gr.Column():
                gr.Markdown('## Agent Tools')
                tools_df = gr.Dataframe(
                    value=pd.DataFrame(tool_summaries), wrap=True)

        final_answer = gr.Textbox(value='', label='Answer', interactive=False)


        chatbot = gr.Chatbot(label='Chatbot message log')
        user_input = gr.Textbox(
            value=(
                'Provide information about the customer who made the '
                'largest withdrawal from an EasyAccess Checking Account. '
                'Make sure the account you are checking is an EasyAccess '
                'Checking account.'),
            label='Ask a question about the data')

        with gr.Row():
            with gr.Column():
                gr.Markdown('## Agent Scratchpad')
                visible_variables_df = gr.Dataframe(value=None, wrap=True)

        with gr.Row():
            with gr.Column():
                gr.Markdown('## All Variables')
                all_variables_df = gr.Dataframe(value=None, wrap=True)

        user_input.submit(
            new_question_wrapper, outputs=final_answer).then(
                agent_wrapper,
                [user_input, chatbot],
                [
                    all_variables_df,
                    visible_variables_df,
                    tools_df,
                    final_answer,
                    chatbot
                ]
            )

        demo.queue()
        demo.launch()


if __name__ == '__main__':
    main()
