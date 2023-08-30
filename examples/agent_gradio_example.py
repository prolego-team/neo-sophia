""" Example script to interact with the Agent class """
import os

import yaml
import click
import gradio as gr
import pandas as pd

import neosophia.agents.utils as autils

from examples import project

from neosophia.db import sqlite_utils as sql_utils
from neosophia.llmtools import openaiapi as oaiapi
from neosophia.agents.agent import Agent
from neosophia.agents.system_prompts import UNLQ_GPT_BASE_PROMPT

opj = os.path.join

RESOURCES_FILENAME = 'resources.yaml'


@click.command()
@click.option('--toggle', '-t', is_flag=True, help='Toggle variables/resources')
def main(toggle):
    """ main """
    print('\n')

    oaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create a workspace for the agent to save things
    workspace_dir = autils.create_workspace_dir(config)

    # Load and/or generate tools
    tools_filepath = opj(workspace_dir, config['Agent']['tools_filename'])
    tool_descriptions = autils.setup_and_load_yaml(tools_filepath, 'tools')
    tools = autils.setup_tools(config['Tools'], tool_descriptions)
    autils.save_tools_to_yaml(tools, tools_filepath)

    # Load and/or generate resources
    resources_filepath = opj(
        workspace_dir, config['Agent']['resources_filename'])

    # Resources loaded from the yaml file saved in the Agent's workspace
    workspace_resources = autils.setup_and_load_yaml(
        resources_filepath, 'resources')

    resources = autils.setup_sqlite_resources(
        config['Resources']['SQLite'], workspace_dir,
        resources_filepath, workspace_resources)

    # Dictionary to store all variables the Agent has access to
    variables = {}

    agent_base_prompt = UNLQ_GPT_BASE_PROMPT

    def variables_to_dataframe(variables):
        df_variables = {}
        for variable in variables.values():
            df_variables.setdefault('Name', []).append(variable.name)
            df_variables.setdefault('Value', []).append(str(variable.value))
            df_variables.setdefault(
                'Description', []).append(variable.description)
        return pd.DataFrame(df_variables)

    def tools_to_dataframe(tools):
        df_tools = {}
        for tool in tools.values():
            df_tools.setdefault('Name', []).append(tool.name)
            df_tools.setdefault(
                'Description', []).append(tool.description)
        return pd.DataFrame(df_tools)

    def resources_to_dataframe(resources):
        df_resources = {}
        for resource in resources.values():
            df_resources.setdefault('Name', []).append(resource.name)
            df_resources.setdefault(
                'Description', []).append(resource.description)
        return pd.DataFrame(df_resources)

    def new_question_wrapper():
        return ''

    def agent_wrapper(user_input, chat_history):

        # Connect to any SQLite databases
        for db_info in config['Resources']['SQLite']:
            conn = sql_utils.get_conn(db_info['path'])
            name = db_info['name']
            var_name = name + '_conn'

            variable = autils.Variable(
                name=var_name,
                value=conn,
                description=f'Connection to {name} database')

            variables[var_name] = variable

            # Add table schemas as variables
            tables = sql_utils.get_tables_from_db(conn)
            for table in tables:

                # Don't include system tables
                if table in ['sqlite_master', 'sqlite_sequence']:
                    continue

                table_schema = sql_utils.get_table_schema(conn, table)

                example_data = 'Example data:\n'
                example_data += str(sql_utils.execute_query(
                    conn, f'SELECT * FROM {table} LIMIT 3'))

                description = f'Schema for table {table} in database {name}\n'
                description += example_data

                variable = autils.Variable(
                    name=table + '_table_schema',
                    value=table_schema,
                    description=description)

                variables[table + '_table_schema'] = variable

        agent = Agent(
            'MyAgent',
            workspace_dir,
            agent_base_prompt,
            tools,
            resources,
            variables,
            toggle=toggle)

        for _ in agent.interact(user_input):

            #chat_history.append(agent.log['prompt'][-1])
            #chat_history.append(agent.log['response'][-1])
            yield (
                variables_to_dataframe(agent.variables),
                tools_to_dataframe(agent.tools),
                resources_to_dataframe(agent.resources),
                chat_history
            )

    with gr.Blocks() as demo:
        gr.Markdown('# Agent Example')

        with gr.Row():
            with gr.Column():
                gr.Markdown('## Data Resources')
                resources_df = gr.Dataframe(value=None, wrap=True)
        with gr.Row():
            with gr.Column():
                gr.Markdown('## Variables')
                variables_df = gr.Dataframe(value=None, wrap=True)
        with gr.Row():
            with gr.Column():
                gr.Markdown('## Tools')
                tools_df = gr.Dataframe(value=None, wrap=True)

        final_answer = gr.Textbox(value='', label='Answer', interactive=False)

        user_input = gr.Textbox(
            value='Which customer has the highest mortgage interest rate?',
            label='Ask a question about the data')

        chatbot = gr.Chatbot(label='Chatbot message log')

        user_input.submit(
            new_question_wrapper, outputs=final_answer).then(
                agent_wrapper,
                [user_input, chatbot],
                [variables_df, tools_df, resources_df, chatbot]
            )

    demo.queue()
    demo.launch()


if __name__ == '__main__':
    main()
