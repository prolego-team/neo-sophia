"""
Test project configuration.
"""

from neosophia.llmtools import openaiapi


def test_config():
    """
    Test importing project and different configuration settings.
    """
    from examples import project

    # attempt to load API key
    api_key = openaiapi.load_api_key(project.OPENAI_API_KEY_FILE_PATH)
    assert api_key
