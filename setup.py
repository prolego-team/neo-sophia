import setuptools
from setuptools import find_namespace_packages


setuptools.setup(
    name='neosophia',
    version='0.1',
    author='Prolego Inc.',
    author_email='russ@prolego.com',
    url='https://github.com/prolego-team/neo-sophia',
    packages=find_namespace_packages(where='src'),
    package_dir={'': 'src'}
)

