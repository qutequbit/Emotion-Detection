from setuptools import find_packages, setup
from typing import List

KEY = '-e .'
def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n',"") for req in requirements]
        if KEY in requirements:
            requirements.remove(KEY)
    return requirements

setup(
    name='Emotion-Detection',
    version='0.0.1',
    author='Simran',
    author_email='myselfsimranray@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)