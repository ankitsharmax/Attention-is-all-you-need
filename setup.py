from os import name
from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str) -> List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # read the entire file
        requirements = [req.replace("\n","") for req in requirements]
        # replace the new-line character with blanks

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

    setup(
        name='Attention-is-all-you-need',
        version='0.0.1',
        author='ankit',
        author_email='kumarankitx022@gmail.com',
        packages=find_packages(),
        install_requires=get_requirements('requirements.txt')
    )