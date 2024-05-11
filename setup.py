from typing import List
from setuptools import find_packages, setup

__version__= "0.0.0"

def get_requirements(file_path:str)->List[str]:
    '''this function return list'''
    requirements=[]
    with open(file_path) as file_obj:
        requirements= file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")
        
    return requirements

setup(
    name = 'mlproject',
    version=__version__,
    author="atishayrokadia",
    author_email="atishayrokadia2402@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements('requirement.txt')
)