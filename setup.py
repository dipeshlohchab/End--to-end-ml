from setuptools import find_packages, setup
from typing import List

def get_packages(filepath: str) -> List[str]:
    requirements = []
    with open(filepath, "r") as f:
        requirements = f.readlines()
        requirements = [i.strip() for i in requirements]  # Use .strip() to remove newlines/spaces
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name="ML_Project",
    version="0.0.1",
    author="Dipesh Lohchab",
    author_email="dipeshlohchab0302@gmail.com",
    description="Machine Learning Project",
    packages=find_packages(),
    install_requires=get_packages("requirements.txt"),
)
