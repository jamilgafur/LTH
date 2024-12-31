from setuptools import setup, find_packages

setup(
    name='pyPrune',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
    ],
    author="Jamil Gafur",
    author_email="jamil-gafur@uiowa.edu",
    description="A package for iterative magnitude pruning with GPU support",
)
