import setuptools
from setuptools import setup

setup(
    name= 'LogisticRegressionExample',
    version = '1.0',
    description = 'Interview questions',
    url = "https://github.com/itsmelaksh/logisticReg",
    author="laxman Singh",
    author_email="itsmelaksh@gmail.com",
    long_description_content_type="src/readme.md", # not available till now
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Not Applicable",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires = ['sklearn','numpy'],
)