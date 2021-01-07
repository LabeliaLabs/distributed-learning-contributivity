from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("mplc/requirements.txt", "r") as fh:
    requirements = fh.read()

setup(
    name="mplc",
    version='0.3.1',
    author="SubstraFoundation",
    author_email="contact@substra.org",
    description="A distributed-learning package for the study of contributivity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SubstraFoundation/distributed-learning-contributivity",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    keywords=['library',
              'substra',
              'machine learning',
              'contributivity',
              'multi-partner learning',
              'distributed learning'],
    license='Apache 2.0',
    install_requires=requirements,
    python_requires='>=3.6',
)
