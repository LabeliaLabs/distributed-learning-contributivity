# Contributing to mplc

> *Please, feel free to suggest any other relevant item, to ask for edits or even to submit a Pull Request!*
> *You can also have a look at these broader [contributing guidelines](https://github.com/SubstraFoundation/.github/blob/master/CONTRIBUTING.md)*.

Table of content:

1. [Git workflow & branching](#1-git-workflow--branching)
1. [Python](#2-python)
   1. [Python Virtual Environment](#2i-python-virtual-environment)
   1. [Python Enhancement Proposals (PEP)](#2ii-python-enhancement-proposals-pep)
      - [Black formatter](#black-formatter)
      - [Flake8 linter](#flake8-linter)
   1. [Basic module structure & Imports order](#2iii-basic-module-structure--imports-order)
   1. [Jupyter Notebooks](#2iv-jupyter-notebooks)
   1. [Sharing & online rendering](#2v-sharing--online-rendering)
1. [Further Resources](#3-further-resources)

## 1. Git workflow & branching

The branching model of the project is a very simplified of the [standard approach](https://nvie.com/posts/a-successful-git-branching-model/):

- The `master` branch is protected
- Contributors create a feature branch from `master`, push it to the repository and open a draft PR
- Once automated tests passe, their PRs are reviewed and merged by the repository maintainers

> **Note**: If you are not yet part of the contributor list, please get in touch with us. In the meantime, you can create your own fork of the project and open a PR pointing to this source repository, so we can evaluate your submission. [Github documentation](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork).

## 2. Python

### 2.i. Python Virtual Environment

In order to keep your installation separated from your general Python environment, which is a general Python good practice, it is recommended to set up a Python [virtual environment](https://virtualenv.pypa.io/en/latest/). In a new terminal window, please use one of the following method:

```sh
# Method 1: install the virtualenv package
pip3 install --user virtualenv

# Create a new virtual environment
virtualenv -p python3 NAME_OF_YOUR_VENV
# or even
virtualenv -p $(which python3) NAME_OF_YOUR_VENV

# Method 2: install the python3-venv package
sudo apt install python3-venv # (Ubuntu)

# Create a new virtual environment
python3 -m venv NAME_OF_YOUR_VENV

# Method 1 & 2: activate your new virtual env
source NAME_OF_YOUR_VENV/bin/activate

# Method 1 & 2: stop your virtual environment
deactivate
```

Some of you might prefer to use [Anaconda](https://anaconda.org/), if so, you will be able to manage your virtual environment like this:

```sh
# Create a new virtual environment
conda create --name NAME_OF_YOUR_VENV

# Activate your new virtual environment
conda activate NAME_OF_YOUR_VENV

# Stop your virtual environment
deactivate
```

Once inside your new virtual environment, you can install the project dependencies with the commands:

- pip setup: `pip3 install -r dev-requirements.txt`
- anaconda setup: `conda install --file dev-requirements.txt`

What is installed inside a virtual environment is separated from your general Python setup.

If you want to go further, please refer to the [official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

If you are looking for more Python Virtual Environment resources, you might be interested in this post from [Real Python](https://realpython.com/python-virtual-environments-a-primer/).

Please note that you can then select your new virtual environment as any other Python interpreter in your favorite IDE and be able to use the new installed packages as if it were installed on your general Python setup.

### 2.ii. Python Enhancement Proposals (PEP)

- [PEP8](https://pep8.org/)
- [PEP Index](https://www.python.org/dev/peps/) & [Repository](https://github.com/python/peps)
- Real Python: [How to Write Beautiful Python Code With PEP 8](https://realpython.com/python-pep8/)

#### Black formatter

This tool might seem a little bit radical, but it is based on PEPs and offers several possibilities of customization. It will help you learn and improve your code.

You don't have to take all the suggested modifications (with the help of `git diff`) but it is a good reference based on PEP rules that will ensure **validity**, **maintainability** and **readability** of the code.

Link: [Black package pepository](https://github.com/psf/black).

##### Installation

```sh
pip3 install black

# with anaconda
conda install -c conda-forge black
```

##### Usage

```sh
# Inspect a file
black FILE.py

# Inspect files in the current folder
black .
```

#### Flake8 linter

[Flake8](https://pypi.org/project/flake8/) is a famous Python linting package that might be of help within your development environment. It will help ensure the correct format of your code.

> Note: `flake8` is part of the automated tests of the repository executed by Travis. Incorrect code formatting will automatically lead to failed tests. 

### 2.iii. Basic module structure & Imports order

```python
# -*- coding : utf-8

"""
Documentation of module
"""

# 1. imports from standard library (ex. sys)

# 2. imports from third party library (ex. arrow)

# 3. imports from project modules (project internal modules)

# 4. global variables

# 5. definition of exception classes

# 6. definition of other classes

# 7. definition of contributivity_methods

# 8. Python main function
def main():
    """
    Documentation of the method
    """
    (...)

if __name__ == '__main__':
    main()
```

### 2.iv. Jupyter Notebooks

> "The future is now, old man!"

Jupyter Notebooks are awesome! It allows you run Python code (but not only!) in your favorite web browser and handles for you all the backend management so you can focus on writing your code in cells, or your notes directly in markdown! There are plenty of fresh contents about ways to adopt, adapt or trick notebooks. If not yet familiar with it, you really should have a look: <https://jupyter.org/>.

Plus, Jupyter Notebooks come with a serious set of crazy cool and neatly documented widgets, [help yourself](
https://ipywidgets.readthedocs.io/en/latest/).

Note: [Jupyter lab](https://github.com/jupyterlab/jupyterlab) is like the future of Notebook, be sure to have a look, you might like it!

#### 2.v. Sharing & online rendering

Notebooks now have really efficient ways to share your code and display your outputs, among them, you will find:

- [Binder](https://mybinder.org/) that lets you turn a Git repository into a collection of interactive notebooks!
- [Voila](https://github.com/voila-dashboards/voila) is a newcomer that will allow to build dashboards for your presentations!
- [Colab](https://colab.research.google.com/) offers online, cloud-hosted notebooks and its free plan enables usage of GPUs

#### 2.vi. Release a new build

To release a new version on PyPI, go at the root of the repository, and trigger the build with `pip`.
You will need all the `dev-requirements` installed.

```bash
$ pip install -r dev-requirements.txt
``` 

Make sure that you have the right access to [PyPI](https://pypi.org/project/mplc/), and enable the 2FA for safety purpose. 

```bash
$ python3 setup.py sdist bdist_wheel
$ python3 twine upload dist/*
```

## 3. Further Resources

- [Substra Contributing](https://github.com/SubstraFoundation/.github/blob/master/CONTRIBUTING.md) & [Coding Style](https://github.com/SubstraFoundation/.github/blob/master/CONTRIBUTING.md#coding-guidelines)
- [RealPython](https://realpython.com)
- A nice [Python Cheat Sheet](https://gto76.github.io/python-cheatsheet/)
- [tips] Debug with colors by replacing `pdb` by `ipdb`: `pip3 install ipdb`. You can then set a break point by including `import ipdb; ipdb.set_trace()`
