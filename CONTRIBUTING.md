# Contributing to Contributivity

> Please, feel free to suggest any other relevant item, to ask for edits or even to submit a Pull Request!
>
> You can also have a look at these broader [contributing guidelines](https://github.com/SubstraFoundation/.github/blob/master/CONTRIBUTING.md).

- [Contributing to Contributivity](#contributing-to-contributivity)
  - [1. Git workflow & branching](#1-git-workflow--branching)
  - [2. Python](#2-python)
    - [2.1 Python Virtual Environment](#21-python-virtual-environment)
    - [2.2 Python Enhancement Proposals (PEP)](#22-python-enhancement-proposals-pep)
      - [Black formatter](#black-formatter)
        - [Installation](#installation)
        - [Usage](#usage)
      - [Flake8 linter](#flake8-linter)
    - [2.3 Basic module structure & Imports order](#23-basic-module-structure--imports-order)
    - [2.4 Jupyter Notebooks](#24-jupyter-notebooks)
      - [2.5 Sharing & online rendering](#25-sharing--online-rendering)
  - [3. Further Resources](#3-further-resources)

## 1. Git workflow & branching

As long as other people contribute to a repository, it is easier and safer to restrain direct actions on the `master` branch. To do so, it is advised to create new branches, based on `master` (or a dedicated sub-branch, `dev` for example), to develop any new feature and then to open a Pull Request. Once audited and validated, it can be merged into `master`, and so on.

To go further, here is a good example of a successful project build with a versioning file system (git, mercurial, etc.): <https://nvie.com/posts/a-successful-git-branching-model/>

## 2. Python

### 2.1 Python Virtual Environment

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

### 2.2 Python Enhancement Proposals (PEP)

- [PEP8](https://pep8.org/)
- [PEP Index](https://www.python.org/dev/peps/) & [Repository](https://github.com/python/peps)
- Real Python: [How to Write Beautiful Python Code With PEP 8](https://realpython.com/python-pep8/)

#### Black formatter

This tool might seem a little bit radical, but it is based on PEPs and offers several possibilities of customization. It will help you learn and improve your code

You don't have to take all the suggested modifications (with the help of `git diff`) but it is a good reference based on PEP rules that will ensure **validity**, **maintainability** and **readability** of the code: [Package Repository](https://github.com/psf/black)

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

Flake8 is a famous Python linting package that might be of help within your development environment: <https://pypi.org/project/flake8/>. It will help ensure the correct format of your code.

### 2.3 Basic module structure & Imports order

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

# 7. definition of methods

# 8. Python main function
def main():
    """
    Documentation of the method
    """
    (...)

if __name__ == '__main__':
    main()
```

### 2.4 Jupyter Notebooks

> The future is now, old man!

Jupyter Notebooks are awesome! It allows you run Python code (but not only!) in your favorite web browser and handles for you all the backend management so you can focus on writing your code in cells, or your notes directly in markdown! There are plenty of fresh contents about ways to adopt, adapt or trick notebooks. If not yet familiar with it, you really should have a look: <https://jupyter.org/>.

Plus, Jupyter Notebooks come with a serious set of crazy cool and neatly documented widgets, [help yourself](
https://ipywidgets.readthedocs.io/en/latest/).

Note: [Jupyter lab](https://github.com/jupyterlab/jupyterlab) is like the future of Notebook, be sure to have a look, you might like it!

#### 2.5 Sharing & online rendering

Notebooks now have really efficient ways to share your code and display your outputs, among them, you will find:

- [Binder](https://mybinder.org/) that lets you turn a Git repository into a collection of interactive notebooks!
- [Voila](https://github.com/voila-dashboards/voila) is a newcomer that will allow to build dashboads for your presentations!

#### 2.6 Release a new build 

To release a new version on Pypi, go at the root of the repository, and trigger the build with `pip`.
You will need all the `dev-requirements` installed.

```bash
$ pip install -r dev-requirements.txt
``` 

Make sure that you have the right access to pypi, and enable the 2FA for safety purpose. 

```bash
$ python3 setup.py sdist bdist_wheel
$ python3 twine upload dist/*
```

## 3. Further Resources

- [Substra Contributing](https://github.com/SubstraFoundation/.github/blob/master/CONTRIBUTING.md) & [Coding Style](https://github.com/SubstraFoundation/.github/blob/master/CONTRIBUTING.md#coding-guidelines)
- [RealPython](https://realpython.com)
- A nice [Python Cheat Sheet](https://gto76.github.io/python-cheatsheet/)
- [tips] Debug with colors by replacing `pdb` by `ipdb`: `pip3 install ipdb`. You can then set a break point by including `import ipdb; ipdb.set_trace()`
