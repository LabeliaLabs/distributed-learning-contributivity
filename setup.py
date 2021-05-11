from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    'h5py==2.10.0',
    'joblib==1.0.0',
    'librosa==0.8.0',
    'matplotlib==3.3.3',
    'numpy==1.19.4',
    'scipy==1.5.4',
    'scikit-learn==0.23.2',
    'pandas==1.1.5',
    'seaborn==0.11.0',
    'loguru==0.5.3',
    'tensorflow==2.4.1',
    'ruamel.yaml==0.16.12'
]

test_requirements = [
    'tox==3.23.0',
    'pytest==6.2.2',
    'pytest-cov==2.10.1',
]

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
    tests_require=test_requirements,
    python_requires='>=3.6',
)
