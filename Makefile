black:
	black .

flake:
	flake8 --max-line-length=120

tests:
	pytest -vv tests/ml_perf_end_to_end_tests.py
	pytest -vv tests/contrib_end_to_end_tests.py

tests-ml-perf:
	pytest -vv tests/ml_perf_end_to_end_tests.py

test-contrib:
	pytest -vv tests/contrib_end_to_end_tests.py

all: black flake tests

