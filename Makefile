flake:
	flake8 --max-line-length=120

tests-unit:
	tox -e unit --parallel 2

tests-ml_perf:
	tox -e ml_perf

tests-contrib:
	tox -e contrib --parallel 2

all: flake tests
