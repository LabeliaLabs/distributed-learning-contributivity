flake:
	flake8 --max-line-length=120

tests-unit:
	tox -e unit --parallel 2 --verbose

tests-ml_perf:
	tox -e ml_perf --verbose

tests-contrib:
	tox -e contrib --parallel 2 --verbose

all: flake tests-unit tests-ml_perf tests-contrib
