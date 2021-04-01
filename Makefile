#!/usr/bin/env bash
flake:
	flake8 --max-line-length=120

tests-unit:
	tox -e unit --parallel 8
tests-ml_perf:
	tox -e ml_perf

tests-contrib:
	tox -e contrib --parallel 8

all: flake tests