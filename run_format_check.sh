#!/bin/bash -e

set -e

export PYTHONPATH=$PWD:$PYTHONPATH
pip install -q -r requires-style.txt
CHECK_DIR="official/vision official/quantization official/nlp official/multimodal"
pylint $CHECK_DIR --rcfile=.pylintrc || pylint_ret=$?
if [ "$pylint_ret" ]; then
    exit $pylint_ret
fi
echo "All lint check passed!"
flake8 official || flake8_ret=$?
if [ "$flake8_ret" ]; then
    exit $flake8_ret
fi
echo "All flake check passed!"
isort --check-only -rc official || isort_ret=$?
if [ "$isort_ret" ]; then
    exit $isort_ret
fi
echo "All isort check passed!"
