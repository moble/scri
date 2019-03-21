#! /bin/bash

set -e

/bin/rm -rf build __pycache__ dist scri.egg-info
python setup.py install
python -c 'import scri; print(scri.__version__)'

pip install --quiet --upgrade twine
/bin/rm -rf build __pycache__ dist scri.egg-info
python setup.py sdist
twine upload dist/*
