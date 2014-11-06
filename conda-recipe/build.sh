$PYTHON setup.py install

git describe | sed s/-/./g  > __conda_version__.txt
