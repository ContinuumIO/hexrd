==============================
 Transforms submodule testing
==============================

The transforms submodule implements a important (and performance
critical in many cases) set of functions for XRD.

Due to performance considerations there have been different
implementations of some of the functions either as Python+Numpy code,
a C extencions module or numba.

When structuring the transforms modules we want to keep these
different implementation, adding tests to make sure that:

a) all implementations share the same exact interface, so that they
   can be used interchangeably.

b) all implementations have at least some minimal testing.

c) a default for each function is provided that should be used
   unless there is a compelling reason not to do so.

d) a reference implementation exists in Python + NumPy


Testing
=======

In this directory there will be unittests to test the different
functions.

To run the test just use the following command line

.. code:: bash
   python -m unittest discover hexrd.xrd.transforms

The tests should also be discovered when run for all hexrd.

The idea is to implement a TestCase per function which performs
all the unittests for that function. The function itself will be
specified as a class variable in the TestCase, using the default
implementation as the value for that class variable.

The TestCase implemented for the default will then be subclassed
once per implementation, the class variable of the test having as
value the specific alternative implementation for the test.

This will look like like:

.. code:: python
   import unittest

   import hexrd.xrd.transforms as xfdefault
   import hexrd.xrd.transforms.xfpy as xfpy
   import hexrd.xrd.transforms.xfnumba as xfnumba
   import hexrd.xrd.transforms.xfcapi as xfcapi


   class Test_<function_name>(unittest.TestCase):
       fn = staticmethod(xfdefault.<function_name>)

       [ALL TESTS GO HERE]

   class Test_<function_name>_py(Test_<function_name>):
       fn = staticmethod(xfpy.<function_name>)

   class Test_<function_name>_numba(Test_<function_name>):
       fn = staticmethod(xfnumba.<function_name>)

   class Test_<function_name>_capi(Test_<function_name>):
       fn = staticmethod(xfcapi.<function_name>)

