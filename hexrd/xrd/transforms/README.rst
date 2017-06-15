================================
 Transforms module (refactored)
================================

This small document tries to describe the organization of the
refactored transforms module.

Previous to refactor, there were several different versions of the
same functions (the transforms module) implemented using different
languages/tools. Typically we could find the same function as Python
(using NumPy), but as code was optimized different versions written in
C (CAPI version) and numba (numba version) started to appear.

As code evolved, the interface of the functions would evolve as well,
sometimes diverging between different implementations (usually not
updating the older, slower functions as they were no longer in use).
This was worsened as the functions didn't have unittests that could
help locate the divergences and force a failure when this happened.


Objectives of the refactor
==========================

The refactor has several objectives:

1. Homogeneize the interface for all the functions in the transform
   module, providing a visible interface.

2. Homogeneize the structure of the code. Right now there are rewrites
   of functions scattered in different files. This is specially true
   for numba versions, which were often placed "ad hoc" on the files
   that needed the speed up.

3. Improve testing of the functions of the transform module. The
   functions in the module are quite important and subject to tweaks
   to improve performance. Testing provides a safety net when
   performing those changes. Using the same testing code for all
   existing implementations is also a tool to prevent interface
   divergence, as an incompatible change to the interface of a
   function will need tests to be modified raising errors if the other
   implementations are not updated.

4. Make it easy to use a "default" implementation for a given function,
   but allow access to a specific implementation if required.


Physical structure of the refactored code
=========================================

All implementations of the new module will exists under a common
"umbrella" sub-module. This sub-module will be "transforms", and
will appear as "hexrd.xrd.transforms". Importing that module will
export the symbols for the whole interface of the module with the
"default" implementations. "Defaults" will be those functions that
strike the optimal balance between debugability and performance
for that specific function.

Under the transforms module, there will be 3 different submodules, one
for each of the possible implementations. The submodules being "xfpy",
"xfcapi" and "xfnumba" for python with NumPy, C implementations and
numba implementations respectively. For xfcapi the implementation will
contain the Python wrapping code that enforces proper layout of the
arguments before calling the C function, which will be in a C module.

Importing "hexrd.xrd.transforms.xfpy", "hexrd.xrd.transforms.xfcapi"
and "hexrd.xrd.transforms.xfnumba" should provide the functions at the
submodule level implemented using Python with NumPy, C and numba
respectively. Only the "xfpy" should guarantee providing the full
interface of the transforms module. The "xfpy" should be used as a
reference, working implementation focused on readability. "xfcapi" and
"xfnumba" will be focused on providing faster alternatives for code
found to be critical adhering to the same interface as "xfpy".

The code inside of each of the submodules is recommended to be
physically similar. If certain function implementations are in certain
files, it is recommended to have an equivalent file in all the
implementations containing that function, which may differ only on a
suffix.

Right now there are two different implementation files in the submodules:

1. rotations, containing code dealing mostly with construction of
   rotation matrices

2. transforms, dealing mostly with xrd important transform chains.

It is likely a third file will appear dealing with miscellaneous
functionality.


Testing
=======

All tests related to the transforms module should be under
"hexrd.xrd.transform.tests". In the directory containing the tests there
is a README files explaining how to structure the rotations.



