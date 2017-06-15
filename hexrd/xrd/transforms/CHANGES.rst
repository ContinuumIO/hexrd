==========================
 TRANSFORM MODULE CHANGES
==========================

This contains a list of changes in the transforms module in order to
homogeneize the different implementations. Some of the functions have
diverged on their different implementations. In this document, I will
point the differences and which implementation is to be kept, as well
as a way to correct code that was based on a discarded behavior.

The changes are mentioned in a per-function basis. Per function as in
"piece of functionality". It will usually refer to a single function
in the code, but may affect more than one in some cases. For example,
when "array" versions of a function exist.

Overall, the criteria is keeping the function interface that is used
in the most critical way. That usually means preserving the interface
of the CAPI functions. There are some functions with "array"
versions. In that case merging the array and scalar versions under the
same interface may make sense, it a "vectorized" interface makes
sense.

===========
 Functions
===========

makeOscillRotMat
================

There are actually two functions in the CAPI versions:

- makeOscillRotMat(oscillAngles). Takes a 2-element arrays [chi, ome]
  and returns the rotation matrix for the oscillation.

- makeOscillRotMatArray(chi, omeArray). Takes a chi scalar and an
  array of n elements of ome values and returns n rotation matrices for
  all the oscillations. Chi remains constant for all arrays, only ome
  varies.

In the Python (and numba) versions only the scalar version exists.

Action: implement the interface using the interface of makeOscillRotMatArray,
but supporting an scalar as ome.

