# tests for unit_vector

from __future__ import absolute_import

from .. import unit_vector as default_unit_vector
from ..xf_numpy import unit_vector as numpy_unit_vector
from ..xf_capi import unit_vector as capi_unit_vector
from ..xf_numba import unit_vector as numba_unit_vector

import numpy as np
from numpy.testing import assert_allclose

import pytest

all_impls = pytest.mark.parametrize('unit_vector_impl, module_name', 
                                    [(numpy_unit_vector, 'numpy'),
                                     (capi_unit_vector, 'capi'),
                                     (numba_unit_vector, 'numba'),
                                     (default_unit_vector, 'default')]
                                )



# ------------------------------------------------------------------------------

# trivial tests

@all_impls
def test_trivial(unit_vector_impl, module_name):
    # all vectors in eye(3) are already unit vectors
    iden = np.eye(3)

    # check a vector at a time
    assert_allclose(unit_vector_impl(iden[0]), iden[0])
    assert_allclose(unit_vector_impl(iden[1]), iden[1])
    assert_allclose(unit_vector_impl(iden[2]), iden[2])

    # use the array version
    assert_allclose(unit_vector_impl(iden), iden)


@all_impls
def test_zero(unit_vector_impl, module_name):
    # When a zero vector is given, a potential "division by zero" happens.
    # in this library, instead of trying to normalize a zero-norm vector
    # (which would trigger the division by zero), the original vector is
    # returned.

    # check vector
    zero_vec = np.zeros((3,))
    assert_allclose(unit_vector_impl(zero_vec), zero_vec)

    # check array
    zero_arr = np.zeros((3,3))
    assert_allclose(unit_vector_impl(zero_arr), zero_arr)

    # check mixed array
    mix_arr = np.eye(3)
    mix_arr[1,:] = 0.0
    assert_allclose(unit_vector_impl(mix_arr), mix_arr)
