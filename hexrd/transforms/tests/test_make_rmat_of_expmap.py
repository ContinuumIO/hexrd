# tests for make_rmat_of_expmap

from __future__ import absolute_import

from .. import make_rmat_of_expmap as default_make_rmat_of_expmap
from ..xf_numpy import make_rmat_of_expmap as numpy_make_rmat_of_expmap
from ..xf_capi import make_rmat_of_expmap as capi_make_rmat_of_expmap
from ..xf_numba import make_rmat_of_expmap as numba_make_rmat_of_expmap

from ... import constants as cnst

import numpy as np
from numpy.testing import assert_allclose

import pytest

ATOL_IDENTITY = 1e-10

all_impls = pytest.mark.parametrize('make_rmat_of_expmap_impl, module_name', 
                                    [(numpy_make_rmat_of_expmap, 'numpy'),
                                     (capi_make_rmat_of_expmap, 'capi'),
                                     (numba_make_rmat_of_expmap, 'numba'),
                                     (default_make_rmat_of_expmap, 'default')]
                                )



# ------------------------------------------------------------------------------

# Test trivial case

@all_impls
def test_zero_expmap(make_rmat_of_expmap_impl, module_name):
    exp_map = np.zeros((3,))
    
    rmat = make_rmat_of_expmap_impl(exp_map)

    assert_allclose(rmat, cnst.identity_3x3, atol=ATOL_IDENTITY)


@all_impls
def test_2pi_expmap(make_rmat_of_expmap_impl, module_name):
    """all this should result in identity - barring numerical error.
    Note this goes via a different codepath as phi in the code is not 0."""

    rmat = make_rmat_of_expmap_impl(np.array([2*np.pi, 0., 0.]))
    assert_allclose(rmat, cnst.identity_3x3, atol=ATOL_IDENTITY)

    rmat = make_rmat_of_expmap_impl(np.array([0., 2*np.pi, 0.]))
    assert_allclose(rmat, cnst.identity_3x3, atol=ATOL_IDENTITY)

    rmat = make_rmat_of_expmap_impl(np.array([0., 0.,2*np.pi]))
    assert_allclose(rmat, cnst.identity_3x3, atol=ATOL_IDENTITY)


# ------------------------------------------------------------------------------

# check that for some random inputs the resulting matrix is orthogonal

@all_impls
def test_orthonormal(make_rmat_of_expmap_impl, module_name):
    rmat = make_rmat_of_expmap_impl(np.array([42.0, 3., 32.5]))
    # dot(A, A.T) == IDENTITY is a good orthonormality check
    assert_allclose(np.dot(rmat, rmat.T), cnst.identity_3x3,
                     atol=ATOL_IDENTITY)

    rmat = make_rmat_of_expmap_impl(np.array([-32.0, 0.0, 17.6]))
    assert_allclose(np.dot(rmat, rmat.T), cnst.identity_3x3,
                     atol=ATOL_IDENTITY)



# ------------------------------------------------------------------------------

# Test strided input
@all_impls
def test_strided(make_rmat_of_expmap_impl, module_name):
    exp_map = np.array([42.0, 3., 32.5]) # A random expmap

    buff = np.zeros((3, 3), order='C')
    buff[:,0] = exp_map[:] # assign the expmap to a column, so it is strided

    result_contiguous = make_rmat_of_expmap_impl(exp_map)
    result_strided = make_rmat_of_expmap_impl(buff[:,0])

    # in fact, a stricter equality check should work as well,
    # but anyways...
    assert_allclose(result_contiguous, result_strided)
