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

