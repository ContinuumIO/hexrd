# tests for make_beam_rmat

from __future__ import absolute_import

from .. import make_beam_rmat as default_make_beam_rmat
from ..xf_numpy import make_beam_rmat as numpy_make_beam_rmat
from ..xf_capi import make_beam_rmat as capi_make_beam_rmat
from ..xf_numba import make_beam_rmat as numba_make_beam_rmat

from ... import constants as cnst

import numpy as np
from numpy.testing import assert_allclose

import pytest

all_impls = pytest.mark.parametrize('make_beam_rmat_impl, module_name',
                                    [(numpy_make_beam_rmat, 'numpy'),
                                     (capi_make_beam_rmat, 'capi'),
                                     (numba_make_beam_rmat, 'numba'),
                                     (default_make_beam_rmat, 'default')]
                                )


@all_impls
def test_reference_beam_mat(make_beam_rmat_impl, module_name):
    """Building from the standard beam_vec and eta_vec should
    yield an identity matrix.

    This is somehow assumed in other parts of the code where using the default
    cnst.beam_vec and cnst.eta_vec implies an identity beam rotation matrix that
    is ellided in operations"""

    rmat = make_beam_rmat_impl(cnst.beam_vec, cnst.eta_vec)

    assert_allclose(rmat, cnst.identity_3x3)


@all_impls
def test_zero_beam_vec(make_beam_rmat_impl, module_name):
    beam_vec = np.array([0. ,0., 0.]) # this is bad...
    eta_vec = np.array([1., 0., 0.])

    with pytest.raises(RuntimeError):
        make_beam_rmat_impl(beam_vec, eta_vec)


@all_impls
def test_colinear_beam_eta_vec(make_beam_rmat_impl, module_name):
    with pytest.raises(RuntimeError):
        make_beam_rmat_impl(cnst.beam_vec, cnst.beam_vec)


@all_impls
def test_orthonormal_1(make_beam_rmat_impl, module_name):
    beam_vec = np.array([1.0, 2.0, 3.0])
    other_vec = np.array([5.0, 2.0, 1.0])

    # force both inputs to be orthogonal and normalized
    eta_vec = np.cross(beam_vec, other_vec)

    beam_vec /= np.linalg.norm(beam_vec)
    eta_vec /= np.linalg.norm(eta_vec)

    rmat = make_beam_rmat_impl(beam_vec, eta_vec)

    # dot(A, A.T) == Identity seems a good orthonormality check
    # Note: atol needed as rtol is not useful for '0.' entries.
    assert_allclose(np.dot(rmat, rmat.T), cnst.identity_3x3, atol=1e+10)


@all_impls
def test_orthonormal_2(make_beam_rmat_impl, module_name):
    # same as above although the inputs are not normalized
    beam_vec = np.array([1.0, 2.0, 3.0])
    other_vec = np.array([5.0, 2.0, 1.0])

    # force both inputs to be orthogonal
    eta_vec = np.cross(beam_vec, other_vec)

    rmat = make_beam_rmat_impl(beam_vec, eta_vec)

    # dot(A, A.T) == Identity seems a good orthonormality check
    # Note: atol needed as rtol is not useful for '0.' entries.
    assert_allclose(np.dot(rmat, rmat.T), cnst.identity_3x3, atol=1e+10)


@all_impls
def test_orthonormal_3(make_beam_rmat_impl, module_name):
    # same as above although the inputs are neither normalized nor orthogonal
    beam_vec = np.array([1.0, 2.0, 3.0])
    other_vec = np.array([5.0, 2.0, 1.0])

    rmat = make_beam_rmat_impl(beam_vec, other_vec)

    # dot(A, A.T) == Identity seems a good orthonormality check
    # Note: atol needed as rtol is not useful for '0.' entries.
    assert_allclose(np.dot(rmat, rmat.T), cnst.identity_3x3, atol=1e+10)
