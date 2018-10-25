# tests for make_beam_rmat

from __future__ import absolute_import

from .. import make_beam_rmat as default_make_beam_rmat
from ..xf_numpy import make_beam_rmat as numpy_make_beam_rmat
from ..xf_capi import make_beam_rmat as capi_make_beam_rmat
from ..xf_numba import make_beam_rmat as numba_make_beam_rmat

from ... import constants as cnst

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

    beam_vec = cnst.beam_vec
    eta_vec = cnst.eta_vec

    rmat = make_beam_rmat_impl(cnst.beam_vec, cnst.eta_vec)

    assert_allclose(rmat, cnst.identity_3x3)
