# tests for make_beam_rmat

from __future__ import absolute_import

from .. import make_beam_rmat as default_make_beam_rmat
from ..xf_numpy import make_beam_rmat as numpy_make_beam_rmat
from ..xf_capi import make_beam_rmat as capi_make_beam_rmat
from ..xf_numba import make_beam_rmat as numba_make_beam_rmat

import pytest

all_impls = pytest.mark.parametrize('make_beam_rmat_impl, module_name', 
                                    [(numpy_make_beam_rmat, 'numpy'),
                                     (capi_make_beam_rmat, 'capi'),
                                     (numba_make_beam_rmat, 'numba'),
                                     (default_make_beam_rmat, 'default')]
                                )


@all_impls
def test_sample1(make_beam_rmat_impl, module_name):
    pass

@all_impls
def test_sample2(make_beam_rmat_impl, module_name):
    pass
