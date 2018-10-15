# tests for make_binary_rmat

from __future__ import absolute_import

from .. import make_binary_rmat as default_make_binary_rmat
from ..xf_numpy import make_binary_rmat as numpy_make_binary_rmat
from ..xf_capi import make_binary_rmat as capi_make_binary_rmat
#from ..xf_numba import make_binary_rmat as numba_make_binary_rmat

import pytest

all_impls = pytest.mark.parametrize('make_binary_rmat_impl, module_name', 
                                    [(numpy_make_binary_rmat, 'numpy'),
                                     (capi_make_binary_rmat, 'capi'),
                                     #(numba_angles_to_gvec, 'numba'),
                                     (default_make_binary_rmat, 'default')]
                                )


@all_impls
def test_sample1(make_binary_rmat_impl, module_name):
    pass

@all_impls
def test_sample2(make_binary_rmat_impl, module_name):
    pass
