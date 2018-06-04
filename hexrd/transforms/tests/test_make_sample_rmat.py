# tests for make_sample_rmat

from __future__ import absolute_import

from .. import make_sample_rmat as default_make_sample_rmat
from ..xf_numpy import make_sample_rmat as numpy_make_sample_rmat
#from ..xf_capi import make_sample_rmat as capi_make_sample_rmat
#from ..xf_numba import make_sample_rmat as numba_make_sample_rmat

import pytest

all_impls = pytest.mark.parametrize('make_sample_rmat_impl, module_name', 
                                    [(numpy_make_sample_rmat, 'numpy'),
                                     #(capi_make_sample_rmat, 'capi'),
                                     #(numba_angles_to_gvec, 'numba'),
                                     (default_make_sample_rmat, 'default')]
                                )


@all_impls
def test_sample1(make_sample_rmat_impl, module_name):
    pass

@all_impls
def test_sample2(make_sample_rmat_impl, module_name):
    pass
