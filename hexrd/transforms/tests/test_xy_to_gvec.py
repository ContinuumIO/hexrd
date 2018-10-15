# tests for xy_to_gvec

from __future__ import absolute_import

from .. import xy_to_gvec as default_xy_to_gvec
from ..xf_numpy import xy_to_gvec as numpy_xy_to_gvec
from ..xf_capi import xy_to_gvec as capi_xy_to_gvec
#from ..xf_numba import xy_to_gvec as numba_xy_to_gvec

import pytest

all_impls = pytest.mark.parametrize('xy_to_gvec_impl, module_name', 
                                    [(numpy_xy_to_gvec, 'numpy'),
                                     (capi_xy_to_gvec, 'capi'),
                                     #(numba_angles_to_gvec, 'numba'),
                                     (default_xy_to_gvec, 'default')]
                                )


@all_impls
def test_sample1(xy_to_gvec_impl, module_name):
    pass

@all_impls
def test_sample2(xy_to_gvec_impl, module_name):
    pass
