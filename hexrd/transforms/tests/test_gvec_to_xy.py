# tests for angles_to_gvec

from __future__ import absolute_import

from .. import gvec_to_xy as default_gvec_to_xy
from ..xf_numpy import gvec_to_xy as numpy_gvec_to_xy
from ..xf_capi import gvec_to_xy as capi_gvec_to_xy
#from ..xf_numba import gvec_to_xy as numba_gvec_to_xy

import pytest

all_impls = pytest.mark.parametrize('gvec_to_xy_impl, module_name', 
                                    [(numpy_gvec_to_xy, 'numpy'),
                                     (capi_gvec_to_xy, 'capi'),
                                     #(numba_gvec_to_xy, 'numba'),
                                     (default_gvec_to_xy, 'default')]
                                )


@all_impls
def test_sample1(gvec_to_xy_impl, module_name):
    pass

@all_impls
def test_sample2(gvec_to_xy_impl, module_name):
    pass
