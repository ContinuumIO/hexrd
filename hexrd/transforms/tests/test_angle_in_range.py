# tests for angles_in_range

from __future__ import absolute_import

from .. import angles_in_range as default_angles_in_range
from ..xf_numpy import angles_in_range as numpy_angles_in_range
#from ..xf_capi import angles_in_range as capi_angles_in_range
#from ..xf_numba import angles_in_range as numba_angles_in_range

import pytest

all_impls = pytest.mark.parametrize('angles_in_range_impl, module_name', 
                                    [(numpy_angles_in_range, 'numpy'),
                                     #(capi_angles_in_range, 'capi'),
                                     #(numba_angles_to_gvec, 'numba'),
                                     (default_angles_in_range, 'default')]
                                )


@all_impls
def test_sample1(angles_in_range_impl, module_name):
    pass

@all_impls
def test_sample2(angles_in_range_impl, module_name):
    pass
