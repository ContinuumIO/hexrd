# tests for angular_difference

from __future__ import absolute_import

from .. import angular_difference as default_angular_difference
from ..xf_numpy import angular_difference as numpy_angular_difference
#from ..xf_capi import angular_difference as capi_angular_difference
#from ..xf_numba import angular_difference as numba_angular_difference

import pytest

all_impls = pytest.mark.parametrize('angular_difference_impl, module_name', 
                                    [(numpy_angular_difference, 'numpy'),
                                     #(capi_angular_difference, 'capi'),
                                     #(numba_angles_to_gvec, 'numba'),
                                     (default_angular_difference, 'default')]
                                )


@all_impls
def test_sample1(angular_difference_impl, module_name):
    pass

@all_impls
def test_sample2(angular_difference_impl, module_name):
    pass
