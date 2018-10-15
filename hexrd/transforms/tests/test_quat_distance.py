# tests for quat_distance

from __future__ import absolute_import

from .. import quat_distance as default_quat_distance
from ..xf_numpy import quat_distance as numpy_quat_distance
from ..xf_capi import quat_distance as capi_quat_distance
#from ..xf_numba import quat_distance as numba_quat_distance

import pytest

all_impls = pytest.mark.parametrize('quat_distance_impl, module_name', 
                                    [(numpy_quat_distance, 'numpy'),
                                     (capi_quat_distance, 'capi'),
                                     #(numba_angles_to_gvec, 'numba'),
                                     (default_quat_distance, 'default')]
                                )


@all_impls
def test_sample1(quat_distance_impl, module_name):
    pass

@all_impls
def test_sample2(quat_distance_impl, module_name):
    pass
