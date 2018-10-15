# tests for validate_angle_ranges

from __future__ import absolute_import

from .. import validate_angle_ranges as default_validate_angle_ranges
from ..xf_numpy import validate_angle_ranges as numpy_validate_angle_ranges
from ..xf_capi import validate_angle_ranges as capi_validate_angle_ranges
#from ..xf_numba import validate_angle_ranges as numba_validate_angle_ranges

import pytest

all_impls = pytest.mark.parametrize('validate_angle_ranges_impl, module_name', 
                                    [(numpy_validate_angle_ranges, 'numpy'),
                                     (capi_validate_angle_ranges, 'capi'),
                                     #(numba_angles_to_gvec, 'numba'),
                                     (default_validate_angle_ranges, 'default')]
                                )


@all_impls
def test_sample1(validate_angle_ranges_impl, module_name):
    pass

@all_impls
def test_sample2(validate_angle_ranges_impl, module_name):
    pass
