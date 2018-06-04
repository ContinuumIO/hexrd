# tests for map_angle

from __future__ import absolute_import

from .. import map_angle as default_map_angle
from ..xf_numpy import map_angle as numpy_map_angle
#from ..xf_capi import map_angle as capi_map_angle
#from ..xf_numba import map_angle as numba_map_angle

import pytest

all_impls = pytest.mark.parametrize('map_angle_impl, module_name', 
                                    [(numpy_map_angle, 'numpy'),
                                     #(capi_map_angle, 'capi'),
                                     #(numba_angles_to_gvec, 'numba'),
                                     (default_map_angle, 'default')]
                                )


@all_impls
def test_sample1(map_angle_impl, module_name):
    pass

@all_impls
def test_sample2(map_angle_impl, module_name):
    pass
