# tests for rotate_vecs_about_axis

from __future__ import absolute_import

from .. import rotate_vecs_about_axis as default_rotate_vecs_about_axis
from ..xf_numpy import rotate_vecs_about_axis as numpy_rotate_vecs_about_axis
from ..xf_capi import rotate_vecs_about_axis as capi_rotate_vecs_about_axis
#from ..xf_numba import rotate_vecs_about_axis as numba_rotate_vecs_about_axis

import pytest

all_impls = pytest.mark.parametrize('rotate_vecs_about_axis_impl, module_name', 
                                    [(numpy_rotate_vecs_about_axis, 'numpy'),
                                     (capi_rotate_vecs_about_axis, 'capi'),
                                     #(numba_angles_to_gvec, 'numba'),
                                     (default_rotate_vecs_about_axis, 'default')]
                                )


@all_impls
def test_sample1(rotate_vecs_about_axis_impl, module_name):
    pass

@all_impls
def test_sample2(rotate_vecs_about_axis_impl, module_name):
    pass
