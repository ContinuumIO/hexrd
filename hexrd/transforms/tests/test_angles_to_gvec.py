# tests for angles_to_gvec

from __future__ import absolute_import

from .. import angles_to_gvec as default_angles_to_gvec
from ..xf_numpy import angles_to_gvec as numpy_angles_to_gvec
from ..xf_capi import angles_to_gvec as capi_angles_to_gvec
from ..xf_numba import angles_to_gvec as numba_angles_to_gvec

import pytest

all_impls = pytest.mark.parametrize('angles_to_gvec_impl, module_name', 
                                    [(numpy_angles_to_gvec, 'numpy'),
                                     (capi_angles_to_gvec, 'capi'),
                                     (numba_angles_to_gvec, 'numba'),
                                     (default_angles_to_gvec, 'default')]
                                )


@all_impls
def test_sample1(angles_to_gvec_impl, module_name):
    pass

@all_impls
def test_sample2(angles_to_gvec_impl, module_name):
    pass
