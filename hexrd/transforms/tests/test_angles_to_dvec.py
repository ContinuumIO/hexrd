# tests for angles_to_dvec

from __future__ import absolute_import

from .. import angles_to_dvec as default_angles_to_dvec
from ..xf_numpy import angles_to_dvec as numpy_angles_to_dvec
from ..xf_capi import angles_to_dvec as capi_angles_to_dvec
from ..xf_numba import angles_to_dvec as numba_angles_to_dvec

import pytest

all_impls = pytest.mark.parametrize('angles_to_dvec_impl, module_name', 
                                    [(numpy_angles_to_dvec, 'numpy'),
                                     (capi_angles_to_dvec, 'capi'),
                                     (numba_angles_to_dvec, 'numba'),
                                     (default_angles_to_dvec, 'default')]
                                )


@all_impls
def test_sample1(angles_to_dvec_impl, module_name):
    pass

@all_impls
def test_sample2(angles_to_dvec_impl, module_name):
    pass
