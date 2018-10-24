# tests for unit_vector

from __future__ import absolute_import

from .. import unit_vector as default_unit_vector
from ..xf_numpy import unit_vector as numpy_unit_vector
from ..xf_capi import unit_vector as capi_unit_vector
from ..xf_numba import unit_vector as numba_unit_vector

import pytest

all_impls = pytest.mark.parametrize('unit_vector_impl, module_name', 
                                    [(numpy_unit_vector, 'numpy'),
                                     (capi_unit_vector, 'capi'),
                                     (numba_unit_vector, 'numba'),
                                     (default_unit_vector, 'default')]
                                )


@all_impls
def test_sample1(unit_vector_impl, module_name):
    pass

@all_impls
def test_sample2(unit_vector_impl, module_name):
    pass
