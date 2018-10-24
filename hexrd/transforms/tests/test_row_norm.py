# tests for row_norm

from __future__ import absolute_import

from .. import row_norm as default_row_norm
from ..xf_numpy import row_norm as numpy_row_norm
#from ..xf_capi import row_norm as capi_row_norm
from ..xf_numba import row_norm as numba_row_norm

import pytest

all_impls = pytest.mark.parametrize('row_norm_impl, module_name', 
                                    [(numpy_row_norm, 'numpy'),
                                     #(capi_row_norm, 'capi'),
                                     (numba_row_norm, 'numba'),
                                     (default_row_norm, 'default')]
                                )


@all_impls
def test_sample1(row_norm_impl, module_name):
    pass

@all_impls
def test_sample2(row_norm_impl, module_name):
    pass
