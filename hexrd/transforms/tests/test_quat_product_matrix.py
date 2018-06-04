# tests for quat_product_matrix

from __future__ import absolute_import

from .. import quat_product_matrix as default_quat_product_matrix
from ..xf_numpy import quat_product_matrix as numpy_quat_product_matrix
#from ..xf_capi import quat_product_matrix as capi_quat_product_matrix
#from ..xf_numba import quat_product_matrix as numba_quat_product_matrix

import pytest

all_impls = pytest.mark.parametrize('quat_product_matrix_impl, module_name', 
                                    [(numpy_quat_product_matrix, 'numpy'),
                                     #(capi_quat_product_matrix, 'capi'),
                                     #(numba_angles_to_gvec, 'numba'),
                                     (default_quat_product_matrix, 'default')]
                                )


@all_impls
def test_sample1(quat_product_matrix_impl, module_name):
    pass

@all_impls
def test_sample2(quat_product_matrix_impl, module_name):
    pass
