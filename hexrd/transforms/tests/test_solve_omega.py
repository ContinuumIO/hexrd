# tests for solve_omega

from __future__ import absolute_import

from .. import solve_omega as default_solve_omega
from ..xf_numpy import solve_omega as numpy_solve_omega
#from ..xf_capi import solve_omega as capi_solve_omega
#from ..xf_numba import solve_omega as numba_solve_omega

import pytest

all_impls = pytest.mark.parametrize('solve_omega_impl, module_name', 
                                    [(numpy_solve_omega, 'numpy'),
                                     #(capi_solve_omega, 'capi'),
                                     #(numba_angles_to_gvec, 'numba'),
                                     (default_solve_omega, 'default')]
                                )


@all_impls
def test_sample1(solve_omega_impl, module_name):
    pass

@all_impls
def test_sample2(solve_omega_impl, module_name):
    pass
