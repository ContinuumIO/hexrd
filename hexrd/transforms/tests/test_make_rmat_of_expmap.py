# tests for make_rmat_of_expmap

from __future__ import absolute_import

from .. import make_rmat_of_expmap as default_make_rmat_of_expmap
from ..xf_numpy import make_rmat_of_expmap as numpy_make_rmat_of_expmap
from ..xf_capi import make_rmat_of_expmap as capi_make_rmat_of_expmap
from ..xf_numba import make_rmat_of_expmap as numba_make_rmat_of_expmap

import pytest

all_impls = pytest.mark.parametrize('make_rmat_of_expmap_impl, module_name', 
                                    [(numpy_make_rmat_of_expmap, 'numpy'),
                                     (capi_make_rmat_of_expmap, 'capi'),
                                     (numba_make_rmat_of_expmap, 'numba'),
                                     (default_make_rmat_of_expmap, 'default')]
                                )


@all_impls
def test_sample1(make_rmat_of_expmap_impl, module_name):
    pass

@all_impls
def test_sample2(make_rmat_of_expmap_impl, module_name):
    pass
