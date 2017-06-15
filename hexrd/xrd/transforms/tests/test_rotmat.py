import unittest

import numpy as np
from numpy import testing as np_testing

import hexrd.xrd.transforms
import hexrd.xrd.transforms as xfdefault
import hexrd.xrd.transforms.xfpy as xfpy
import hexrd.xrd.transforms.xfnumba as xfnumba
import hexrd.xrd.transforms.xfcapi as xfcapi

# ==============================================================================

# makeEtaFrameRotMat

class Test_makeEtaFrameRotMat(unittest.TestCase):
    fn = staticmethod(xfdefault.makeEtaFrameRotMat)

    def test_simple(self):
        # with this setup we should get something similar to identity as result
        # note that in practice this is the only case that is used.
        bHat = np.r_[0.0, 0.0, -1.0]
        eHat = np.r_[1.0, 0.0, 0.0]
        res = self.fn(bHat, eHat)
        np_testing.assert_almost_equal(res, np.eye(3))

class Test_makeEtaFrameRotMat_py(Test_makeEtaFrameRotMat):
    fn = staticmethod(xfpy.makeEtaFrameRotMat)

# numba version not available
# class Test_makeEtaFrameRotMat_numba(Test_makeEtaFrameRotMat):
#     fn = staticmethod(xfnumba.makeEtaFrameRotMat)

class Test_makeEtaFrameRotMat_capi(Test_makeEtaFrameRotMat):
    fn = staticmethod(xfcapi.makeEtaFrameRotMat)

# ==============================================================================

# makeOscillRotMat

class TestMakeOscillRotMat(unittest.TestCase):
    fn = staticmethod(xfdefault.makeOscillRotMat)

    def test_trivial(self):
        # 0 angles results in identity matrix
        expected = np.eye(3)
        res = self.fn(0.0, 0.0)
        np_testing.assert_almost_equal(res, expected)

    def test_chi(self):
        # rotation of pi/2 only on chi
        expected = np.array([[ 1.0,  0.0,  0.0],
                             [ 0.0,  0.0, -1.0],
                             [ 0.0,  1.0,  0.0]])
        res = self.fn(0.5*np.pi, 0.0)
        np_testing.assert_almost_equal(res, expected)

    def test_ome(self):
        # rotation of pi/2 only on ome
        expected = np.array([[ 0.0,  0.0,  1.0],
                             [ 0.0,  1.0,  0.0],
                             [-1.0,  0.0,  0.0]])
        res = self.fn(0.0, 0.5*np.pi)
        print (res, res.shape)
        np_testing.assert_almost_equal(res, expected)

    def test_chiome(self):
        # rotation of pi/2 for both chi and ome)
        expected = np.array([[ 0.0,  0.0,  1.0],
                             [ 1.0,  0.0,  0.0],
                             [ 0.0,  1.0,  0.0]])
        res = self.fn(0.5*np.pi, 0.5*np.pi)
        np_testing.assert_almost_equal(res, expected)


class TestMakeOscillRotMat_py(TestMakeOscillRotMat):
    fn = staticmethod(xfpy.makeOscillRotMat)

# numba version not available
# class TestMakeOscillRotMat_numba(TestMakeOscillRotMat):
#     fn = staticmethod(xfnumba.makeOscillRotMat)

class TestMakeOscillRotMat_capi(TestMakeOscillRotMat):
    fn = staticmethod(xfcapi.makeOscillRotMat)


class TestMakeOscillRotMatArray(unittest.TestCase):
    fn = staticmethod(xfdefault.makeOscillRotMatArray)

    def test_multiple(self):
        # checks the array version of the functionality
        chi = 0.5*np.pi
        ome = np.array([0.0, 0.5*np.pi])
        expected = np.array([[[ 1.0,  0.0,  0.0],
                              [ 0.0,  0.0, -1.0],
                              [ 0.0,  1.0, 0.0 ]],

                             [[ 0.0,  0.0,  1.0],
                              [ 1.0,  0.0,  0.0],
                              [ 0.0,  1.0,  0.0]]])
        res = self.fn(chi, ome)
        np_testing.assert_almost_equal(res, expected)

class TestMakeOscillRotMat_py(TestMakeOscillRotMat):
    fn = staticmethod(xfpy.makeOscillRotMatArray)

# numba version not available
#class TestMakeOscillRotMat_numba(TestMakeOscillRotMat):
#    fn = staticmethod(xfnumba.makeOscillRotMatArray)

class TestMakeOscillRotMat_capi(TestMakeOscillRotMat):
    fn = staticmethod(xfcapi.makeOscillRotMatArray)

# ==============================================================================

class TestMakeDetectorRotMat(unittest.TestCase):
    pass


class TestMakeRotMatOfExpMap(unittest.TestCase):
    pass


class TestMakeRotMatOfQuat(unittest.TestCase):
    pass


class TestMakeBinaryRotMat(unittest.TestCase):
    pass


