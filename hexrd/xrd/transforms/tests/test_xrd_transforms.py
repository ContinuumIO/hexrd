# Test suite for transforms module.

import unittest

import numpy as np
import numpy.testing as np_testing

import hexrd.xrd.transforms as xfdefault

#class Test_makeEtaFrameRotMat_CAPI(Test_makeEtaFrameRotMat_Python):
#    fn = staticmethod(xfcapi.makeEtaFrameRotMat)


class Test_anglesToGVec(unittest.TestCase):
    fn = staticmethod(xfdefault.anglesToGVec)

    def test_simple(self):
        """test with rMat_s and rMat_c defaults"""

        bHat = np.r_[0.0, 0.0, -1.0]
        eHat = np.r_[1.0, 0.0, 0.0]
        angs = np.array([[np.pi, 0.0]], dtype=np.double)
        expected = np.c_[[0.0, 0.0, 1.0]]

        res = self.fn(angs, bHat, eHat)

        np_testing.assert_almost_equal(res, expected)

    def test_keywords(self):
        # similar to test_simple but with implicit keywords (as they are
        # identities the result should be the same, but actually performs the
        # computations.
        bHat = np.r_[0.0, 0.0, -1.0]
        eHat = np.r_[1.0, 0.0, 0.0]
        angs = np.array([[np.pi, 0.0]], dtype=np.double)
        I = np.eye(3)
        expected = np.c_[[0.0, 0.0, 1.0]]

        res = self.fn(angs, bHat, eHat, rMat_s=I, rMat_c=I)

        np_testing.assert_almost_equal(res, expected)


