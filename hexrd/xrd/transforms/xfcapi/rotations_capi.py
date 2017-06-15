from __future__ import absolute_import

import numpy as np
from hexrd.xrd import _transforms_CAPI


def makeOscillRotMat(chi, ome):
    """
    oscillAngles = [chi, ome]
    """
    arg = np.array([chi, ome])

    return _transforms_CAPI.makeOscillRotMat(arg)


def makeOscillRotMatArray(chi, omeArray):
    """
    Applies makeOscillAngles multiple times, for one
    chi value and an array of omega values.
    """
    chi = float(chi)
    arg = np.ascontiguousarray(omeArray)
    return _transforms_CAPI.makeOscillRotMatArray(chi, arg)


def makeEtaFrameRotMat(bHat_l, eHat_l):
    arg1 = np.ascontiguousarray(bHat_l.flatten())
    arg2 = np.ascontiguousarray(eHat_l.flatten())
    return _transforms_CAPI.makeEtaFrameRotMat(arg1, arg2)
