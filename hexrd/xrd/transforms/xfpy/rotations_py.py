"""Submodule with general rotation matrix related code"""

from __future__ import absolute_import

import numpy as np

# this is ugly
from .... import matrixutil as mutil

def makeRotMatAroundX(angle, out=None):
    out = out if out is not None else np.empty((3,3))
    c_a = np.cos(angle)
    s_a = np.sin(angle)

    out[0][0] =  1.0; out[0][1] =  0.0; out[0][2] =  0.0
    out[1][0] =  0.0; out[1][1] =  c_a; out[1][2] = -s_a
    out[2][0] =  0.0; out[2][1] =  s_a; out[2][2] =  c_a

    return out


def makeRotMatAroundY(angle, out=None):
    out = out if out is not None else np.empty((3,3))
    c_a = np.cos(angle)
    s_a = np.sin(angle)

    out[0][0] =  c_a; out[0][1] =  0.0; out[0][2] =  s_a
    out[1][0] =  0.0; out[1][1] =  1.0; out[1][2] =  0.0
    out[2][0] = -s_a; out[2][1] =  0.0; out[2][2] =  c_a

    return out


def makeRotMatAroundZ(angle, out=None):
    out = out if out is not None else np.empty((3,3))
    c_a = np.cos(angle)
    s_a = np.sin(angle)

    out[0][0] =  c_a; out[0][1] = -s_a; out[0][2] =  0.0
    out[1][0] =  s_a; out[1][1] =  c_a; out[1][2] =  0.0
    out[2][0] =  0.0; out[2][1] =  0.0; out[2][2] =  1.0

    return out


def makeOscillRotMat(chi, ome):
    """Create oscillation rotation matrices based on chi and omega.

    chi -- float  canting angle.
    ome -- float  oscillation angle.

    note: makeOscillRotMatArray is a superset of this.
    """
    c_chi = np.cos(chi)
    s_chi = np.sin(chi)
    c_ome = np.cos(ome)
    s_ome = np.sin(ome)

    return np.array([[       c_ome,    0.,        s_ome],
                     [ s_chi*s_ome, c_chi, -s_chi*c_ome],
                     [-c_chi*s_ome, s_chi,  c_chi*c_ome]])


def makeOscillRotMatArray(chi, ome_array):
    """Create oscillation rotation matrices based on chi and omegas.

    chi       -- float      canting angle.
    ome_array -- array (n,) oscillation angles.
    """

    c_chi = np.cos(chi)
    s_chi = np.sin(chi)
    c_ome = np.cos(ome_array) # vector
    s_ome = np.sin(ome_array) # vector

    outer_dim = 1 if s_ome.ndim is 0 else len(s_ome)
    result = np.zeros((outer_dim, 3, 3))

    result[:, 0, 0] = c_ome
    # result[:, 0, 1] = 0.0
    result[:, 0, 2] = s_ome
    result[:, 1, 0] = s_chi*s_ome
    result[:, 1, 1] = c_chi
    result[:, 1, 2] = -s_chi*c_ome
    result[:, 2, 0] = -c_chi*s_ome
    result[:, 2, 1] = s_chi
    result[:, 2, 2] = c_chi*c_ome

    return result[0] if s_ome.ndim is 0 else result


def makeEtaFrameRotMat(bHat_l, eHat_l):
    """
    make eta basis COB matrix with beam antiparallel with Z

    bHat_l -- beam vector
    eHat_l -- reference azimuth vector

    returns:
        rotation matrix to transform from ETA frame to LAB
    """

    bHat_l = mutil.unitVector(bHat_l.reshape(3, 1))
    eHat_l = mutil.unitVector(eHat_l.reshape(3, 1))

    Ye = np.cross(eHat_l.flatten(), bHat_l.flatten())
    if np.sqrt(np.sum(Ye*Ye)) < 1e-8:
        raise RuntimeError, "bHat_l and eHat_l must NOT be colinear!"
    Ye = mutil.unitVector(Ye.reshape(3,1))

    Xe = np.cross(bHat_l.flatten(), Ye.flatten()).reshape(3,1)

    return np.hstack([Xe, Ye, -bHat_l])
