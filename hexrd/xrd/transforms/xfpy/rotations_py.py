"""Submodule with general rotation matrix related code"""

import numpy as np

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
