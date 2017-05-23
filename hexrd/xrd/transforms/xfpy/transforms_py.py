#
# Python version of transforms related functions.
#
# These are reference implementations.
#
# Some notes on naming conventions:
# - Rotation matrix variables are usually prefixed with rMat
# - Normalized vectors are suffixed with "Hat"
#
# Reference of Frames involved in the computations:
# - Laboratory: Global reference frame
# - Beam: 
# - Detector: Frame of the detector. X and Y match the detector axes,
#             Z represents the plane normal
# - Sample: Sample Frame associated to an specimen
# - Crystal:
################################################################################

import numpy as np

epsf = np.finfo(float).eps
bVec_ref = np.array([[0., 0., 1.]], order='C').T
eta_ref = np.array([[1., 0., 0.]], order='C').T

def unitVector(a):
    """
    normalize array of column vectors (hstacked, axis=0)
    """
    assert a.ndim in [1,2], "incorrect arg shape; must be 1-d or 2-d, yours is %d-d" % (a.ndim)

    m = a.shape[0]; n = 1
    nrm = np.tile(np.sqrt(sum(np.asarray(a)**2, 0)), (m, n))

    zchk = nrm >= epsf
    nrm[zchk] = 1.

    nrma = a/nrm

    return nrma


def makeEtaFrameRotMat(bHat_l, eHat_l):
    """
    make eta basis COB matrix with beam antiparallel with Z

    takes components from ETA fram to LAB
    """

    bHat_l = unitVector(bHat_l.reshape(3, 1))
    eHat_l = unitVector(eHat_l.reshape(3, 1))

    Ye = np.cross(eHat_l.flatten(), bHat_l.flatten())
    if np.sqrt(np.sum(Ye*Ye)) < 1e-8:
        raise RuntimeError, "bHat_l and eHat_l must NOT be colinear!"
    Ye = unitVector(Ye.reshape(3,1))

    Xe = np.cross(bHat_l.flatten(), Ye.flatten()).reshape(3,1)
    return np.hstack([Xe, Ye, -bHat_l])


def anglesToGVec_legacy(angs, bHat_l, eHat_l, rMat_s=None, rMat_c=None):
    """convert angles to GVecs"""

    rMat_e = makeEtaFrameRotMat(bHat_l, eHat_l)
    c0 = np.cos(0.5*angs[:,0])
    c1 = np.cos(angs[:,1])
    s0 = np.sin(0.5*angs[:,0])
    s1 = np.sin(angs[:,1])

    gVec_e = np.vstack([[c0*c1], [c0*s1], [s0]])

    # if rMat_s or rMat_c are not provided, they are assumed to be the identity.
    # instead of computing the dot product we can just skip the computation.
    tmp0 = rMat_e if rMat_s is None else np.dot(rMat_s.T, rMat_e)
    tmp1 = tmp0 if rMat_c is None else np.dot(rMat_c.T, rMat_e)

    return np.dot(tmp1, gVec_e)


def makeOscillRotMat(chi, ome):
    """Create oscillation rotation matrices based on chi and omega angles.

    chi -- float  canting angles.
    ome --  (n,)  oscillation angles
    TODO: finish this
    """

    pass


def anglesToGVec(angs, bHat_l=bVec_ref, eHat_l=eta_ref, chi=0.0, rMat_c=None):
    """convert angles to GVecs using the same interface as the CAPI version.

    angs   -- (n, 3)  array of angles to convert <psi, eta, ome>
    bHat_l --   (3,)  beam unit vector
    eHat_l --   (3,)  e unit vector perpendicular to b.
    chi    --  float  canting angle (chi). Defaults to 0.0.
    rMat_c -- (3, 3)  rotation matrix for the crystal frame. If not provided,
                      Identity will be assumed.
    """

    rMat_e = makeEtaFrameRotMat(bHat_l, eHat_l)
    c0 = np.cos(0.5*angs[:,0])
    c1 = np.cos(angs[:,1])
    s0 = np.sin(0.5*angs[:,0])
    s1 = np.sin(angs[:,1])

    gVec_e = np.vstack([[c0*c1], [c0*s1], [s0]])
    rMat_s = makeOscillRotMat(chi, angs[:,2])

    # if rMat_s or rMat_c are not provided, they are assumed to be the identity.
    # instead of computing the dot product we can just skip the computation.
    tmp0 = rMat_e if rMat_s is None else np.dot(rMat_s.T, rMat_e)
    tmp1 = tmp0 if rMat_c is None else np.dot(rMat_c.T, rMat_e)

    return np.dot(tmp1, gVec_e)
    
