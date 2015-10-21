from __future__ import division
from math import hypot, fabs, cos, sin, acos, pi, atan2, asin, sqrt
from numba import guvectorize, jit
import numpy as np

#
# We don't necessarily need to expose full Python versions
# of all of these functions, but we do anyway for testing.
#

@jit('f8(f8[:])', nopython=True, nogil=True)
def norm3_(vec):
    return hypot(hypot(vec[0], vec[1]), vec[2])

@guvectorize('(f8[:],f8[:],f8[:])','(n),(n)->()', nopython=True)
def norm3_gufunc_(vec, dummy, out):
    out[0] = norm3_(vec)

VEC3 = np.empty((3,))
def norm3(vec, out=None):
    '''
    Optimized vector norm for 3 vectors (and arrays of same).
    '''
    return norm3_gufunc_(vec, VEC3, out)


@jit('f8(f8[:])', nopython=True, nogil=True)
def normq_(vec):
    return hypot(hypot(vec[0], vec[1]), hypot(vec[2], vec[3]))

@guvectorize('(f8[:],f8[:],f8[:])','(n),(n)->()', nopython=True)
def normq_gufunc_(vec, dummy, out):
    out[0] = normq_(vec)

VEC4 = np.empty((4,))
def normq(vec, out=None):
    '''
    Optimized norm for quaternion (and arrays of same).
    '''
    return normq_gufunc_(vec, VEC4, out)


@jit('f8(f8[:],f8[:])', nopython=True, nogil=True)
def dot3_(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

@guvectorize('(f8[:],f8[:],f8[:],f8[:])', '(n),(n),(n)->()', nopython=True)
def dot3_gufunc_(a, b, dummy, out):
    out[0] = dot3_(a, b)

VEC3 = np.empty((3,))
def dot3(a, b, out=None):
    '''
    Optimized dot product for 3 vectors (and arrays of same).
    '''
    return dot3_gufunc_(a, b, VEC3, out)


tiny64 = np.finfo(np.float64).tiny
@jit(nopython=True, nogil=True)
def quatToHomochoric_(q, h):
    phi = 2.0 * acos(q[0])
    tmp = (0.75*(phi-sin(phi)))**(1.0/3.0) / (tiny64 + norm3_(q[1:]))
    h[0] = q[1] * tmp
    h[1] = q[2] * tmp
    h[2] = q[3] * tmp
    return h

@guvectorize('(f8[:],f8[:],f8[:],f8[:])', '(m),(m),(n)->(n)', nopython=True)
def quatToHomochoric_gufunc_(q, d4, d3, h):
    quatToHomochoric_(q, h)

def quatToHomochoric(quats, out=None):
    '''
    Converts quaternions (or arrays of same) into homochoric coordinates.
    '''
    return quatToHomochoric_gufunc_(quats, VEC4, VEC3, out)


@jit(nopython=True, nogil=True)
# Reference: http://fabiensanglard.net/doom3_documentation/37726-293748.pdf
# I needed to swap elements and negate some signs to get the results to
# match up with the original C code. I'm assuming this is due to differences
# in the coordinate systems assumed. But it avoids the use of trigonometric
# functions, which is certainly a win.
def quatToRotMat_(q, m):
    x2 = 2.0 * q[0]
    y2 = 2.0 * q[1]
    z2 = 2.0 * q[2]

    xx2 = q[0] * x2
    yy2 = q[1] * y2
    zz2 = q[2] * z2
    m[0,0] = ( xx2 + yy2 ) - 1.0
    m[1,1] = ( xx2 + zz2 ) - 1.0
    m[2,2] = 1.0 - ( yy2 + zz2 )

    yz2 = q[1] * z2
    wx2 = q[3] * x2
    m[0,1] = yz2 - wx2
    m[1,0] = yz2 + wx2

    xy2 = q[0] * y2
    wz2 = q[3] * z2
    m[1,2] = wz2 - xy2
    m[2,1] = wz2 + xy2

    xz2 = q[0] * z2
    wy2 = q[3] * y2
    m[2,0] = wy2 - xz2
    m[0,2] = wy2 + xz2

@guvectorize('(f8[:],f8[:],f8[:],f8[:,:])', '(m),(m),(n)->(n,n)', nopython=True)
def quatToRotMat_gufunc_(q, d4, d3, m):
    quatToRotMat_(q, m)

def quatToRotMat(quats, out=None):
    """
    Make rotation matrices from unit quaternions

    """
    return quatToRotMat_gufunc_(quats, VEC4, VEC3, out)



sqr05 = sqrt(0.5)
@jit('(f8[:],f8[:,:])', nopython=True, nogil=True)
def vecMVToSymm_(A,symm):
    symm[0, 0] = A[0]
    symm[0, 1] = symm[1, 0] = A[5] * sqr05
    symm[0, 2] = symm[2, 0] = A[4] * sqr05
    symm[1, 1] = A[1]
    symm[1, 2] = symm[2, 1] = A[3] * sqr05
    symm[2, 2] = A[2]

@guvectorize('(f8[:],f8[:],f8[:],f8[:,:])','(n6),(n6),(n3)->(n3,n3)', nopython=True)
def vecMVToSymm_gufunc_(A, d6, d3, symm):
    vecMVToSymm_(A, symm)

VEC6 = np.empty((6,))
def vecMVToSymm(A, out=None):
    """
    convert from Mandel-Voigt vector to symmetric matrix representation
    """
    return vecMVToSymm_gufunc_(A, VEC6, VEC3, out)


etaFrameTol = 1.0 - sqrt(np.finfo(np.float32).eps)
@jit('(f8[:],f8[:],f8[:,:])', nopython=True, nogil=True)
def etaFrameToRotMat_(beamVec, etaVec, rMat):
    np.divide(beamVec, -norm3_(beamVec), rMat[:, 2])
    dt = dot3_(rMat[:, 2], etaVec)
    if fabs(dt) >= norm3_(etaVec) * etaFrameTol:
        rMat[0, 0] = rMat[1, 0] = rMat[2, 0] = 0.0
        rMat[0, 1] = rMat[1, 1] = rMat[2, 1] = np.nan
    else:
        rMat[0, 0] = etaVec[0] - dt * rMat[0, 2]
        rMat[1, 0] = etaVec[1] - dt * rMat[1, 2]
        rMat[2, 0] = etaVec[2] - dt * rMat[2, 2]
        np.divide(rMat[:, 0], norm3_(rMat[:, 0]), rMat[:, 0])
        rMat[0, 1] = rMat[1, 2] * rMat[2, 0] - rMat[2, 2] * rMat[1, 0]
        rMat[1, 1] = rMat[2, 2] * rMat[0, 0] - rMat[0, 2] * rMat[2, 0]
        rMat[2, 1] = rMat[0, 2] * rMat[1, 0] - rMat[1, 2] * rMat[0, 0]

@guvectorize('(f8[:],f8[:],f8[:],f8[:,:])','(n3),(n3),(n3)->(n3,n3)', nopython=True)
def etaFrameToRotMat_gufunc_(beamVec, etaVec, dummy, rMat):
    etaFrameToRotMat_(beamVec, etaVec, rMat)

def etaFrameToRotMat(beamVec, etaVec, out=None):
    return etaFrameToRotMat_gufunc_(beamVec, etaVec, VEC3, out)


@jit('f8[:](f8[:,:],f8[:])', nopython=True, nogil=True)
def mv3_(a, b):
    c = np.empty((3,))
    c[0] = dot3_(a[0], b)
    c[1] = dot3_(a[1], b)
    c[2] = dot3_(a[2], b)
    return c

@jit('(f8[:],f8,f8,f8[:,:],f8[:,:],f8,f8[:,:],f8[:,:],f8[:,:])', nopython=True, nogil=True)
def oscillAnglesOfHKL_(hkl, cc, sc, rMat_c, bMat, wavelength, vMat_s, rMat_e, oangs):
    # reciprocal lattice vector in SAMPLE frame
    gHat_s = mv3_(vMat_s, mv3_(rMat_c, mv3_(bMat, hkl)))
    
    # Normalize
    nrm0 = norm3_(gHat_s)
    np.divide(gHat_s, nrm0, gHat_s)

    # sin of the Bragg angle assoc. with wavelength
    sintht = 0.5 * wavelength * nrm0              

    # coefficients for harmonic equation
    # The normalized beam vector is the negative of the third column of
    # the eta frame rotation matrix.
    bHat = rMat_e[:,2]
    t1 = sc * bHat[1] - cc * bHat[2]
    t2 = cc * bHat[1] + sc * bHat[2]
    a = - gHat_s[2] * bHat[0] - gHat_s[0] * t1
    b = - gHat_s[0] * bHat[0] + gHat_s[2] * t1
    c =               -sintht + gHat_s[1] * t2

    # form solution
    abMag = hypot(a,b)
    phaseAng = atan2(b,a)
    rhs = c / abMag

    # quick exit for an infeasible case
    if fabs(rhs) > 1.0:
        oangs[0,0] = oangs[0,1] = oangs[0,2] = \
            oangs[1,0] = oangs[1,1] = oangs[1,2] = np.nan
        return

    rhsAng = asin(rhs)
    oangs[0,2] = rhsAng - phaseAng
    oangs[1,2] = pi - rhsAng - phaseAng

    tmp = np.empty((3,))
    oangs[0,0] = oangs[1,0] = 2.0 * asin(sintht)
    for i in range(2):
        co = cos(oangs[i, 2])
        so = sin(oangs[i, 2])
        tmp[0] = co * gHat_s[0] + so * gHat_s[2]
        tmp[1] = cc * gHat_s[1] + sc * ( so * gHat_s[0] - co * gHat_s[2] )
        tmp[2] = sc * gHat_s[1] + cc * ( co * gHat_s[2] - so * gHat_s[0] )
        oangs[i,1] = atan2(dot3_(tmp, rMat_e[:,1]), dot3_(tmp, rMat_e[:,0]))

@guvectorize(
    '(f8[:],f8[:],f8[:],f8[:,:],f8[:,:],f8[:],f8[:,:],f8[:,:],f8[:],f8[:,:])',
    '(n3),(),(),(n3,n3),(n3,n3),(),(n3,n3),(n3,n4),(n2)->(n2,n3)', nopython=True)
def oscillAnglesOfHKL_gufunc_(hkl, cc, sc, rMat_c, bMat, wavelength, vMat_s, rMat_e, dummy2, oangs):
    oscillAnglesOfHKL_(hkl, cc[0], sc[0], rMat_c, bMat, wavelength[0], vMat_s, rMat_e, oangs)

VEC2 = np.empty((2,))
vInv_ref = np.array([1.,1.,1.,0.,0.,0.])
bVec_ref = np.array([0.,0.,-1.])
eta_ref = np.array([1.,0.,0.])
def oscillAnglesOfHKL(hkl, chi, rMat, bMat, wavelength, vInv_s=vInv_ref, beamVec=bVec_ref, etaVec=eta_ref, out=None):
    vMat_s = vecMVToSymm(vInv_s)                # stretch tensor in SAMPLE frame
    rMat_e = etaFrameToRotMat(beamVec, etaVec)  # eta basis COB with beam antiparallel with Z
    nrmat = rMat.ndim - 2
    nhkl = hkl.ndim - 1
    if nrmat > 0:
        hkl = hkl.reshape((1,) * nrmat + hkl.shape)
    if nhkl > 0:
        rMat = rMat.reshape(rMat.shape[:1] + ((1,) * nhkl) + rMat.shape[1:])
    return oscillAnglesOfHKL_gufunc_(hkl, np.cos(chi), np.sin(chi), rMat, bMat, wavelength, 
        vMat_s, rMat_e, VEC2, out)


@jit(nopython=True, nogil=True)
def find_in_range_(value, spans):
    """find the index in spans where value >= spans[i] and value < spans[i].

    spans is an ordered array where spans[i] <= spans[i+1] (most often <
    will hold).

    If value is not in the range [spans[0], spans[-1][, then -2 is returned.

    This is equivalent to "bisect_right" in the bisect package, in which
    code it is based, and it is somewhat similar to NumPy's searchsorted,
    but non-vectorized

    """
    if value < spans[0] or value >= spans[-1]:
        return -2
    # from the previous check, we know 0 is not a possible result
    li = 0
    ri = len(spans)
    while li < ri:
        mi = (li + ri) // 2
        if value < spans[mi]:
            ri = mi
        else:
            li = mi+1
    return li

twopi = 2 * np.pi
@jit(nopython=True, nogil=True)
def map_angle_(angle, offset):
    """
    Recasts an angle to the range [offset,offset+2*pi]

    """
    return np.mod(angle-offset, twopi)+offset


@jit('b1(i8,i8,i8,i8,f8[:,:],f8)', nopython=True, nogil=True)
def check_dilated_(eta, ome, dpix_eta, dpix_ome, etaOmeMap, threshold):
    """
    Check if there exists a sample over the given threshold in the etaOmeMap
    at (eta, ome), with a tolerance of (dpix_eta, dpix_ome) samples.
    """
    i_max, j_max = etaOmeMap.shape
    ome_start, ome_stop = max(ome - dpix_ome, 0), min(ome + dpix_ome + 1, i_max)
    eta_start, eta_stop = max(eta - dpix_eta, 0), min(eta + dpix_eta + 1, j_max)
    for i in range(ome_start, ome_stop):
        for j in range(eta_start, eta_stop):
            if etaOmeMap[i,j] > threshold:
                return True
    return False

#'(f8[:],f8,f8[:,:], f8,f8[:],f8[:],i8, f8,f8[:],f8[:],i8)', 
@jit(nopython=True, nogil=True)
def angle_is_hit_(angles, threshold, etaOmeMap,
                  eta_offset, valid_eta_spans, etaEdges, dpix_eta,
                  ome_offset, valid_ome_spans, omeEdges, dpix_ome):
    """perform work on one of the angles.

    This includes:
    - filtering nan values
    - filtering out angles not in the specified spans
    - checking that the discretized angle fits into the sensor range (maybe
      this could be merged with the previous test somehow, for extra speed)
    - actual check for a hit, using dilation for the tolerance.
    Note the function returns both, if it was a hit and if it passed the the
    filtering, as we'll want to discard the filtered values when computing
    the hit percentage.

    """
    if np.isnan(angles[0]):
        return (0,0)
    eta = map_angle_(angles[1], eta_offset)
    if find_in_range_(eta, valid_eta_spans) & 1 == 0:
        # index is even: out of valid eta spans
        return (0,0)
    ome = map_angle_(angles[2], ome_offset)
    if find_in_range_(ome, valid_ome_spans) & 1 == 0:
        # index is even: out of valid ome spans
        return (0,0)
    # discretize the angles
    eta_idx = find_in_range_(eta, etaEdges) - 1
    if eta_idx < 0:
        # out of range
        return (0,0)
    ome_idx = find_in_range_(ome, omeEdges) - 1
    if ome_idx < 0:
        # out of range
        return (0,0)
    out = check_dilated_(eta_idx, ome_idx, dpix_eta, dpix_ome, etaOmeMap, threshold)
    return (1 if out else 0, 1)


@guvectorize(
'(f8[:],i8[:],f8[:],f8[:,:,:], f8[:],f8[:],f8[:],i8[:], f8[:],f8[:],f8[:],i8[:], f8[:],i8[:])',
'(n3),(),(m1),(m1,m2,m3), (),(m4),(m5),(), (),(m7),(m8),(), (n2)->(n2)', nopython=True)
def angle_is_hit_gufunc_(ang, hkl, threshold, etaOmeMaps,
        offset_eta, valid_eta_spans, etaEdges, dpix_eta,
        offset_ome, valid_ome_spans, omeEdges, dpix_ome, 
        dummy2, out):
    out[0], out[1] = angle_is_hit_(
            ang, threshold[hkl[0]], etaOmeMaps[hkl[0]],
            offset_eta[0], valid_eta_spans, etaEdges, dpix_eta[0],
            offset_ome[0], valid_ome_spans, omeEdges, dpix_ome[0])


@jit(nopython=True, nogil=True)
def paintGridThis_(quat,
    hkls, cc, sc, bMat, wavelength, vMat_s, rMat_e, 
    hkl_ix, thresholds, etaOmeMaps,
    offset_eta, valid_eta_spans, etaEdges, dpix_eta,
    offset_ome, valid_ome_spans, omeEdges, dpix_ome):
    rMat = np.empty((3,3))
    oangs = np.empty((2,3))
    quatToRotMat_(quat, rMat)
    total = hits = 0
    for i in range(hkls.shape[0]):
        oscillAnglesOfHKL_(hkls[i], cc, sc, rMat, bMat, wavelength, vMat_s, rMat_e, oangs)
        etaOmeMap = etaOmeMaps[hkl_ix[i]]
        threshold = thresholds[hkl_ix[i]]
        for j in range(2):
            oang = oangs[j]
            nh, nt = angle_is_hit_(oang, threshold, etaOmeMap,
                offset_eta, valid_eta_spans, etaEdges, dpix_eta,
                offset_ome, valid_ome_spans, omeEdges, dpix_ome)
            hits += nh
            total += nt
    return float(hits) / (float(total) + tiny64)

@guvectorize(
    '(f8[:], f8[:,:],f8[:],f8[:],f8[:,:],f8[:],f8[:,:],f8[:,:], i8[:],f8[:],f8[:,:,:],  f8[:],f8[:],f8[:],i8[:], f8[:],f8[:],f8[:],i8[:], f8[:])',
    '(n4),   (mm,n3),(),   (),   (n3,n3),(),   (n3,n3),(n3,n3), (mm), (m1), (m1,m2,m3), (),   (m4), (m5), (),    (),   (m7), (m8), ()  -> ()',
    nopython=True, target='cpu')
def paintGridThis_gufunc_(quat,
    hkls, cc, sc, bMat, wavelength, vMat_s, rMat_e, 
    hkl_ix, threshold, etaOmeMaps,
    offset_eta, valid_eta_spans, etaEdges, dpix_eta,
    offset_ome, valid_ome_spans, omeEdges, dpix_ome, out):
    out[0] = paintGridThis_(quat,
            hkls, cc[0], sc[0], bMat, wavelength[0], vMat_s, rMat_e,
            hkl_ix, threshold, etaOmeMaps,
            offset_eta[0], valid_eta_spans, etaEdges, dpix_eta[0],
            offset_ome[0], valid_ome_spans, omeEdges, dpix_ome[0])

def paintGridThis(quats,
    hkls, chi, bMat, wavelength, vInv_s, beamVec, etaVec,
    hkl_ix, threshold, etaOmeMaps,
    offset_eta, valid_eta_spans, etaEdges, dpix_eta,
    offset_ome, valid_ome_spans, omeEdges, dpix_ome, out=None):
    if quats.ndim == 0 or quats.shape[-1] != 4:
        raise RuntimeError('An array of quaternions is expected, but quats.shape[-1] != 4')
    vMat_s = vecMVToSymm(vInv_s)                # stretch tensor in SAMPLE frame
    rMat_e = etaFrameToRotMat(beamVec, etaVec)  # eta basis COB with beam antiparallel with Z
    cc = cos(chi)
    sc = sin(chi)
    return paintGridThis_gufunc_(quats,
                hkls, cc, sc, bMat, wavelength, vMat_s, rMat_e,
                hkl_ix, threshold, etaOmeMaps,
                offset_eta, valid_eta_spans, etaEdges, dpix_eta,
                offset_ome, valid_ome_spans, omeEdges, dpix_ome, out)
