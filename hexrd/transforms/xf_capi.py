# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

from hexrd import constants as cnst
from .. import _transforms_CAPI
from .transforms_definitions import xf_api

@xf_api
def angles_to_gvec(
        angs, 
        beam_vec=cnst.beam_vec, eta_vec=cnst.eta_vec,
        chi=0., rmat_c=cnst.identity_3x3):

    angs = np.ascontiguousarray( np.atleast_2d(angs) )
    beam_vec = np.ascontiguousarray( beam_vec.flatten() )
    eta_vec = np.ascontiguousarray( eta_vec.flatten() )
    rmat_c = np.eye(3) if rmat_c is None else np.ascontiguousarray( rmat_c )
    chi = 0.0 if chi is None else float(chi)
    return _transforms_CAPI.anglesToGVec(angs, 
                                         bHat_l, eHat_l,
                                         chi, rMat_c)


@xf_api
def angles_to_dvec(
        angs, 
        beam_vec=cnst.beam_vec, eta_vec=cnst.eta_vec,
        chi=0., rmat_c=None):
    # TODO: Improve capi to avoid multiplications when rmat_c is None
    angs = np.ascontiguousarray( np.atleast_2d(angs) )
    beam_vec = np.ascontiguousarray( beam_vec.flatten() )
    eta_vec = np.ascontiguousarray( eta_vec.flatten() )
    rmat_c = rmat_c if rmat_c is not None else np.ascontiguousarray( np.eye(3) )
    chi = float(chi)

    return _transforms_CAPI.anglesToDVec(angs,
                                         bHat_l, eHat_l,
                                         chi, rMat_c)

#@xf_api
def makeGVector(hkl, bMat):
    assert hkl.shape[0] == 3, 'hkl input must be (3, n)'
    return unitVector(np.dot(bMat, hkl))


@xf_api
def gvec_to_xy(gvec_c,
               rmat_d, rmat_s, rmat_c,
               tvec_d, tvec_s, tvec_c,
               beam_vec=cnst.beam_vec,
               vmat_inv=None,
               bmat=None):
    gvec_c  = np.ascontiguousarray( np.atleast_2d( gvec_c ) )
    rmat_s  = np.ascontiguousarray( rmat_s)
    tvec_d  = np.ascontiguousarray( tvec_d.flatten()  )
    tvec_s  = np.ascontiguousarray( tvec_s.flatten()  )
    tvec_c  = np.ascontiguousarray( tvec_c.flatten()  )
    beam_vec = np.ascontiguousarray( beam_vec.flatten() )

    # depending on the number of dimensions of rmat_s use either the array version
    # or the "scalar" (over rmat_s) version. Note that rmat_s is either a 3x3 matrix
    # (ndim 2) or an nx3x4 array of matrices (ndim 3) 
    if rmat_s.ndim > 2:
        return _transforms_CAPI.gvecToDetectorXYArray(gvec_c,
                                                      rmat_d, tvec_s, tvec_c,
                                                      beam_vec)
    else:
        return _transforms_CAPI.gvecToDetectorXY(gvec_c,
                                                 rmat_d, rmat_s, rmat_c,
                                                 tvec_d, tvec_s, tvec_c,
                                                 beam_vec)


@xf_api
def xy_to_gvec(xy_d,
               rmat_d, rmat_s,
               tvec_d, tvec_s, tvec_c,
               rmat_b=None,
               distortion=None,
               output_ref=False):
    # This was "detectorXYToGvec" previously. There is a change in the interface where
    # 'beamVec'
    # beamVec? ->
    # etaVec?

    # TODO: Fix this with the new interface
    xy_det  = np.ascontiguousarray( np.atleast_2d(xy_det) )
    tVec_d  = np.ascontiguousarray( tVec_d.flatten() )
    tVec_s  = np.ascontiguousarray( tVec_s.flatten() )
    tVec_c  = np.ascontiguousarray( tVec_c.flatten() )
    beamVec = np.ascontiguousarray( beamVec.flatten() )
    etaVec  = np.ascontiguousarray( etaVec.flatten() )
    return _transforms_CAPI.detectorXYToGvec(xy_det,
                                             rMat_d, rMat_s,
                                             tVec_d, tVec_s, tVec_c,
                                             beamVec, etaVec)



#@xf_api
def oscillAnglesOfHKLs(hkls, chi, rMat_c, bMat, wavelength,
                       vInv=None, beamVec=cnst.beam_vec, etaVec=cnst.eta_vec):
    # this was oscillAnglesOfHKLs
    hkls = np.array(hkls, dtype=float, order='C')
    if vInv is None:
        vInv = np.ascontiguousarray(vInv_ref.flatten())
    else:
        vInv = np.ascontiguousarray(vInv.flatten())
    beamVec = np.ascontiguousarray(beamVec.flatten())
    etaVec  = np.ascontiguousarray(etaVec.flatten())
    bMat = np.ascontiguousarray(bMat)
    return _transforms_CAPI.oscillAnglesOfHKLs(
        hkls, chi, rMat_c, bMat, wavelength, vInv, beamVec, etaVec
        )


#@xf_api
def unitRowVector(vecIn):
    vecIn = np.ascontiguousarray(vecIn)
    if vecIn.ndim == 1:
        return _transforms_CAPI.unitRowVector(vecIn)
    elif vecIn.ndim == 2:
        return _transforms_CAPI.unitRowVectors(vecIn)
    else:
        assert vecIn.ndim in [1,2], "incorrect arg shape; must be 1-d or 2-d, yours is %d-d" % (a.ndim)


#@xf_api
def makeDetectorRotMat(tiltAngles):
    arg = np.ascontiguousarray(np.r_[tiltAngles].flatten())
    return _transforms_CAPI.makeDetectorRotMat(arg)


#@xf_api
def makeOscillRotMat(oscillAngles):
    arg = np.ascontiguousarray(np.r_[oscillAngles].flatten())
    return _transforms_CAPI.makeOscillRotMat(arg)


#@xf_api
def makeOscillRotMatArray(chi, omeArray):
    arg = np.ascontiguousarray(omeArray)
    return _transforms_CAPI.makeOscillRotMatArray(chi, arg)


#@xf_api
def makeRotMatOfExpMap(expMap):
    arg = np.ascontiguousarray(expMap.flatten())
    return _transforms_CAPI.makeRotMatOfExpMap(arg)


#@xf_api
def makeRotMatOfQuat(quats):
    arg = np.ascontiguousarray(quats)
    return _transforms_CAPI.makeRotMatOfQuat(arg)


#@xf_api
def makeBinaryRotMat(axis):
    arg = np.ascontiguousarray(axis.flatten())
    return _transforms_CAPI.makeBinaryRotMat(arg)


#@xf_api
def makeEtaFrameRotMat(bHat_l, eHat_l):
    arg1 = np.ascontiguousarray(bHat_l.flatten())
    arg2 = np.ascontiguousarray(eHat_l.flatten())
    return _transforms_CAPI.makeEtaFrameRotMat(arg1, arg2)


#@xf_api
def validateAngleRanges(angList, angMin, angMax, ccw=True):
    angList = angList.astype(np.double, order="C")
    angMin = angMin.astype(np.double, order="C")
    angMax = angMax.astype(np.double, order="C")
    return _transforms_CAPI.validateAngleRanges(angList,angMin,angMax,ccw)


#@xf_api
def rotate_vecs_about_axis(angle, axis, vecs):
    return _transforms_CAPI.rotate_vecs_about_axis(angle, axis, vecs)


#@xf_api
def quat_distance(q1, q2, qsym):
    q1 = np.ascontiguousarray(q1.flatten())
    q2 = np.ascontiguousarray(q2.flatten())
    return _transforms_CAPI.quat_distance(q1, q2, qsym)


#@xf_api
def homochoricOfQuat(quats):
    q = np.ascontiguousarray(quats.T)
    return _transforms_CAPI.homochoricOfQuat(q)
