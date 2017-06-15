#
# CAPI version of transforms related function
#
# This file provides alternative implementations to xfpy/transforms_py.py using
# a C module.
################################################################################

import numpy as np

from hexrd.xrd import _transforms_CAPI as _xfcapi

epsf = np.finfo(float).eps
bVec_ref = np.array([[0., 0., 1.]], order='C').T
eta_ref = np.array([[1., 0., 0.]], order='C').T

I3 = np.eye(3)

def anglesToGVec(angs, bHat_l=bVec_ref, eHat_l=eta_ref, chi=0., rMat_c=I3):
    """
    from 'eta' frame out to lab (with handy kwargs to go to crystal or sample)

    * setting omega to zero in ang imput and omitting rMat_c leaves 
      in the lab frame in accordance with beam frame specs.
    """
    angs = np.ascontiguousarray( np.atleast_2d(angs) )
    bHat_l = np.ascontiguousarray( bHat_l.flatten() )
    eHat_l = np.ascontiguousarray( eHat_l.flatten() )
    rMat_c = np.ascontiguousarray( rMat_c )
    chi = float(chi)

    return _transforms_CAPI.anglesToGVec(angs, 
                                         bHat_l, eHat_l,
                                         chi, rMat_c)


def anglesToDVec(angs, bHat_l=bVec_ref, eHat_l=eta_ref, chi=0., rMat_c=I3):
    """
    from 'eta' frame out to lab (with handy kwargs to go to crystal or sample)

    * setting omega to zero in ang imput and omitting rMat_c leaves 
      in the lab frame in accordance with beam frame specs.
    """
    angs = np.ascontiguousarray( np.atleast_2d(angs) )
    bHat_l = np.ascontiguousarray( bHat_l.flatten() )
    eHat_l = np.ascontiguousarray( eHat_l.flatten() )
    rMat_c = np.ascontiguousarray( rMat_c )
    chi = float(chi)
    return _transforms_CAPI.anglesToDVec(angs, 
                                         bHat_l, eHat_l,
                                         chi, rMat_c)
