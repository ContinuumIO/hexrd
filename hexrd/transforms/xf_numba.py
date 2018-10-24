#! /usr/bin/env python
# =============================================================================
# Copyright (c) 2012, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by Joel Bernier <bernier2@llnl.gov> and others.
# LLNL-CODE-529294.
# All rights reserved.
#
# This file is part of HEXRD. For details on dowloading the source,
# see the file COPYING.
#
# Please also see the file LICENSE.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program (see file LICENSE); if not, write to
# the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
# Boston, MA 02111-1307 USA or visit <http://www.gnu.org/licenses/>.
# =============================================================================

# ??? do we want to set np.seterr(invalid='ignore') to avoid nan warnings?
from __future__ import absolute_import

import numpy as np
from numpy import float_ as npfloat
from numpy import int_ as npint

# from hexrd import constants as cnst
from .. import constants as cnst
from .transforms_definitions import xf_api, get_signature

import numba

# Use the following decorator instead of numba.jit for interface functions.
# This is so we can patch certain features.
def xfapi_jit(fn):
    out = numba.jit(fn)
    out.__signature__ = get_signature(fn)

    return out


def _beam_to_crystal(vecs, rmat_b=None, rmat_s=None, rmat_c=None):
    """
    Helper function to take vectors definced in the BEAM frame through LAB
    to either SAMPLE or CRYSTAL

    """
    vecs = np.atleast_2d(vecs)
    nvecs = len(vecs)
    if rmat_s is not None:
        rmat_s = np.squeeze(rmat_s)
        if rmat_s.ndim == 3:
            assert len(rmat_s) == nvecs, \
                "if specifying an array of rmat_s, dimensions must be " + \
                "(%d, 3, 3), not (%d, %d, %d)" \
                % tuple([nvecs] + list(rmat_s.shape))

    # take to lab frame (row order)
    if rmat_b is not None:
        vecs = np.dot(vecs, rmat_b.T)

    # to go to CRYSTAL in column vec order (hstacked gvec_l):
    #
    # gvec_l = np.dot(np.dot(rmat_c.T, np.dot(rmat_s.T, rmat_b)), gvec_b)
    #
    # rmat_s = np.dot(rchi, rome)
    #
    # --> in row vec order (vstacked gvec_l, C order):
    #
    # gvec_l = np.dot(gvec_b, np.dot(rmat_b.T, np.dot(rmat_s, rmat_c)))
    if rmat_s is not None:
        if rmat_s.ndim > 2:
            for i in range(nvecs):
                vecs[i] = np.dot(vecs[i], rmat_s[i])
        else:
            vecs = np.dot(vecs, rmat_s)
    if rmat_c is None:
        return vecs
    else:
        return np.dot(vecs, rmat_c)


@numba.njit
def _angles_to_gvec_helper(angs):
    """
    angs are vstacked [2*theta, eta, omega]

    This should be equivalent to the one-liner numpy version:
    out = np.vstack([[np.cos(0.5*angs[:, 0]) * np.cos(angs[:, 1])],
                     [np.cos(0.5*angs[:, 0]) * np.sin(angs[:, 1])],
                     [np.sin(0.5*angs[:, 0])]])

    although much faster,
    """
    out = np.empty((nvecs, 3))
    for i in range(n):
        ca0 = np.cos(0.5*angs[i, 0])
        sa0 = np.sin(0.5*angs[i, 0])
        ca1 = np.cos(angs[i, 1])
        sa1 = np.sin(angs[i, 1])
        out[i, 0] = ca0 * ca1
        out[i, 1] = ca0 * sa1
        out[i, 2] = sa0

    return out


@numba.njit
def _angles_to_dvec_helper(angs, out):
    """
    angs are vstacked [2*theta, eta, omega]

    dvec_b = np.vstack([[np.sin(angs[:, 0]) * np.cos(angs[:, 1])],
                        [np.sin(angs[:, 0]) * np.sin(angs[:, 1])],
                        [-np.cos(angs[:, 0])]])
    """
    n = angs.shape[0]
    for i in range(n):
        ca0 = np.cos(angs[i, 0])
        sa0 = np.sin(angs[i, 0])
        ca1 = np.cos(angs[i, 1])
        sa1 = np.sin(angs[i, 1])
        out[i, 0] = sa0 * ca1
        out[i, 1] = sa0 * sa1
        out[i, 2] = -ca0


@numba.njit
def _rmat_s_helper(chi, omes, out):
    """
    simple utility for calculating sample rotation matrices based on
    standard definition for HEDM
    """
    n = len(omes)
    cx = np.cos(chi)
    sx = np.sin(chi)
    for i in range(n):
        cw = np.cos(omes[i])
        sw = np.sin(omes[i])
        out[i, 0, 0] = cw
        out[i, 0, 1] = 0.
        out[i, 0, 2] = sw
        out[i, 1, 0] = sx * sw
        out[i, 1, 1] = cx
        out[i, 1, 2] = -sx * cw
        out[i, 2, 0] = -cx * sw
        out[i, 2, 1] = sx
        out[i, 2, 2] = cx * cw


@xf_api
def angles_to_gvec(angs,
                   beam_vec=None, eta_vec=None,
                   chi=None, rmat_c=None):
    """Note about this implementation:
    This used to take rmat_b instead of the pair beam_vec, eta_vec. So it may require
    some checking.
    """
    # apply defaults to beam_vec and eta_vec:
    beam_vec = beam_vec if beam_vec is not None else cnst.beam_vec
    eta_vec = eta_vec if eta_vec is not None else cnst.beam_vec

    angs = np.atleast_2d(angs)
    nvecs, dim = angs.shape

    # make vectors in beam frame
    gvec_b = _angles_to_gvec_helper(angs)



    # for NUMBA case, need chi as a flaot
    if chi is None:
        chi = 0.

    # calculate rmat_s
    if dim > 2:
        rmat_s = np.empty((nvecs, 3, 3))
        _rmat_s_helper(chi, angs[:, 2], rmat_s)
    else:  # no omegas given
        rmat_s = np.empty((1, 3, 3))
        _rmat_s_helper(chi, [0., ], rmat_s)

    rmat_b = make_beam_rmat(beam_vec, eta_vec)
    out = _beam_to_crystal(gvec_b,
                           rmat_b=rmat_b, rmat_s=rmat_s, rmat_c=rmat_c)
    return out


def angles_to_dvec(angs, rmat_b=None, chi=None, rmat_c=None):
    """
    Takes triplets of angles in the beam frame (2*theta, eta, omega)
    to components of unit diffraction vectors in the LAB frame.  If the
    omega values are not trivial (i.e. angs[:, 2] = 0.), then the
    components are in the SAMPLE frame.  If the crystal rmat is specified
    and is not the identity, then the components are in the CRYSTAL frame.
    """
    angs = np.atleast_2d(angs)
    nvecs, dim = angs.shape

    # make vectors in beam frame
    dvec_b = np.empty((nvecs, 3))
    _angles_to_dvec_helper(angs, dvec_b)

    # for NUMBA case, need chi as a flaot
    if chi is None:
        chi = 0.

    # calcuate rmat_s
    if dim > 2:
        rmat_s = np.empty((nvecs, 3, 3))
        _rmat_s_helper(chi, angs[:, 2], rmat_s)
    else:  # no omegas given
        rmat_s = np.empty((1, 3, 3))
        _rmat_s_helper(chi, [0., ], rmat_s)
    return _beam_to_crystal(
        dvec_b, rmat_b=rmat_b, rmat_s=rmat_s, rmat_c=rmat_c
        )


@numba.njit
def _row_norm(a, b):
    n, dim = a.shape
    for i in range(n):
        nrm = 0.0
        for j in range(dim):
            nrm += a[i, j]*a[i, j]
        b[i] = np.sqrt(nrm)


@numba.njit
def _unit_vector_single(a, b):
    n = len(a)
    nrm = 0.0
    for i in range(n):
        nrm += a[i]*a[i]
    nrm = np.sqrt(nrm)
    # prevent divide by zero
    if nrm > cnst.epsf:
        for i in range(n):
            b[i] = a[i] / nrm
    else:
        for i in range(n):
            b[i] = a[i]


@numba.njit
def _unit_vector_multi(a, b):
    n, dim = a.shape
    for i in range(n):
        nrm = 0.0
        for j in range(dim):
            nrm += a[i, j]*a[i, j]
        nrm = np.sqrt(nrm)
        # prevent divide by zero
        if nrm > cnst.epsf:
            for i in range(n):
                b[i, j] = a[i, j] / nrm
        else:
            for i in range(n):
                b[i, j] = a[i, j]


def row_norm(a):
    """
    retrun row-wise norms for a list of vectors
    """
    a = np.atleast_2d(a)
    result = np.empty(len(a))
    _row_norm(a, result)
    return result


def unit_vector(a):
    """
    normalize array of column vectors (hstacked, axis = 0)
    """
    result = np.empty_like(a)
    if a.ndim == 1:
        _unit_vector_single(a, result)
    elif a.ndim == 2:
        _unit_vector_multi(a, result)
    else:
        raise ValueError(
            "incorrect arg shape; must be 1-d or 2-d, yours is %d-d"
            % (a.ndim)
        )
    return result


@numba.njit
def _make_rmat_of_expmap(x, z):
    """
    TODO:

    Test effectiveness of two options:

    1) avoid conditional inside for loop and use np.divide to return NaN
       for the phi = 0 cases, and deal with it later; or
    2) catch phi = 0 cases inside the loop and just return squeezed answer
    """
    n = x.shape[0]
    for i in range(n):
        phi = np.sqrt(x[i, 0]*x[i, 0] + x[i, 1]*x[i, 1] + x[i, 2]*x[i, 2])
        if phi <= cnst.sqrt_epsf:
            z[i, 0, 0] = 1.
            z[i, 0, 1] = 0.
            z[i, 0, 2] = 0.
            z[i, 1, 0] = 0.
            z[i, 1, 1] = 1.
            z[i, 1, 2] = 0.
            z[i, 2, 0] = 0.
            z[i, 2, 1] = 0.
            z[i, 2, 2] = 1.
        else:
            f1 = np.sin(phi)/phi
            f2 = (1. - np.cos(phi)) / (phi*phi)
            z[i, 0, 0] = 1. - f2*(x[i, 2]*x[i, 2] + x[i, 1]*x[i, 1])
            z[i, 0, 1] = f2*x[i, 1]*x[i, 0] - f1*x[i, 2]
            z[i, 0, 2] = f1*x[i, 1] + f2*x[i, 2]*x[i, 0]
            z[i, 1, 0] = f1*x[i, 2] + f2*x[i, 1]*x[i, 0]
            z[i, 1, 1] = 1. - f2*(x[i, 2]*x[i, 2] + x[i, 0]*x[i, 0])
            z[i, 1, 2] = f2*x[i, 2]*x[i, 1] - f1*x[i, 0]
            z[i, 2, 0] = f2*x[i, 2]*x[i, 0] - f1*x[i, 1]
            z[i, 2, 1] = f1*x[i, 0] + f2*x[i, 2]*x[i, 1]
            z[i, 2, 2] = 1. - f2*(x[i, 1]*x[i, 1] + x[i, 0]*x[i, 0])


"""
if the help above was set up to return nans...

def make_rmat_of_expmap(exp_map):
    exp_map = np.atleast_2d(exp_map)
    rmats = np.empty((len(exp_map), 3, 3))
    _make_rmat_of_expmap(exp_map, rmats)
    chk = np.isnan(rmats)
    if np.any(chk):
        rmats[chk] = np.tile(
            [1., 0., 0., 0., 1., 0., 0., 0., 1.], np.sum(chk)/9
            )
    return rmats
"""


def make_rmat_of_expmap(exp_map):
    exp_map = np.atleast_2d(exp_map)
    rmats = np.empty((len(exp_map), 3, 3))
    _make_rmat_of_expmap(exp_map, rmats)
    return np.squeeze(rmats)


@xf_api
@xfapi_jit
def make_beam_rmat(bvec_l, evec_l):
    # bhat_l and ehat_l CANNOT have 0 magnitude!
    # must catch this case as well as colinear bhat_l/ehat_l elsewhere...

    bvec_mag = np.sqrt(bvec_l[0]**2 + bvec_l[1]**2 + bvec_l[2]**2)

    if bvec_mag < cnst.sqrt_epsf:
        #can numba raise?
        # raise RuntimeError("beam_vec MUST NOT be ZERO!")
        pass

    # assign Ze as -bhat_l
    for i in range(3):
        out[i, 2] = -bvec_l[i] / bvec_mag
    Ze0 = -bvec_l[0] / bvec_mag
    Ze1 = -bvec_l[1] / bvec_mag
    Ze2 = -bvec_l[2] / bvec_mag    

    # find Ye as Ze ^ ehat_l
    Ye0 = Ze1*evec_l[2] - evec_l[1]*Ze2
    Ye1 = Ze2*evec_l[0] - evec_l[2]*Ze0
    Ye2 = Ze0*evec_l[1] - evec_l[0]*Ze1

    Ye_mag = np.sqrt(Ye0**2 + Ye1**2 + Ye2**2)
    if Ye_mag < cnst.sqrt_epsf:
        # raise RuntimeError("beam_vec and eta_vec MUST NOT be colinear!")
        pass

    out = np.empty((3,3)) # numba can now allocate
    Ye0 /= Ye_mag
    Ye1 /= Ye_mag
    Ye2 /= Ye_mag

    # find Xe as Ye ^ Ze
    Xe0 = Ye1*Ze2 - Ze1*Ye2
    Xe1 = Ye2*Ze0 - Ze2*Ye0
    Xe2 = Ye0*Ze1 - Ze0*Ye1


    out[0, 1] = Ye0 / Ye_mag
    out[1, 1] = Ye1 / Ye_mag
    out[2, 1] = Ye2 / Ye_mag

    out[0, 0] = out[1, 1]*out[2, 2] - out[1, 2]*out[2, 1]
    out[1, 0] = out[2, 1]*out[0, 2] - out[2, 2]*out[0, 1]
    out[2, 0] = out[0, 1]*out[1, 2] - out[0, 2]*out[1, 1]

    return out


