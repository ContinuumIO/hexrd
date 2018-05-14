"""Transforms module.

Contains different implementations based on Python+Numpy, numba and
a supporting C module. All three should adhere to the same interface,
but performance will vary.

Use the functions under this module scope to use the preferred versions,
import the specific submodule if you want to use a specific version.
"""
from __future__ import absolute_import

# While this may not be ideal, use an import line per API function.

from .xf_numpy import angles_to_gvec
from .xf_numpy import angles_to_dvec
from .xf_numpy import gvec_to_xy
from .xf_numpy import xy_to_gvec
from .xf_numpy import solve_omega

# utility functions
from .xf_numpy import angular_difference
from .xf_numpy import map_angle
from .xf_numpy import row_norm
from .xf_numpy import unit_vector
from .xf_numpy import make_sample_rmat
from .xf_numpy import make_rmat_of_expmap
from .xf_numpy import make_binary_rmat
from .xf_numpy import make_beam_rmat
from .xf_numpy import angles_in_range
from .xf_numpy import validate_angle_ranges
from .xf_numpy import rotate_vecs_about_axis
from .xf_numpy import quat_product_matrix
from .xf_numpy import quat_distance
