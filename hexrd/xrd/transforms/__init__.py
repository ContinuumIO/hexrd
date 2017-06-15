"""This module produces the standard gateway to access the transform code

Transform code is central to hexrd. There may be different implementations
of the same functionality which should be interchangeable but could present
different characteristics.

- Python versions (py) being a reference implementation written in Python+Numpy.
- CAPI versions (capi) are implemented with the assistance of a C module, being
  a faster version that can be called in Python code directly.
- numba versions (numba) are implemented mostly using numba. Performance should
  be in the same order as the capi version and can be called inside other numba
  functions without forcing fallback to python mode in numba. Eventually, numba
  functions could be run on a GPU.

The module has the py, capi and numba versions implemented as submodules. The
main transforms module will provide the versions of the functions deemed as the
most appropriate for use by default. Note that the "capi" and "numba" submodules
are not necessarely complete, and functions will be implemented "as required".
Every function should have a "py" implementation.

The transforms module contains unittest that lie in the "tests" subdirectory.
Test coverage should include all functions in all different implementations.

The following functions are the official interface for transforms:

Category: xrd
------------------------------------------------------------------------
- anglesToGVec
- anglesToDVec
- makeGVector
- gvecToDetectorXY
- gvecToDetectorXYArray
- detectorXYToGVec
- detectorXYToGVecArray
- oscillAnglesOfHKLs

Category: rotations
------------------------------------------------------------------------
- makeDetectorRotMat
- makeOscillRotMat
- makeOscillRotMatArray
- makeRotMatOfExpMap
- makeRotMatOfQuat
- makeBinaryRotMat
- makeEtaFrameRotMat


Category: misc
------------------------------------------------------------------------
- arccosSafe
- angularDifference
- mapAngle
- columnNorm
- rowNorm
- unitRowVector (unitRowVectors)
- validateAngleRanges
- rotate_vecs_about_axis
- quat_distance
- homochoricOfQuat
"""

from __future__ import absolute_import

# transition code: mimic old transform module
# Right now some function only exist in some of the implementations and even
# the Python version is not complete. This serves as a map to locate where
# the implementations are.

# xrd 
from ..transforms_old  import anglesToGVec
from ..transforms_CAPI import anglesToDVec
from ..transforms_old  import (makeGVector,
                               gvecToDetectorXY)
from ..transforms_CAPI import gvecToDetectorXYArray
from ..transforms_old  import detectorXYToGvec
from ..transforms_CAPI import detectorXYToGvecArray
from ..transforms_old  import oscillAnglesOfHKLs

# rotations
from .xfpy    import makeOscillRotMat
from .xfpy    import makeOscillRotMatArray
from ..transforms_old  import makeRotMatOfExpMap
from ..transforms_CAPI import makeRotMatOfQuat
from ..transforms_old  import (makeBinaryRotMat,
                               makeEtaFrameRotMat)

# misc
from ..transforms_old  import (arccosSafe,
                               angularDifference,
                               mapAngle,
                               columnNorm,
                               rowNorm)
from ..transforms_CAPI import unitRowVector
from ..transforms_old  import (validateAngleRanges,
                               rotate_vecs_about_axis,
                               quat_distance)
from ..transforms_CAPI import homochoricOfQuat
                              
                            
