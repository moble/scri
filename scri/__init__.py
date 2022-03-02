# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>
"""Module for operating on gravitational waveforms in various forms

Classes
-------
WaveformBase : Base class
    This is probably not needed directly; it is just used for inheritance by other objects.
WaveformModes: Complex spin-weighted spherical-harmonic modes
    The modes must include all `m` values for a range of `ell` values.  This is the "classic" version of a WaveformBase
    object we might normally think of.
WaveformGrid: Complex quantity evaluated along world lines of grid points on the sphere
    To perform translations or boosts, we need to transform to physical space, along a series of selected world lines
    distributed evenly across the sphere.  These values may need to be interpolated to new time values, and they will
    presumably need to be transformed back to `WaveformModes`.
WaveformInDetector: Real quantities as observed in an inertial detector
    Detectors only measure one polarization, so they deal with real quantities.  Also, data is measured in evenly
    spaced time steps.  This object can be created from a `WaveformModes` object.
WaveformInDetectorFT: (Complex) Fourier transform of a `WaveformInDetector`
    This contains only the positive-frequency values since the transformed data is real.

"""

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:  # pragma: no cover
    import importlib_metadata

import sys
import functools
import numba

__version__ = importlib_metadata.version(__name__)

jit = functools.partial(numba.njit, cache=True)
jitclass = numba.experimental.jitclass


def version_info():
    """Show version information about this module and various dependencies"""
    import spherical_functions
    import quaternion
    import scipy
    import numba
    import numpy

    versions = "\n".join(
        [
            f"scri.__version__ = {__version__}",
            f"spherical_functions.__version__ = {spherical_functions.__version__}",
            f"quaternion.__version__ = {quaternion.__version__}",
            f"scipy.__version__ = {scipy.__version__}",
            f"numba.__version__ = {numba.__version__}",
            f"numpy.__version__ = {numpy.__version__}",
        ]
    )
    return versions


# The speed of light is, of course, defined to be exact:
speed_of_light = 299792458.0  # m/s

# The value of the solar mass parameter G*M_sun is known to higher accuracy than either of its factors.  The value
# here is taken from the publication "2015 Selected Astronomical Constants", which can be found at
# <http://asa.usno.navy.mil/SecK/Constants.html>.  This is (one year more current than, but numerically the same as)
# the source cited by the Particle Data Group.  It is given as 1.32712440041e20 m^3/s^2 in the TDB (Barycentric
# Dynamical Time) time scale, which seems to be the more relevant one, and looks like the more standard one for LIGO.
# Dividing by the speed of light squared, we get the mass of the sun in meters; dividing again, we get the mass of
# the sun in seconds:
m_sun_in_meters = 1476.62503851  # m
m_sun_in_seconds = 4.92549094916e-06  # s

# By "IAU 2012 Resolution B2", the astronomical unit is defined to be exactly 1 au = 149597870700 m.  The parsec
# is, in turn, defined as "The distance at which 1 au subtends 1 arc sec: 1 au divided by pi/648000."  Thus, the
# future-proof value of the parsec in meters is
parsec_in_meters = 3.0856775814913672789139379577965e16  # m

FrameType = [UnknownFrameType, Inertial, Coprecessing, Coorbital, Corotating] = range(5)
FrameNames = ["UnknownFrameType", "Inertial", "Coprecessing", "Coorbital", "Corotating"]

DataType = [UnknownDataType, psi0, psi1, psi2, psi3, psi4, sigma, h, hdot, news, psin] = range(11)
DataNames = ["UnknownDataType", "Psi0", "Psi1", "Psi2", "Psi3", "Psi4", "sigma", "h", "hdot", "news", "psin"]
SpinWeights = [sys.maxsize, 2, 1, 0, -1, -2, 2, -2, -2, -2, sys.maxsize]
ConformalWeights = [sys.maxsize, 2, 1, 0, -1, -2, 1, 0, -1, -1, -3]
RScaling = [sys.maxsize, 5, 4, 3, 2, 1, 2, 1, 1, 1, 0]
MScaling = [sys.maxsize, 2, 2, 2, 2, 2, 0, 0, 1, 1, 2]
DataNamesLaTeX = [
    r"\mathrm{unknown data type}",
    r"\psi_0",
    r"\psi_1",
    r"\psi_2",
    r"\psi_3",
    r"\psi_4",
    r"\sigma",
    r"h",
    r"\dot{h}",
    r"\mathrm{n}",
    r"\psi_n",
]
# It might also be worth noting that:
# - the radius `r` has spin weight 0 and boost weight -1
# - a time-derivative `d/du` has spin weight 0 and boost weight -1
# - \eth has spin weight +1; \bar{\eth} has spin weight -1
# - \eth in the GHP formalism has boost weight 0
# - \eth in the original NP formalism has undefined boost weight
# - It seems like `M` should have boost weight 1, but I'll have to think about the implications

# Set up the WaveformModes object, by adding some methods
from .waveform_modes import WaveformModes
from .mode_calculations import (
    LdtVector,
    LVector,
    LLComparisonMatrix,
    LLMatrix,
    LLDominantEigenvector,
    angular_velocity,
    corotating_frame,
    inner_product,
)

from .flux import (
    energy_flux,
    momentum_flux,
    angular_momentum_flux,
    poincare_fluxes,
    boost_flux
)


WaveformModes.LdtVector = LdtVector
WaveformModes.LVector = LVector
WaveformModes.LLComparisonMatrix = LLComparisonMatrix
WaveformModes.LLMatrix = LLMatrix
WaveformModes.LLDominantEigenvector = LLDominantEigenvector
WaveformModes.angular_velocity = angular_velocity
from .rotations import (
    rotate_decomposition_basis,
    rotate_physical_system,
    to_coprecessing_frame,
    to_corotating_frame,
    to_inertial_frame,
    align_decomposition_frame_to_modes,
)

WaveformModes.rotate_decomposition_basis = rotate_decomposition_basis
WaveformModes.rotate_physical_system = rotate_physical_system
WaveformModes.to_coprecessing_frame = to_coprecessing_frame
WaveformModes.to_corotating_frame = to_corotating_frame
WaveformModes.to_inertial_frame = to_inertial_frame
WaveformModes.align_decomposition_frame_to_modes = align_decomposition_frame_to_modes
WaveformModes.energy_flux = energy_flux
WaveformModes.momentum_flux = momentum_flux
WaveformModes.angular_momentum_flux = angular_momentum_flux
WaveformModes.boost_flux = boost_flux
WaveformModes.poincare_fluxes = poincare_fluxes


from .waveform_grid import WaveformGrid

# from .waveform_in_detector import WaveformInDetector
from .extrapolation import extrapolate

from .modes_time_series import ModesTimeSeries
from .asymptotic_bondi_data import AsymptoticBondiData

from . import sample_waveforms, SpEC, LVC, utilities
from .SpEC import rpxmb

__all__ = [
    "WaveformModes",
    "WaveformGrid",
    "WaveformInDetector",
    "FrameType",
    "UnknownFrameType",
    "Inertial",
    "Coprecessing",
    "Coorbital",
    "Corotating",
    "FrameNames",
    "DataType",
    "UnknownDataType",
    "psi0",
    "psi1",
    "psi2",
    "psi3",
    "psi4",
    "sigma",
    "h",
    "hdot",
    "news",
    "psin",
    "DataNames",
    "DataNamesLaTeX",
    "SpinWeights",
    "ConformalWeights",
    "RScaling",
    "MScaling",
    "speed_of_light",
    "m_sun_in_meters",
    "m_sun_in_seconds",
    "parsec_in_meters",
]
