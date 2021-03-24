import numpy as np
from spherical_functions import LM_total_size
from .. import ModesTimeSeries
from .. import Inertial


class AsymptoticBondiData:
    """Class to store asymptotic Bondi data

    This class stores time data, along with the corresponding values of psi0 through psi4 and sigma.
    For simplicity, the data are stored as one contiguous array.  That is, *all* values are stored
    at all times, even if they are zero, and all Modes objects are stored with ell_min=0, even when
    their spins are not zero.

    The single contiguous array is then viewed as 6 separate ModesTimeSeries objects, which enables
    them to track their spin weights, and provides various convenient methods like `eth` and
    `ethbar`; `dot` and `ddot` for time-derivatives; `int` and `iint` for time-integrations; `norm`
    to take the norm of a function over the sphere; `bar` for conjugation of the functions (which is
    different from just conjugating the mode weights); etc.  It also handles algebra correctly --
    particularly addition (which is disallowed when the spin weights differ) and multiplication
    (which can be delicate with regards to the resulting ell values).

    This may lead to some headaches when the user tries to do things that are disabled by Modes
    objects.  The goal is to create headaches if and only if the user is trying to do things that
    really should never be done (like conjugating mode weights, rather than the underlying function;
    adding modes with different spin weights; etc.).  Please open issues for any situations that
    don't meet this standard.

    This class also provides various convenience methods for computing things like the mass aspect,
    the Bondi four-momentum, the Bianchi identities, etc.

    """

    def __init__(self, time, ell_max, multiplication_truncator=sum, frameType=Inertial):
        """Create new storage for asymptotic Bondi data

        Parameters
        ==========
        time: int or array_like
            Times at which the data will be stored.  If this is an int, an empty array of that size
            will be created.  Otherwise, this must be a 1-dimensional array of floats.
        ell_max: int
            Maximum ell value to be stored
        multiplication_truncator: callable [defaults to `sum`, even though `max` is nicer]
            Function to be used by default when multiplying Modes objects together.  See the
            documentation for spherical_functions.Modes.multiply for more details.  The default
            behavior with `sum` is the most correct one -- keeping all ell values that result -- but
            also the most wasteful, and very likely to be overkill.  The user should probably always
            use `max`.  (Unfortunately, this must remain an opt-in choice, to ensure that the user
            is aware of the situation.)

        """
        import functools

        if np.ndim(time) == 0:
            # Assume this is just the size of the time array; construct an empty array
            time = np.empty((time,), dtype=float)
        elif np.ndim(time) > 1:
            raise ValueError(f"Input `time` parameter must be an integer or a 1-d array; it has shape {time.shape}")
        if time.dtype != float:
            raise ValueError(f"Input `time` parameter must have dtype float; it has dtype {time.dtype}")
        ModesTS = functools.partial(ModesTimeSeries, ell_max=ell_max, multiplication_truncator=multiplication_truncator)
        shape = [6, time.size, LM_total_size(0, ell_max)]
        self.frame = np.array([])
        self.frameType = frameType
        self._time = time.copy()
        self._raw_data = np.zeros(shape, dtype=complex)
        self._psi0 = ModesTS(self._raw_data[0], self._time, spin_weight=2)
        self._psi1 = ModesTS(self._raw_data[1], self._time, spin_weight=1)
        self._psi2 = ModesTS(self._raw_data[2], self._time, spin_weight=0)
        self._psi3 = ModesTS(self._raw_data[3], self._time, spin_weight=-1)
        self._psi4 = ModesTS(self._raw_data[4], self._time, spin_weight=-2)
        self._sigma = ModesTS(self._raw_data[5], self._time, spin_weight=2)

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, new_time):
        self._time[:] = new_time
        return self._time

    u = time

    t = time

    @property
    def n_times(self):
        return self.time.size

    @property
    def n_modes(self):
        return self._raw_data.shape[-1]

    @property
    def ell_min(self):
        return self._psi2.ell_min

    @property
    def ell_max(self):
        return self._psi2.ell_max

    @property
    def LM(self):
        return self.psi2.LM

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigmaprm):
        self._sigma[:] = sigmaprm
        return self.sigma

    @property
    def psi4(self):
        return self._psi4

    @psi4.setter
    def psi4(self, psi4prm):
        self._psi4[:] = psi4prm
        return self.psi4

    @property
    def psi3(self):
        return self._psi3

    @psi3.setter
    def psi3(self, psi3prm):
        self._psi3[:] = psi3prm
        return self.psi3

    @property
    def psi2(self):
        return self._psi2

    @psi2.setter
    def psi2(self, psi2prm):
        self._psi2[:] = psi2prm
        return self.psi2

    @property
    def psi1(self):
        return self._psi1

    @psi1.setter
    def psi1(self, psi1prm):
        self._psi1[:] = psi1prm
        return self.psi1

    @property
    def psi0(self):
        return self._psi0

    @psi0.setter
    def psi0(self, psi0prm):
        self._psi0[:] = psi0prm
        return self.psi0

    def copy(self):
        import copy

        new_abd = type(self)(self.t, self.ell_max)
        state = copy.deepcopy(self.__dict__)
        new_abd.__dict__.update(state)
        return new_abd

    def interpolate(self, new_times):
        new_abd = type(self)(new_times, self.ell_max)
        new_abd.frameType = self.frameType
        # interpolate waveform data
        new_abd.sigma = self.sigma.interpolate(new_times)
        new_abd.psi4 = self.psi4.interpolate(new_times)
        new_abd.psi3 = self.psi3.interpolate(new_times)
        new_abd.psi2 = self.psi2.interpolate(new_times)
        new_abd.psi1 = self.psi1.interpolate(new_times)
        new_abd.psi0 = self.psi0.interpolate(new_times)
        # interpolate frame data if necessary
        if self.frame.shape[0] == self.n_times:
            import quaternion
            new_abd.frame = quaternion.squad(self.frame, self.t, new_times)
        return new_abd

    from .from_initial_values import from_initial_values

    from .constraints import (
        bondi_constraints,
        bondi_violations,
        bondi_violation_norms,
        bianchi_0,
        bianchi_1,
        bianchi_2,
        constraint_3,
        constraint_4,
        constraint_mass_aspect,
    )

    from .transformations import transform
    from .bms_charges import (
        mass_aspect,
        bondi_rest_mass,
        bondi_four_momentum,
        bondi_angular_momentum,
        bondi_dimensionless_spin,
        bondi_boost_charge,
        bondi_CoM_charge,
        supermomentum,
    )
