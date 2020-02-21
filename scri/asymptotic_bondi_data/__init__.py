import math
import numpy as np
from spherical_functions import LM_total_size
from .. import ModesTimeSeries


class AsymptoticBondiData(object):
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
    def __init__(self, time, ell_max, multiplication_truncator=sum):
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
        if np.ndim(time) == 0:
            # Assume this is just the size of the time array; construct an empty array
            time = np.empty((time,), dtype=float)
        elif np.ndim(time) > 1:
            raise ValueError(f"Input `time` parameter must be an integer or a 1-d array; it has shape {time.shape}")
        if time.dtype != float:
            raise ValueError(f"Input `time` parameter must have dtype float; it has dtype {time.dtype}")
        shape = [6, time.size, LM_total_size(0, ell_max)]
        self._time = time.copy()
        self._raw_data = np.zeros(shape, dtype=complex)
        self._psi0 = ModesTimeSeries(self._raw_data[0], self._time, spin_weight=2,
                                     ell_max=ell_max, multiplication_truncator=multiplication_truncator)
        self._psi1 = ModesTimeSeries(self._raw_data[1], self._time, spin_weight=1,
                                     ell_max=ell_max, multiplication_truncator=multiplication_truncator)
        self._psi2 = ModesTimeSeries(self._raw_data[2], self._time, spin_weight=0,
                                     ell_max=ell_max, multiplication_truncator=multiplication_truncator)
        self._psi3 = ModesTimeSeries(self._raw_data[3], self._time, spin_weight=-1,
                                     ell_max=ell_max, multiplication_truncator=multiplication_truncator)
        self._psi4 = ModesTimeSeries(self._raw_data[4], self._time, spin_weight=-2,
                                     ell_max=ell_max, multiplication_truncator=multiplication_truncator)
        self._sigma = ModesTimeSeries(self._raw_data[5], self._time, spin_weight=2,
                                      ell_max=ell_max, multiplication_truncator=multiplication_truncator)

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

    from .from_initial_values import from_initial_values

    def mass_aspect(self, truncate_ell=max):
        """Compute the Bondi mass aspect of the AsymptoticBondiData

        The Bondi mass aspect is given by

            \Psi = \psi_2 + \eth \eth \bar{\sigma} + \sigma * \dot{\bar{\sigma}}

        Note that the last term is a product between two fields.  If, for example, these both have
        ell_max=8, then their full product would have ell_max=16, meaning that we would go from
        tracking 77 modes to 289.  This shows that deciding how to truncate the output ell is
        important, which is why this function has the extra argument that it does.

        Parameters
        ==========
        truncate_ell: int, or callable [defaults to `max`]
            Determines how the ell_max value of the output is determined.  If an integer is passed,
            each term in the output is truncated to have at most that ell_max.  (In particular,
            terms that will not be used in the output are simply not computed, without incurring any
            errors due to aliasing.)  If a callable is passed, it is passed on to the
            spherical_functions.Modes.multiply method.  See that function's docstring for details.
            The default behavior will result in the output having ell_max equal to the largest of
            any of the individual Modes objects in the equation for \Psi above -- but not the
            product.

        """
        if callable(truncate_ell):
            return self.psi2 + self.sigma.bar.eth.eth + self.sigma.multiply(self.sigma.bar.dot, truncator=truncate_ell)
        elif truncate_ell:
            return (
                self.psi2.truncate_ell(truncate_ell)
                + self.sigma.bar.eth.eth.truncate_ell(truncate_ell)
                + self.sigma.multiply(self.sigma.bar.dot, truncator=lambda tup: truncate_ell)
            )
        else:
            return self.psi2 + self.sigma.bar.eth.eth + self.sigma * self.sigma.bar.dot

    @property
    def bondi_four_momentum(self):
        """Compute the Bondi four-momentum of the AsymptoticBondiData"""
        Psi_restricted = self.mass_aspect(1).view(np.ndarray).real  # Compute only the parts of the mass aspect we need, ell<=1
        four_momentum = np.empty(Psi_restricted.shape, dtype=float)
        four_momentum[..., 0] = - Psi_restricted[..., 0] / math.sqrt(8)
        four_momentum[..., 1:4] = - Psi_restricted[..., 1:4] / 6
        return four_momentum

    from .constraints import (
        bondi_constraints, bondi_violations, bondi_violation_norms,
        bianchi_0, bianchi_1, bianchi_2,
        constraint_3, constraint_4, constraint_mass_aspect
    )

    from .transformations import transform
