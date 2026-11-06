import numpy as np
from spherical import LM_total_size
from .. import WaveformModes
from .. import ModesTimeSeries
from .. import Inertial
from .. import h as h_DataType

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

        self.frame = np.array([])
        self.frameType = frameType
        self._time = time.copy()
        self._ell_max = ell_max
        self._ell_min = 0
        self._multiplication_truncator = multiplication_truncator

        fields = ("psi0", "psi1", "psi2", "psi3", "psi4", "sigma")
        #Initialize field attributes with None.
        for field in fields:
            setattr(self, f"_{field}", None)

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
        return self.data.shape[-1]

    @property
    def ell_min(self):
        return self._ell_min

    @property
    def ell_max(self):
        return self._ell_max

    @property
    def LM(self):
        """Array of (ell, m) values in the data
        This array is just a flat array of `[ell, m]` pairs.
        """
        return np.array([[ell, m] for ell in range(self.ell_min, self.ell_max+1) for m in range(-ell, ell+1)])

    @property
    def sigma(self):
        if self._sigma is None:
            raise AttributeError("sigma data has not been provided.")
        else:
            return self._sigma

    @sigma.setter
    def sigma(self, sigmaprm):
        if self._sigma is None:
            self._sigma = self._construct_MTS_with_zeros(spin_weight=2)
        self._sigma[:] = sigmaprm
        return self.sigma

    @property
    def h(self):
        h_mts = 2.0 * self._sigma.bar
        return WaveformModes(
            t=h_mts.t,
            data=np.array(h_mts)[:, h_mts.index(abs(h_mts.s), -abs(h_mts.s)) :],
            ell_min=abs(h_mts.s),
            ell_max=h_mts.ell_max,
            frameType=Inertial,
            dataType=h_DataType,
            r_is_scaled_out=True,
            m_is_scaled_out=True,
        )

    @property
    def psi4(self):
        if self._psi4 is None:
            raise AttributeError("psi4 data has not been provided.")
        else:
            return self._psi4

    @psi4.setter
    def psi4(self, psi4prm):
        if self._psi4 is None:
            self._psi4 = self._construct_MTS_with_zeros(spin_weight=-2)
        self._psi4[:] = psi4prm
        return self.psi4

    @property
    def psi3(self):
        if self._psi3 is None:
            raise AttributeError("psi3 data has not been provided.")
        else:
            return self._psi3

    @psi3.setter
    def psi3(self, psi3prm):
        if self._psi3 is None:
            self._psi3 = self._construct_MTS_with_zeros(spin_weight=-1)
        self._psi3[:] = psi3prm
        return self.psi3

    @property
    def psi2(self):
        if self._psi2 is None:
            raise AttributeError("psi2 data has not been provided.")
        else:
            return self._psi2

    @psi2.setter
    def psi2(self, psi2prm):
        if self._psi2 is None:
            self._psi2 = self._construct_MTS_with_zeros(spin_weight=0)
        self._psi2[:] = psi2prm
        return self.psi2

    @property
    def psi1(self):
        if self._psi1 is None:
            raise AttributeError("psi1 data has not been provided.")
        else:
            return self._psi1

    @psi1.setter
    def psi1(self, psi1prm):
        if self._psi1 is None:
            self._psi1 = self._construct_MTS_with_zeros(spin_weight=1)
        self._psi1[:] = psi1prm
        return self.psi1

    @property
    def psi0(self):
        if self._psi0 is None:
            raise AttributeError("psi0 data has not been provided")
        else:
            return self._psi0

    @psi0.setter
    def psi0(self, psi0prm):
        if self._psi0 is None:
            self._psi0 = self._construct_MTS_with_zeros(spin_weight=2)
        self._psi0[:] = psi0prm
        return self.psi0

    @property
    def data_components(self):
        """Return fields that are present in the ABD object."""

        fields = ["psi0", "psi1", "psi2", "psi3", "psi4", "sigma"]
        fields_present = [field for field in fields if getattr(self, f"_{field}") is not None]

        return fields_present

    @property
    def data(self):
        """Retun a consolidated array of the fields present in the ABD object
        with shape (n_components, n_times, n_modes).
        """
        fields = self.data_components
        n_modes = LM_total_size(0, self.ell_max)
        n_times = self.n_times
        n_components = len(fields)

        shape = [n_components, n_times, n_modes]
        data = np.empty(shape, dtype=complex)

        for i, field in enumerate(fields):
            data[i] = getattr(self, f"_{field}")

        return data

    #Slicing
    def __getitem__(self, key):
        """
        Extract time slices of the asymptotic Bondi data efficiently.
        """
        # If key is a valid time slice or index, extract the corresponding
        # sliced data
        if not isinstance(key, (slice, int)):
            raise ValueError(f"Invalid key `{key}` of type `{type(key)}`.")

        import functools

        ModesTS = functools.partial(ModesTimeSeries, ell_max=self.ell_max, multiplication_truncator=self._multiplication_truncator)

        new_times = self.time[key]
        new_abd = type(self)(new_times, self.ell_max)
        new_abd.frameType = self.frameType
        new_abd._multiplication_truncator = self._multiplication_truncator

        for i, field in enumerate(self.data_components):
            sliced_data = getattr(self, field)[key]
            setattr(new_abd, f"{field}", sliced_data)

        if self.frame.shape[0] == self.n_times:
            new_abd.frame = self.frame[key]
        return new_abd

    def copy(self):
        import copy

        new_abd = type(self)(self.t, self.ell_max)
        state = copy.deepcopy(self.__dict__)
        new_abd.__dict__.update(state)
        return new_abd

    def interpolate(self, new_times):
        new_abd = type(self)(new_times, self.ell_max)
        new_abd.frameType = self.frameType
        new_abd._multiplication_truncator = self._multiplication_truncator
        # interpolate waveform data
        for field in self.data_components:
            interpolated_fields = getattr(self, field).interpolate(new_times)
            setattr(new_abd, f"_{field}", interpolated_fields)

        # interpolate frame data if necessary
        if self.frame.shape[0] == self.n_times:
            import quaternion

            new_abd.frame = quaternion.squad(self.frame, self.t, new_times)
        return new_abd

    def _construct_MTS_with_zeros(self, spin_weight=0):
        """Return a new zero-filled ModesTimeSeries of shape (n_times, n_modes),
        spanning ell = 0 to ell_max, with the given spin weight. This is a
        helper function used for supporting the setter property of ABD fields.
        """
        n_modes = LM_total_size(0, self.ell_max)
        n_times = self.n_times

        shape = [n_times, n_modes]
        zeros_array = np.zeros(shape, dtype=complex)

        zeros_MTS = ModesTimeSeries(zeros_array, self.t, ell_min=0, ell_max=self.ell_max, spin_weight=spin_weight, multiplication_truncator=self._multiplication_truncator)

        return zeros_MTS

    def validate_fields(self):
        """Check if the provided fields are sensible for performing BMS
        transformations.
        """
        fields = ["psi0", "psi1", "psi2", "psi3", "psi4"]
        data_components = self.data_components
        fields_present = [True if field in data_components else False for field in fields]

        for i, (name, present) in enumerate(zip(fields, fields_present)):
            if present:
                missing = [f for f, p in zip(fields[i + 1 :], fields_present[i + 1 :]) if not p]
                if missing:
                    raise ValueError(f"{name} is present but higher-order Weyl scalars are missing: {missing}")

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
        CWWY_angular_momentum,
        bondi_dimensionless_spin,
        bondi_boost_charge,
        bondi_CoM_charge,
        supermomentum,
    )

    from .map_to_superrest_frame import map_to_superrest_frame
    from .map_to_abd_frame import map_to_abd_frame
