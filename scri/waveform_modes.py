# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

from .waveform_base import WaveformBase, waveform_alterations

import warnings
import numpy as np
import spherical_functions as sf
from . import *


class WaveformModes(WaveformBase):
    """Object containing time, frame, and data, along with related information

    This object collects all the data needed to manipulate time-series of spin-weighted spherical-harmonic modes of
    gravitational waves, as well as a few useful informational members.  Various methods exist to manipulate the data
    and extract important information.

    As much as possible, completeness of modes is enforced -- meaning that if any (ell,m) mode with a given ell is
    included in the data, all modes with abs(m)<=ell must also be included.  The data are stored as complex numbers,
    and assumed to be in standard order corresponding to

        [f(ell,m) for ell in range(ell_min, ell_max+1) for m in range(-ell,ell+1)]

    There is no automated check that can be done to assure that the order is correct.  (Precession breaks all
    symmetries, for example.)  The only check that is done ensures that the input data array has the correct
    dimensions.  It is up to the user to ensure that the order is correct.

    Attributes
    ----------
    t : float array
        Time steps corresponding to other data
    frame : quaternion array
        Rotors taking static basis onto decomposition basis
    data : 2-d complex array
        Mode values of the spin-weighted spherical-harmonic decomposition. The array is assumed to have a first
        dimension equal to the length of `t`, and a second dimension equal to the number of modes as deduced from the
        values of ell_min and ell_max below.  As noted above, a very particular order is assumed for the data.
    ell_min : int
        Smallest ell value present in `data`.
    ell_max : int
        Largest ell value present in `data`.
    LM : int array (read only)
        Array of (ell,m) values corresponding to the `data` member.  This is automatically constructed based on the
        values of ell_min and ell_max, and cannot be reassigned.
    history : list of strings
        As far as possible, all functions applied to the object are recorded in the `history` variable.  In fact,
        the object should almost be able to be recreated using the commands in the history list. Commands taking
        large arrays, however, are shortened -- so the data will not be entirely reconstructable.
    frameType : int
        Index corresponding to `scri.FrameType` appropriate for `data`.
    dataType : int
        Index corresponding to `scri.DataType` appropriate for `data`.
    r_is_scaled_out : bool
        True if the `data` have been multiplied by the appropriate power of radius so that the asymptotic value can
        be finite and nonzero.
    m_is_scaled_out : bool
        True if the `data` have been scaled by the appropriate value of the total mass so that they are dimensionless.
    num : int (read only)
        Automatically assigned number of this object.  The constructor of this type keeps count of the number of
        objects it has created, to assign each object a more-or-less unique ID for use in the history strings.  This
        counter is reset at the beginning of each python session.  Subclasses should automatically have a different
        counter.

    Indexing
    --------
    WaveformMode objects can be indexed much like a numpy array, where the first dimension gives the time indices,
    and the second gives the mode indices. This will return another WaveformMode object containing slices of the
    original data.

    It is important to note, however, that as with numpy array slices, slicing a WaveformMode object will not
    typically copy the original data; the result will simply be a view into the data.  This means that changing the
    data in the slice can change the data in the original.  If you want to make a copy, you should probably use the
    copy constructor: `W2 = WaveformMode(W1)`. It is also possible to use the standard copy.deepcopy method.

    Also note that the first slice dimension corresponds to the indices of the time data, but the second dimension
    does NOT correspond to indices.  Instead, because this object tries to enforce completeness of the mode data,
    you can only slice a WaveformBase with respect to modes.  If you really want just some of the data, you'll need
    to extract it as a plain array, and then just slice as usual.

    For example,

    >>> W  = WaveformModes()
    >>> W[10:-20]

    will give all modes in the data, but only at times starting with the 10th time step, and ending one before the
    -20th time step.  Meanwhile,

    >>> W[10:-20,2]

    will give the same range of times, but only the ell=2 modes -- which includes (ell,m) modes (2,-2) through (2, 2).
    Similarly,

    >>> W[10:-20,2:5]

    will return the same range of times, along with the ell=2,3,4 modes. Note the lack of ell=5 mode, for consistency
    with python's usual slice syntax.

    >>> W[:,:0]

    will return all time steps, along with all `frame` data, but the `lm` data and `data` will be empty (because the
    `:0` term selects everything before the 0th element).  Similarly,

    >>> W[:0,:0]

    is empty of all the numerical data.



    """

    def __init__(self, *args, **kwargs):
        """Initializer for WaveformModes object

        This initializer is primarily a wrapper around the WaveformBase initializer.  See the docstring of
        WaveformBase for more details.  The only difference in calling is that this takes two additional keyword
        parameters:

        Keyword parameters
        ------------------
        ell_min : int, defaults to 0
        ell_max : int, defaults to -1

        """
        if len(args) == 1 and isinstance(args[0], type(self)):
            other = args[0]
            # Do not directly access __ell_min, __ell_max, or __LM outside of this initializer function; use ell_min,
            #  ell_max, or LM instead
            self.__ell_min = other.__ell_min
            self.__ell_max = other.__ell_max
            self.__LM = np.copy(other.__LM)
        else:
            # Do not directly access __ell_min, __ell_max, or __LM outside of this initializer function; use ell_min,
            # ell_max, or LM instead
            self.__ell_min = kwargs.pop("ell_min", 0)
            self.__ell_max = kwargs.pop("ell_max", -1)
            self.__LM = sf.LM_range(self.__ell_min, self.__ell_max)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_sxs(cls, w_sxs, override_exception_from_invalidity=False):
        """Construct this object from an `sxs.WaveformModes` object

        Note that the resulting object will likely contain references to the same
        underlying data contained in the original object; modifying one will modify the
        other.  You can make a copy of the result — using code like
        `WaveformModes.from_sxs(w_sxs).copy()` — to obtain separate data.

        """
        import quaternion
        constructor_statement = (
            f"WaveformModes.from_sxs({w_sxs}, "
            f"override_exception_from_invalidity={override_exception_from_invalidity})"
        )

        try:
            frameType = [n.lower() for n in FrameNames].index(w_sxs.frame_type.lower())
        except ValueError:
            frameType = 0

        try:
            dataType = [n.lower() for n in DataNames].index(w_sxs.data_type.lower())
        except ValueError:
            dataType = 0

        kwargs = dict(
            t=w_sxs.t,
            #frame=,  # see below
            data=w_sxs.data,
            history=w_sxs._metadata.get("history", []),
            version_hist=w_sxs._metadata.get("version_hist", []),
            frameType=frameType,
            dataType=dataType,
            r_is_scaled_out=w_sxs._metadata.get("r_is_scaled_out", True),
            m_is_scaled_out=w_sxs._metadata.get("m_is_scaled_out", True),
            override_exception_from_invalidity=override_exception_from_invalidity,
            constructor_statement=constructor_statement,
            ell_min=w_sxs.ell_min,
            ell_max=w_sxs.ell_max,
        )

        frame = w_sxs.frame.ndarray
        if np.array_equal(frame, [[1.,0,0,0]]):
            pass  # The default will handle itself
        elif frame.shape[0] == 1:
            kwargs["frame"] = quaternion.as_quat_array(frame[0, :])
        elif frame.shape[0] == w_sxs.n_times:
            kwargs["frame"] = quaternion.as_quat_array(frame)
        else:
            raise ValueError(
                f"Frame size ({frame.size}) should be 1 or "
                f"equal to the number of time steps ({self.n_times})"
            )

        return cls(**kwargs)

    @property
    def to_sxs(self):
        """Convert this object to an `sxs.WaveformModes` object

        Note that the resulting object will likely contain references to the same
        underlying data contained in the original object; modifying one will modify the
        other.  You can make a copy of this object *before* calling this function —
        using code like `w.copy().to_sxs` — to obtain separate data.

        """
        import sxs
        import quaternionic
        import quaternion

        # All of these will be stored in the `_metadata` member of the resulting WaveformModes
        # object; most of these will also be accessible directly as attributes.
        kwargs = dict(
            time=self.t,
            time_axis=0,
            modes_axis=1,
            #frame=,  # see below
            spin_weight=self.spin_weight,
            data_type=self.data_type_string.lower(),
            frame_type=self.frame_type_string.lower(),
            history=self.history,
            version_hist=self.version_hist,
            r_is_scaled_out=self.r_is_scaled_out,
            m_is_scaled_out=self.m_is_scaled_out,
            ell_min=self.ell_min,
            ell_max=self.ell_max,
        )

        # If self.frame.size==0, we just don't pass any argument
        if self.frame.size == 1:
            kwargs["frame"] = quaternionic.array([quaternion.as_float_array(self.frame)])
        elif self.frame.size == self.n_times:
            kwargs["frame"] = quaternionic.array(quaternion.as_float_array(self.frame))
        elif self.frame.size > 0:
            raise ValueError(
                f"Frame size ({self.frame.size}) should be 0, 1, or "
                f"equal to the number of time steps ({self.n_times})"
            )

        w = sxs.WaveformModes(self.data, **kwargs)

        # Special case for the translation and boost
        if hasattr(self, "space_translation") or hasattr(self, "boost_velocity"):
            w.register_modification(
                self.transform,
                space_translation=list(getattr(self, "space_translation", [0., 0., 0.])),
                boost_velocity=list(getattr(self, "boost_velocity", [0., 0., 0.])),
            )

        return w

    @waveform_alterations
    def ensure_validity(self, alter=True, assertions=False):
        """Try to ensure that the `WaveformModes` object is valid

        See `WaveformBase.ensure_validity` for the basic tests.  This function also includes tests that `data` is
        complex, and consistent with the ell_min and ell_max values.

        """
        import numbers

        errors = []
        alterations = []

        if assertions:
            from .waveform_base import test_with_assertions

            test = test_with_assertions
        else:
            from .waveform_base import test_without_assertions

            test = test_without_assertions

        # We first need to check that the ell values make sense,
        # because we'll use these below
        test(
            errors,
            isinstance(self.__ell_min, numbers.Integral),
            "isinstance(self.__ell_min, numbers.Integral) # type(self.__ell_min)={}".format(type(self.__ell_min)),
        )
        test(
            errors,
            isinstance(self.__ell_max, numbers.Integral),
            "isinstance(self.__ell_max, numbers.Integral) # type(self.__ell_max)={}".format(type(self.__ell_max)),
        )
        test(errors, self.__ell_min >= 0, f"self.__ell_min>=0 # {self.__ell_min}")
        test(
            errors,
            self.__ell_max >= self.__ell_min - 1,
            "self.__ell_max>=self.__ell_min-1 # self.__ell_max={}; self.__ell_min-1={}".format(
                self.__ell_max, self.__ell_min - 1
            ),
        )
        if alter and not np.array_equal(self.__LM, sf.LM_range(self.ell_min, self.ell_max)):
            self.__LM = sf.LM_range(self.ell_min, self.ell_max)
            alterations += [
                "{}._{}__LM = sf.LM_range({}, {})".format(self, type(self).__name__, self.ell_min, self.ell_max)
            ]
        test(
            errors,
            np.array_equal(self.__LM, sf.LM_range(self.ell_min, self.ell_max)),
            "np.array_equal(self.__LM, sf.LM_range(self.ell_min, self.ell_max))",
        )

        test(
            errors,
            self.data.dtype == np.dtype(complex),
            f"self.data.dtype == np.dtype(complex) # self.data.dtype={self.data.dtype}",
        )
        test(errors, self.data.ndim >= 2, f"self.data.ndim >= 2 # self.data.ndim={self.data.ndim}")
        test(
            errors,
            self.data.shape[1] == self.__LM.shape[0],
            "self.data.shape[1]==self.__LM.shape[0] "
            "# self.data.shape={}; self.__LM.shape[0]={}".format(self.data.shape[1], self.__LM.shape[0]),
        )

        if alterations:
            self._append_history(alterations)
            print("The following alterations were made:\n\t" + "\n\t".join(alterations))
        if errors:
            print("The following conditions were found to be incorrectly False:\n\t" + "\n\t".join(errors))
            return False

        # Call the base class's version
        super().ensure_validity(alter, assertions)

        self.__history_depth__ -= 1
        self._append_history("WaveformModes.ensure_validity" + f"({self}, alter={alter}, assertions={assertions})")

        return True

    # Mode information
    @property
    def n_modes(self):
        return self.data.shape[1]

    @property
    def ell_min(self):
        return self.__ell_min

    @ell_min.setter
    def ell_min(self, new_ell_min):
        self.__ell_min = new_ell_min
        self.__LM = sf.LM_range(self.ell_min, self.ell_max)
        if self.n_modes != self.__LM.shape[0]:
            warning = (
                f"\nWaveform's data.shape={self.data.shape} does not agree with "
                + f"(ell_min,ell_max)=({self.ell_min},{self.ell_max}).\n"
                + "Hopefully you are about to reset `data`.  To suppress this warning,\n"
                + "reset `data` before resetting ell_min and/or ell_max."
            )
            warnings.warn(warning)

    @property
    def ell_max(self):
        return self.__ell_max

    @ell_max.setter
    def ell_max(self, new_ell_max):
        self.__ell_max = new_ell_max
        self.__LM = sf.LM_range(self.ell_min, self.ell_max)
        if self.n_modes != self.__LM.shape[0]:
            warning = (
                f"\nWaveform's data.shape={self.data.shape} does not agree with "
                + f"(ell_min,ell_max)=({self.ell_min},{self.ell_max}).\n"
                + "Hopefully you are about to reset `data`.  To suppress this warning,\n"
                + "reset `data` before resetting ell_min and/or ell_max."
            )
            warnings.warn(warning)

    @property
    def ells(self):
        """Return self.ell_min,self.ell_max"""
        return self.__ell_min, self.__ell_max

    @ells.setter
    def ells(self, new_ells):
        """Setting both at once can be necessary when changing the shape of `data`"""
        self.__ell_min = new_ells[0]
        self.__ell_max = new_ells[1]
        self.__LM = sf.LM_range(self.ell_min, self.ell_max)
        if self.n_modes != self.__LM.shape[0]:
            warning = (
                f"\nWaveform's data.shape={self.data.shape} does not agree with "
                + f"(ell_min,ell_max)=({self.ell_min},{self.ell_max}).\n"
                + "Hopefully you are about to reset `data`.  To avoid this warning,\n"
                + "reset `data` before resetting ell_min and/or ell_max."
            )
            warnings.warn(warning)

    @property
    def LM(self):
        """Array of (ell,m) values in Waveform data

        This array is just a flat array of `[ell,m]` pairs.  It is
        automatically recomputed each time `ell_min` or `ell_max` is
        changed.  Specifically, it is

            np.array([[ell,m]
                      for ell in range(self.ell_min,self.ell_max+1)
                      for m in range(-ell,ell+1)])

        """
        return self.__LM

    @LM.setter
    def LM(self, newLM):
        raise AttributeError("Can't set LM data.  This can only be controlled through the ell_min and ell_max members.")

    def index(self, ell, m):
        """Index of given (ell,m) mode in the data

        Parameters
        ----------
        ell: int
        m: int

        Returns
        -------
        idx: int
            Index such that self.LM[idx] is [ell, m]

        """
        return sf.LM_index(ell, m, self.ell_min)

    def indices(self, ell_m):
        """Indices of given (ell,m) modes in the data

        Parameters
        ----------
        ell_m: Nx2 array-like, int

        Returns
        -------
        idcs: array of int
            Index such that self.LM[idcs] is `ell_m`

        """
        ell_m = np.array(ell_m, copy=False)
        if not (ell_m.dtype == int and ell_m.shape[1] == 2):
            raise ValueError("Input `ell_m` should be an Nx2 sequence of integers")
        return ell_m[:, 0] * (ell_m[:, 0] + 1) - self.ell_min ** 2 + ell_m[:, 1]

    @waveform_alterations
    def truncate(self, tol=1e-10):
        """Truncate the precision of this object's `data` in place

        This function sets bits in `self.data` to 0 when they will typically contribute
        less than `tol` times the norm of the Waveform at that instant in time.  Here,
        "typically" means that we divide `tol` by the square-root of the number of
        modes in the waveform, so that if each mode contributes a random amount of
        error at that level, we can expect a *total* error of roughly `tol` times the
        norm.

        """
        if tol != 0.0:
            tol_per_mode = tol / np.sqrt(self.n_modes)
            absolute_tolerance = np.linalg.norm(self.data, axis=1) * tol_per_mode
            power_of_2 = (2.0 ** np.floor(-np.log2(absolute_tolerance)))[:, np.newaxis]
            self.data *= power_of_2
            np.round(self.data, out=self.data)
            self.data /= power_of_2
        self._append_history(f"{self}.truncate(tol={tol})")

    def ladder_factor(self, operations, s, ell, eth_convention="NP"):
        """Compute the 'ladder factor' for applying a sequence of spin
        raising/lower operations on a SWSH of given s, ell.

        Parameters
        ----------
        operations: list or str composed of +1, -1, '+', '-', 'ð', or 'ð̅'
            The order of eth (+1, '+', or 'ð') and
            ethbar (-1, '-', or 'ð̅') operations to perform, applied from right to
            left. Example, operations='--+' will perform on WaveformModes data f the operation
            ethbar(ethbar(eth(f))).  Note that the order of operations is right-to-left.
        eth_convention: either 'NP' or 'GHP' [default: 'NP']
            Choose between Newman-Penrose or Geroch-Held-Penrose convention

        Returns
        -------
        float
        """

        op_dict = {"ð": +1, "ð̅": -1, "+": +1, "-": -1, +1: +1, -1: -1}
        conv_dict = {"NP": 1.0, "GHP": 0.5}

        # Normalize combining unicode characters
        if isinstance(operations, str):
            operations = operations.replace("ð̅", "-").replace("ð", "+")

        keys = op_dict.keys()
        key_strings = {key for key in keys if isinstance(key, str)}
        if not set(operations).issubset(keys):
            raise ValueError(
                "operations must be a string composed of "
                "{} or a list with "
                "elements coming from the set {}".format(key_strings, set(keys))
            )

        if eth_convention not in conv_dict:
            raise ValueError("eth_convention must be one of {}".format(set(conv_dict.keys())))

        convention_factor = conv_dict[eth_convention]

        ladder = 1.0
        sign_factor = 1.0

        for op in reversed(operations):
            sign = op_dict[op]
            sign_factor *= sign
            ladder *= (ell - s * sign) * (ell + s * sign + 1.0) if (ell >= abs(s)) else 0.0
            ladder *= convention_factor
            s += sign

        ladder = sign_factor * np.sqrt(ladder)
        return ladder

    def apply_eth(self, operations, eth_convention="NP"):
        """Apply spin raising/lowering operators to waveform mode data
        in a specified order.  This does not modify the original
        waveform object.

        Parameters
        ----------
        operations: list or str composed of +1, -1, '+', '-', 'ð', or 'ð̅'
            The order of eth (+1, '+', or 'ð') and
            ethbar (-1, '-', or 'ð̅') operations to perform, applied from right to
            left. Example, operations='--+' will perform on WaveformModes data f the operation
            ethbar(ethbar(eth(f))).  Note that the order of operations is right-to-left.
        eth_convention: either 'NP' or 'GHP' [default: 'NP']
            Choose between Newman-Penrose or Geroch-Held-Penrose convention

        Returns
        -------
        mode_data: array of complex
            Note that the returned data has the same shape as this object's `data` attribute, and
            the modes correspond to the same (ell, m) values.  In particular, if the spin weight
            changes, the output data will no longer satisfy the expectation that ell_min == abs(s).
        """

        s = self.spin_weight
        mode_data = self.data.copy()

        for ell in range(self.ell_min, self.ell_max + 1):
            ladder_factor = self.ladder_factor(operations, s, ell, eth_convention=eth_convention)
            lm_indices = [sf.LM_index(ell, m, self.ell_min) for m in range(-ell, ell + 1)]
            mode_data[:, lm_indices] *= ladder_factor

        return mode_data

    @property
    def eth(self):
        """Returns the spin-raised waveform mode data."""
        return self.apply_eth(operations="+")

    @property
    def ethbar(self):
        """Returns the spin-lowered waveform mode data."""
        return self.apply_eth(operations="-")

    def inner_product(self, b, t1=None, t2=None, allow_LM_differ=False, allow_times_differ=False):
        """Compute the all-angles inner product <self, b>.

        self and b should have the same spin-weight, set of (ell,m)
        modes, and times.  Differing sets of modes and times can
        optionally be turned on.

        Parameters
        ----------
        b : WaveformModes object
            The other set of modes to inner product with.

        t1 : float, optional [default: None]
        t2 : float, optional [default: None]
            Lower and higher bounds of time integration

        allow_LM_differ : bool, optional [default: False]
            If True and if the set of (ell,m) modes between self and b
            differ, then the inner product will be computed using the
            intersection of the set of modes.

        allow_times_differ: bool, optional [default: False]
            If True and if the set of times between self and b differ,
            then both WaveformModes will be interpolated to the
            intersection of the set of times.

        Returns
        -------
        inner_product : complex
            <self, b>
        """

        from quaternion.calculus import spline_definite_integral as sdi

        from .extrapolation import intersection

        if self.spin_weight != b.spin_weight:
            raise ValueError("Spin weights must match in inner_product")

        LM_clip = None
        if (self.ell_min != b.ell_min) or (self.ell_max != b.ell_max):
            if allow_LM_differ:
                LM_clip = slice(max(self.ell_min, b.ell_min), min(self.ell_max, b.ell_max) + 1)
                if LM_clip.start >= LM_clip.stop:
                    raise ValueError("Intersection of (ell,m) modes is " "empty.  Assuming this is not desired.")
            else:
                raise ValueError(
                    "ell_min and ell_max must match in inner_product " "(use allow_LM_differ=True to override)"
                )

        t_clip = None
        if not np.array_equal(self.t, b.t):
            if allow_times_differ:
                t_clip = intersection(self.t, b.t)
            else:
                raise ValueError(
                    "Time samples must match in inner_product " "(use allow_times_differ=True to override)"
                )

        ##########

        times = self.t
        A = self
        B = b

        if LM_clip is not None:
            A = A[:, LM_clip]
            B = B[:, LM_clip]

        if t_clip is not None:
            times = t_clip
            A = A.interpolate(t_clip)
            B = B.interpolate(t_clip)

        if t1 is None:
            t1 = times[0]

        if t2 is None:
            t2 = times[-1]

        integrand = np.sum(np.conj(A.data) * B.data, axis=1)

        return sdi(integrand, times, t1=t1, t2=t2)

    @waveform_alterations
    def convert_to_conjugate_pairs(self):
        """Convert modes to conjugate-pair format in place

        This function alters this object's modes to store the sum and difference of pairs with
        opposite `m` values.  If we denote the modes `f[l, m]`, then we define

            s[l, m] = (f[l, m] + f̄[l, -m]) / √2
            d[l, m] = (f[l, m] - f̄[l, -m]) / √2

        For m<0 we replace the mode data with `d[l, -m]`, for m=0 we do nothing, and for m>0 we
        replace the mode data with `s[l, m]`.  That is, the mode data on output look like this:

            [..., d[2, 2], d[2, 1], f[2, 0], s[2, 1], s[2, 2], d[3, 3], d[3, 2], ...]

        The factor of √2 is chosen so that the sum of the magnitudes squared at each time for this
        data is the same as it is for the original data.

        """
        for ell in range(self.ell_min, self.ell_max + 1):
            for m in range(1, ell + 1):
                i_plus = self.index(ell, m)
                i_minus = self.index(ell, -m)
                mode_plus = self.data[..., i_plus].copy()
                mode_minus = self.data[..., i_minus].copy()
                self.data[..., i_plus] = (mode_plus + np.conjugate(mode_minus)) / np.sqrt(2)
                self.data[..., i_minus] = (mode_plus - np.conjugate(mode_minus)) / np.sqrt(2)
        self._append_history(f"{self}.convert_to_conjugate_pairs()")

    @waveform_alterations
    def convert_from_conjugate_pairs(self):
        """Convert modes from conjugate-pair format in place

        This function reverses the effects of `convert_to_conjugate_pairs`.  See that function's
        docstring for details.

        """
        for ell in range(self.ell_min, self.ell_max + 1):
            for m in range(1, ell + 1):
                i_plus = self.index(ell, m)
                i_minus = self.index(ell, -m)
                mode_plus = self.data[..., i_plus].copy()
                mode_minus = self.data[..., i_minus].copy()
                self.data[..., i_plus] = (mode_plus + mode_minus) / np.sqrt(2)
                self.data[..., i_minus] = np.conjugate(mode_plus - mode_minus) / np.sqrt(2)
        self._append_history(f"{self}.convert_from_conjugate_pairs()")

    def transform(self, **kwargs):
        """Transform modes by some BMS transformation

        This simply applies the `WaveformGrid.from_modes` function, followed by the
        `WaveformGrid.to_modes` function.  See their respective docstrings for more
        details.

        However, note that the `ell_max` parameter used in the second function call
        defaults here to the `ell_max` value in the input waveform.  This is slightly
        different from the usual default, because `WaveformGrid.from_modes` usually
        increases the effective ell value by 1.

        """
        from . import WaveformGrid
        return WaveformGrid.transform(self, **kwargs)

    # Involutions
    @property
    @waveform_alterations
    def x_parity_conjugate(self):
        """Reflect modes across y-z plane (along x axis)

        See "Gravitational-wave modes from precessing black-hole binaries" by
        Boyle et al. (2014) for more details.

        """
        if self.dataType == UnknownDataType:
            raise ValueError(f"Cannot compute parity type for {self.data_type_string}.")
        W = self[:, :0]  # W without `data`, and `ells`=(0,-1); different from `self.copy_without_data()`
        W.data = np.empty_like(self.data)
        W.ells = self.ells
        W.frame = np.x_parity_conjugate(self.frame)
        for ell in range(W.ell_min, W.ell_max + 1):
            lm_indices = [sf.LM_index(ell, m, W.ell_min) for m in range(-ell, ell + 1) if (m % 2) == 0]
            W.data[:, lm_indices] = np.conjugate(self.data[:, lm_indices])
            lm_indices = [sf.LM_index(ell, m, W.ell_min) for m in range(-ell, ell + 1) if (m % 2) != 0]
            W.data[:, lm_indices] = -np.conjugate(self.data[:, lm_indices])
        W.__history_depth__ -= 1
        W._append_history(f"{W} = {self}.x_parity_conjugate")
        return W

    @property
    @waveform_alterations
    def x_parity_symmetric_part(self):
        """Return component of waveform invariant under `x_parity_conjugate`"""
        W = self.x_parity_conjugate
        W.data = 0.5 * (self.data + W.data)
        W.frame = np.x_parity_symmetric_part(self.frame)
        W.__history_depth__ -= 1
        W._append_history(f"{W} = {self}.x_parity_symmetric_part")
        return W

    @property
    @waveform_alterations
    def x_parity_antisymmetric_part(self):
        """Return component of waveform that changes sign under `x_parity_conjugate`"""
        W = self.x_parity_conjugate
        W.data = 0.5 * (self.data - W.data)
        W.frame = np.x_parity_antisymmetric_part(self.frame)
        W.__history_depth__ -= 1
        W._append_history(f"{W} = {self}.x_parity_antisymmetric_part")
        return W

    @property
    def x_parity_violation_squared(self):
        """(Squared) norm of x-parity-antisymmetric component of waveform"""
        return self.x_parity_antisymmetric_part.norm()

    @property
    def x_parity_violation_normalized(self):
        """Norm of x-parity-antisymmetric component divided by norm of waveform"""
        return np.sqrt(self.x_parity_antisymmetric_part.norm() / self.norm())

    @property
    @waveform_alterations
    def y_parity_conjugate(self):
        """Reflect modes across x-z plane (along y axis)

        See "Gravitational-wave modes from precessing black-hole binaries" by
        Boyle et al. (2014) for more details.

        """
        if self.dataType == UnknownDataType:
            raise ValueError(f"Cannot compute parity type for {self.data_type_string}.")
        W = self[:, :0]  # W without `data`, and `ells`=(0,-1); different from `self.copy_without_data()`
        W.data = np.conjugate(self.data)
        W.ells = self.ells
        W.frame = np.y_parity_conjugate(self.frame)
        W.__history_depth__ -= 1
        W._append_history(f"{W} = {self}.y_parity_conjugate")
        return W

    @property
    @waveform_alterations
    def y_parity_symmetric_part(self):
        """Component of waveform invariant under `y_parity_conjugate`"""
        W = self.y_parity_conjugate
        W.data = 0.5 * (self.data + W.data)
        W.frame = np.y_parity_symmetric_part(self.frame)
        W.__history_depth__ -= 1
        W._append_history(f"{W} = {self}.y_parity_symmetric_part")
        return W

    @property
    @waveform_alterations
    def y_parity_antisymmetric_part(self):
        """Component of waveform that changes sign under `y_parity_conjugate`"""
        W = self.y_parity_conjugate
        W.data = 0.5 * (self.data - W.data)
        W.frame = np.y_parity_antisymmetric_part(self.frame)
        W.__history_depth__ -= 1
        W._append_history(f"{W} = {self}.y_parity_antisymmetric_part")
        return W

    @property
    def y_parity_violation_squared(self):
        """(Squared) norm of y-parity-antisymmetric component of waveform"""
        return self.y_parity_antisymmetric_part.norm()

    @property
    def y_parity_violation_normalized(self):
        """Norm of y-parity-antisymmetric component divided by norm of waveform"""
        return np.sqrt(self.y_parity_antisymmetric_part.norm() / self.norm())

    @property
    @waveform_alterations
    def z_parity_conjugate(self):
        """Reflect modes across x-y plane (along z axis)

        See "Gravitational-wave modes from precessing black-hole binaries" by
        Boyle et al. (2014) for more details.

        """
        if self.dataType == UnknownDataType:
            raise ValueError(f"Cannot compute parity type for {self.data_type_string}.")
        W = self[:, :0]  # W without `data`, and `ells`=(0,-1); different from `self.copy_without_data()`
        W.data = np.empty_like(self.data)
        W.ells = self.ells
        W.frame = np.z_parity_conjugate(self.frame)
        s = self.spin_weight
        for ell in range(W.ell_min, W.ell_max + 1):
            if ((ell + s) % 2) == 0:
                lm_indices = [sf.LM_index(ell, m, W.ell_min) for m in range(-ell, ell + 1)]
                W.data[:, lm_indices] = np.conjugate(self.data[:, list(reversed(lm_indices))])
            else:
                lm_indices = [sf.LM_index(ell, m, W.ell_min) for m in range(-ell, ell + 1)]
                W.data[:, lm_indices] = -np.conjugate(self.data[:, list(reversed(lm_indices))])
        W.__history_depth__ -= 1
        W._append_history(f"{W} = {self}.z_parity_conjugate")
        return W

    @property
    @waveform_alterations
    def z_parity_symmetric_part(self):
        """Component of waveform invariant under `z_parity_conjugate`"""
        W = self.z_parity_conjugate
        W.data = 0.5 * (self.data + W.data)
        W.frame = np.z_parity_symmetric_part(self.frame)
        W.__history_depth__ -= 1
        W._append_history(f"{W} = {self}.z_parity_symmetric_part")
        return W

    @property
    @waveform_alterations
    def z_parity_antisymmetric_part(self):
        """Component of waveform that changes sign under `z_parity_conjugate`"""
        W = self.z_parity_conjugate
        W.data = 0.5 * (self.data - W.data)
        W.frame = np.z_parity_antisymmetric_part(self.frame)
        W.__history_depth__ -= 1
        W._append_history(f"{W} = {self}.z_parity_antisymmetric_part")
        return W

    @property
    def z_parity_violation_squared(self):
        """(Squared) norm of z-parity-antisymmetric component of waveform"""
        return self.z_parity_antisymmetric_part.norm()

    @property
    def z_parity_violation_normalized(self):
        """Norm of z-parity-antisymmetric component divided by norm of waveform"""
        return np.sqrt(self.z_parity_antisymmetric_part.norm() / self.norm())

    @property
    @waveform_alterations
    def parity_conjugate(self):
        """Reflect modes along all axes

        See "Gravitational-wave modes from precessing black-hole binaries" by
        Boyle et al. (2014) for more details.

        """
        if self.dataType == UnknownDataType:
            raise ValueError(f"Cannot compute parity type for {self.data_type_string}.")
        W = self[:, :0]  # W without `data`, and `ells`=(0,-1); different from `self.copy_without_data()`
        W.data = np.empty_like(self.data)
        W.ells = self.ells
        W.frame = np.parity_conjugate(self.frame)
        s = self.spin_weight
        for ell in range(W.ell_min, W.ell_max + 1):
            lm_indices = [sf.LM_index(ell, m, W.ell_min) for m in range(-ell, ell + 1) if ((ell + s + m) % 2) == 0]
            W.data[:, lm_indices] = np.conjugate(self.data[:, list(reversed(lm_indices))])
            lm_indices = [sf.LM_index(ell, m, W.ell_min) for m in range(-ell, ell + 1) if ((ell + s + m) % 2) != 0]
            W.data[:, lm_indices] = -np.conjugate(self.data[:, list(reversed(lm_indices))])
        W.__history_depth__ -= 1
        W._append_history(f"{W} = {self}.parity_conjugate")
        return W

    @property
    @waveform_alterations
    def parity_symmetric_part(self):
        """Component of waveform invariant under `parity_conjugate`"""
        W = self.parity_conjugate
        W.data = 0.5 * (self.data + W.data)
        W.frame = np.parity_symmetric_part(self.frame)
        W.__history_depth__ -= 1
        W._append_history(f"{W} = {self}.parity_symmetric_part")
        return W

    @property
    @waveform_alterations
    def parity_antisymmetric_part(self):
        """Component of waveform that changes sign under `parity_conjugate`"""
        W = self.parity_conjugate
        W.data = 0.5 * (self.data - W.data)
        W.frame = np.parity_antisymmetric_part(self.frame)
        W.__history_depth__ -= 1
        W._append_history(f"{W} = {self}.parity_antisymmetric_part")
        return W

    @property
    def parity_violation_squared(self):
        """(Squared) norm of parity-antisymmetric component of waveform"""
        return self.parity_antisymmetric_part.norm()

    @property
    def parity_violation_normalized(self):
        """Norm of parity-antisymmetric component divided by norm of waveform"""
        return np.sqrt(self.parity_antisymmetric_part.norm() / self.norm())

    @waveform_alterations
    def copy_without_data(self):
        W = super().copy_without_data()
        W.ells = 0, -1
        W.__history_depth__ -= 1
        W._append_history(f"{W} = {self}.copy_without_data()")
        return W

    @waveform_alterations
    def __getitem__(self, key):
        """Extract subsets of the data efficiently

        Note that if a second index is given to this function, it corresponds to modes, rather than data indices. See
        the docstring of the WaveformModes class for examples.

        """
        # Remove trivial tuple structure first
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]

        # Now figure out which type of return is desired
        if isinstance(key, tuple) and len(key) == 2:
            # Return a subset of the data from a subset of times
            if isinstance(key[1], int):
                if key[1] < self.ell_min or key[1] > self.ell_max:
                    raise ValueError(
                        "Requested ell value {} lies outside ".format(key[1])
                        + f"WaveformModes object's ell range ({self.ell_min},{self.ell_max})."
                    )
                new_ell_min = key[1]
                new_ell_max = key[1]
                new_slice = slice(
                    new_ell_min ** 2 - self.ell_min ** 2, new_ell_max * (new_ell_max + 2) + 1 - self.ell_min ** 2
                )
            elif isinstance(key[1], slice):
                if key[1].step and key[1].step != 1:
                    raise ValueError(
                        "Can only slice WaveformModes over contiguous ell values (step={})".format(key[1].step)
                    )
                if not key[1].start and key[1].stop == 0:
                    new_ell_min = 0
                    new_ell_max = -1
                    new_slice = slice(0)
                else:
                    if not key[1].start:
                        new_ell_min = self.ell_min
                    else:
                        new_ell_min = key[1].start
                    if not key[1].stop:
                        new_ell_max = self.ell_max
                    else:
                        new_ell_max = key[1].stop - 1
                    if new_ell_min < self.ell_min or new_ell_max > self.ell_max:
                        raise ValueError(
                            f"Requested ell range [{new_ell_min},{new_ell_max}] lies outside "
                            + f"WaveformBase's ell range [{self.ell_min},{self.ell_max}]."
                        )
                    new_slice = slice(
                        new_ell_min ** 2 - self.ell_min ** 2, new_ell_max * (new_ell_max + 2) + 1 - self.ell_min ** 2
                    )
            else:
                raise ValueError("Don't know what to do with slice of type `{}`".format(type(key[1])))
            W = super().__getitem__((key[0], new_slice))
            W.ells = new_ell_min, new_ell_max
        elif isinstance(key, slice) or isinstance(key, int):
            # Return complete data from a subset of times (key is slice), or
            # return complete data from a single instant in time (key is int)
            W = super().__getitem__(key)
            W.ells = self.ells
        else:
            raise ValueError("Could not understand input `{}` (of type `{}`) ".format(key, type(key)))

        W.history.pop()  # remove WaveformBase history append, to be replaced next
        W.__history_depth__ -= 1
        W._append_history(f"{W} = {self}[{key}]")

        return W

    def __repr__(self):
        # "The goal of __str__ is to be readable; the goal of __repr__ is to be unambiguous." --- stackoverflow
        rep = super().__repr__()
        rep += f"\n# ell_min={self.ell_min}, ell_max={self.ell_max}"
        return rep

    # N.B.: There are additional methods attached to `WaveformModes` in the `waveform_base` file.  These functions
    # cannot be added here, because they depend on `WaveformGrid` objects, which depend on `WaveformModes` objects.
