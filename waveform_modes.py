# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import

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
            self.__ell_min = kwargs.pop('ell_min', 0)
            self.__ell_max = kwargs.pop('ell_max', -1)
            self.__LM = sf.LM_range(self.__ell_min, self.__ell_max)
        super(WaveformModes, self).__init__(*args, **kwargs)

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
        test(errors,
             isinstance(self.__ell_min, numbers.Integral),
             'isinstance(self.__ell_min, numbers.Integral) # type(self.__ell_min)={0}'.format(type(self.__ell_min)))
        test(errors,
             isinstance(self.__ell_max, numbers.Integral),
             'isinstance(self.__ell_max, numbers.Integral) # type(self.__ell_max)={0}'.format(type(self.__ell_max)))
        test(errors,
             self.__ell_min >= 0,
             'self.__ell_min>=0 # {0}'.format(self.__ell_min))
        test(errors,
             self.__ell_max >= self.__ell_min - 1,
             'self.__ell_max>=self.__ell_min-1 # self.__ell_max={0}; self.__ell_min-1={1}'.format(self.__ell_max,
                                                                                                  self.__ell_min - 1))
        if alter and not np.array_equal(self.__LM, sf.LM_range(self.ell_min, self.ell_max)):
            self.__LM = sf.LM_range(self.ell_min, self.ell_max)
            alterations += [
                '{0}._{1}__LM = sf.LM_range({2}, {3})'.format(self, type(self).__name__, self.ell_min, self.ell_max)]
        test(errors,
             np.array_equal(self.__LM, sf.LM_range(self.ell_min, self.ell_max)),
             'np.array_equal(self.__LM, sf.LM_range(self.ell_min, self.ell_max))')

        test(errors,
             self.data.dtype == np.dtype(np.complex),
             'self.data.dtype == np.dtype(np.complex) # self.data.dtype={0}'.format(self.data.dtype))
        test(errors,
             self.data.ndim >= 2,
             'self.data.ndim >= 2 # self.data.ndim={0}'.format(self.data.ndim))
        test(errors,
             self.data.shape[1] == self.__LM.shape[0],
             'self.data.shape[1]==self.__LM.shape[0] '
             '# self.data.shape={0}; self.__LM.shape[0]={1}'.format(self.data.shape[1], self.__LM.shape[0]))

        if alterations:
            self._append_history(alterations)
            print("The following alterations were made:\n\t" + '\n\t'.join(alterations))
        if errors:
            print("The following conditions were found to be incorrectly False:\n\t" + '\n\t'.join(errors))
            return False

        # Call the base class's version
        super(WaveformModes, self).ensure_validity(alter, assertions)

        self.__history_depth__ -= 1
        self._append_history('WaveformModes.ensure_validity' +
                             '({0}, alter={1}, assertions={2})'.format(self, alter, assertions))

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
            warning = "\nWaveform's data.shape={0} does not agree with ".format(self.data.shape) \
                      + "(ell_min,ell_max)=({0},{1}).\n".format(self.ell_min, self.ell_max) \
                      + "Hopefully you are about to reset `data`.  To suppress this warning,\n" \
                      + "reset `data` before resetting ell_min and/or ell_max."
            warnings.warn(warning)

    @property
    def ell_max(self):
        return self.__ell_max

    @ell_max.setter
    def ell_max(self, new_ell_max):
        self.__ell_max = new_ell_max
        self.__LM = sf.LM_range(self.ell_min, self.ell_max)
        if self.n_modes != self.__LM.shape[0]:
            warning = "\nWaveform's data.shape={0} does not agree with ".format(self.data.shape) \
                      + "(ell_min,ell_max)=({0},{1}).\n".format(self.ell_min, self.ell_max) \
                      + "Hopefully you are about to reset `data`.  To suppress this warning,\n" \
                      + "reset `data` before resetting ell_min and/or ell_max."
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
            warning = "\nWaveform's data.shape={0} does not agree with ".format(self.data.shape) \
                      + "(ell_min,ell_max)=({0},{1}).\n".format(self.ell_min, self.ell_max) \
                      + "Hopefully you are about to reset `data`.  To avoid this warning,\n" \
                      + "reset `data` before resetting ell_min and/or ell_max."
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
        if not (ell_m.dtype == np.int and ell_m.shape[1] == 2):
            raise ValueError("Input `ell_m` should be an Nx2 sequence of integers")
        return ell_m[:, 0] * (ell_m[:, 0] + 1) - self.ell_min ** 2 + ell_m[:, 1]

    def apply_eth(self, operations, eth_convention='NP'):
        """Apply spin raising/lowering operators to waveform mode data in a specified order.

        Parameters
        ----------
        operations: str of combinations of '+' and/or '-'
            The order of eth ('+') and ethbar ('-') operations to perform, applied from right to
            left. Example, operations='--+' will perform on WaveformModes data f the operation
            ethbar(ethbar(eth(f))).  Note that the order of operations is right-to-left.
        eth_convention: either 'NP' or 'GHP' 
            Choose between Newman-Penrose or Geroch-Held-Penrose convention

        Returns
        -------
        mode_data: array of complex
            Note that the returned data has the same shape as this object's `data` attribute, and
            the modes correspond to the same (ell, m) values.  In particular, if the spin weight
            changes, the output data will no longer satisfy the expectation that ell_min == abs(s).

        """
        import spherical_functions as sf
        if eth_convention == 'NP':
            eth = sf.eth_NP
            ethbar = sf.ethbar_NP
        elif eth_convention == 'GHP':
            eth = sf.eth_GHP
            ethbar = sf.ethbar_GHP
        else:
            raise ValueError("eth_convention must either be 'NP' or 'GHP'; got '{}'".format(eth_convention))
        s = self.spin_weight
        mode_data = self.data.transpose()
        if not set(operations).issubset({'+','-'}):
            raise ValueError("Operations must be combinations of '+' and '-'; got '{}'".format(operations))
        for operation in reversed(operations):
            if operation == '+':
                mode_data = eth(mode_data, s, self.ell_min)
                s += 1
            elif operation == '-':
                mode_data = ethbar(mode_data, s, self.ell_min)
                s -= 1
        return mode_data.transpose()

    @property
    def eth(self):
        """Returns the spin-raised waveform mode data.

        """
        return self.apply_eth(operations='+')
    
    @property
    def ethbar(self):
        """Returns the spin-lowered waveform mode data.

        """
        return self.apply_eth(operations='-')

    def inner_product(self, b,
                      t1=None, t2=None,
                      allow_LM_differ=False, allow_times_differ=False):
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

        if (self.spin_weight != b.spin_weight):
            raise ValueError("Spin weights must match in inner_product")

        LM_clip = None
        if ((self.ell_min != b.ell_min) or (self.ell_max != b.ell_max)):
            if (allow_LM_differ):
                LM_clip = slice( max(self.ell_min, b.ell_min),
                                 min(self.ell_max, b.ell_max) + 1 )
                if (LM_clip.start >= LM_clip.stop):
                    raise ValueError("Intersection of (ell,m) modes is "
                                     "empty.  Assuming this is not desired.")
            else:
                raise ValueError("ell_min and ell_max must match in inner_product "
                                 "(use allow_LM_differ=True to override)")

        t_clip = None
        if not np.array_equal(self.t, b.t):
            if (allow_times_differ):
                t_clip = intersection(self.t, b.t)
            else:
                raise ValueError("Time samples must match in inner_product "
                                 "(use allow_times_differ=True to override)")

        ##########

        times = self.t
        A = self
        B = b

        if (LM_clip is not None):
            A = A[:,LM_clip]
            B = B[:,LM_clip]

        if (t_clip is not None):
            times = t_clip
            A = A.interpolate(t_clip)
            B = B.interpolate(t_clip)

        if (t1 is None):
            t1 = times[0]

        if (t2 is None):
            t2 = times[-1]

        integrand = np.sum(np.conj(A.data) * B.data, axis=1)

        return sdi(integrand, times, t1=t1, t2=t2)

    # Involutions
    @property
    @waveform_alterations
    def x_parity_conjugate(self):
        """Reflect modes across y-z plane (along x axis)

        See "Gravitational-wave modes from precessing black-hole binaries" by
        Boyle et al. (2014) for more details.

        """
        if self.dataType == UnknownDataType:
            raise ValueError("Cannot compute parity type for {0}.".format(self.data_type_string))
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
        W._append_history('{0} = {1}.x_parity_conjugate'.format(W, self))
        return W

    @property
    @waveform_alterations
    def x_parity_symmetric_part(self):
        """Return component of waveform invariant under `x_parity_conjugate`"""
        W = self.x_parity_conjugate
        W.data = 0.5 * (self.data + W.data)
        W.frame = np.x_parity_symmetric_part(self.frame)
        W.__history_depth__ -= 1
        W._append_history('{0} = {1}.x_parity_symmetric_part'.format(W, self))
        return W

    @property
    @waveform_alterations
    def x_parity_antisymmetric_part(self):
        """Return component of waveform that changes sign under `x_parity_conjugate`"""
        W = self.x_parity_conjugate
        W.data = 0.5 * (self.data - W.data)
        W.frame = np.x_parity_antisymmetric_part(self.frame)
        W.__history_depth__ -= 1
        W._append_history('{0} = {1}.x_parity_antisymmetric_part'.format(W, self))
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
            raise ValueError("Cannot compute parity type for {0}.".format(self.data_type_string))
        W = self[:, :0]  # W without `data`, and `ells`=(0,-1); different from `self.copy_without_data()`
        W.data = np.conjugate(self.data)
        W.ells = self.ells
        W.frame = np.y_parity_conjugate(self.frame)
        W.__history_depth__ -= 1
        W._append_history('{0} = {1}.y_parity_conjugate'.format(W, self))
        return W

    @property
    @waveform_alterations
    def y_parity_symmetric_part(self):
        """Component of waveform invariant under `y_parity_conjugate`"""
        W = self.y_parity_conjugate
        W.data = 0.5 * (self.data + W.data)
        W.frame = np.y_parity_symmetric_part(self.frame)
        W.__history_depth__ -= 1
        W._append_history('{0} = {1}.y_parity_symmetric_part'.format(W, self))
        return W

    @property
    @waveform_alterations
    def y_parity_antisymmetric_part(self):
        """Component of waveform that changes sign under `y_parity_conjugate`"""
        W = self.y_parity_conjugate
        W.data = 0.5 * (self.data - W.data)
        W.frame = np.y_parity_antisymmetric_part(self.frame)
        W.__history_depth__ -= 1
        W._append_history('{0} = {1}.y_parity_antisymmetric_part'.format(W, self))
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
            raise ValueError("Cannot compute parity type for {0}.".format(self.data_type_string))
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
        W._append_history('{0} = {1}.z_parity_conjugate'.format(W, self))
        return W

    @property
    @waveform_alterations
    def z_parity_symmetric_part(self):
        """Component of waveform invariant under `z_parity_conjugate`"""
        W = self.z_parity_conjugate
        W.data = 0.5 * (self.data + W.data)
        W.frame = np.z_parity_symmetric_part(self.frame)
        W.__history_depth__ -= 1
        W._append_history('{0} = {1}.z_parity_symmetric_part'.format(W, self))
        return W

    @property
    @waveform_alterations
    def z_parity_antisymmetric_part(self):
        """Component of waveform that changes sign under `z_parity_conjugate`"""
        W = self.z_parity_conjugate
        W.data = 0.5 * (self.data - W.data)
        W.frame = np.z_parity_antisymmetric_part(self.frame)
        W.__history_depth__ -= 1
        W._append_history('{0} = {1}.z_parity_antisymmetric_part'.format(W, self))
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
            raise ValueError("Cannot compute parity type for {0}.".format(self.data_type_string))
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
        W._append_history('{0} = {1}.parity_conjugate'.format(W, self))
        return W

    @property
    @waveform_alterations
    def parity_symmetric_part(self):
        """Component of waveform invariant under `parity_conjugate`"""
        W = self.parity_conjugate
        W.data = 0.5 * (self.data + W.data)
        W.frame = np.parity_symmetric_part(self.frame)
        W.__history_depth__ -= 1
        W._append_history('{0} = {1}.parity_symmetric_part'.format(W, self))
        return W

    @property
    @waveform_alterations
    def parity_antisymmetric_part(self):
        """Component of waveform that changes sign under `parity_conjugate`"""
        W = self.parity_conjugate
        W.data = 0.5 * (self.data - W.data)
        W.frame = np.parity_antisymmetric_part(self.frame)
        W.__history_depth__ -= 1
        W._append_history('{0} = {1}.parity_antisymmetric_part'.format(W, self))
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
        W = super(WaveformModes, self).copy_without_data()
        W.ells = 0, -1
        W.__history_depth__ -= 1
        W._append_history('{0} = {1}.copy_without_data()'.format(W, self))
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
                    raise ValueError("Requested ell value {0} lies outside ".format(key[1]) +
                                     "WaveformModes object's ell range ({0},{1}).".format(self.ell_min, self.ell_max))
                new_ell_min = key[1]
                new_ell_max = key[1]
                new_slice = slice(new_ell_min ** 2 - self.ell_min ** 2,
                                  new_ell_max * (new_ell_max + 2) + 1 - self.ell_min ** 2)
            elif isinstance(key[1], slice):
                if key[1].step and key[1].step != 1:
                    raise ValueError(
                        "Can only slice WaveformModes over contiguous ell values (step={0})".format(key[1].step))
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
                            "Requested ell range [{0},{1}] lies outside ".format(new_ell_min, new_ell_max) +
                            "WaveformBase's ell range [{0},{1}].".format(self.ell_min, self.ell_max))
                    new_slice = slice(new_ell_min ** 2 - self.ell_min ** 2,
                                      new_ell_max * (new_ell_max + 2) + 1 - self.ell_min ** 2)
            else:
                raise ValueError("Don't know what to do with slice of type `{0}`".format(type(key[1])))
            W = super(WaveformModes, self).__getitem__((key[0], new_slice))
            W.ells = new_ell_min, new_ell_max
        elif isinstance(key, slice) or isinstance(key, int):
            # Return complete data from a subset of times (key is slice), or
            # return complete data from a single instant in time (key is int)
            W = super(WaveformModes, self).__getitem__(key)
            W.ells = self.ells
        else:
            raise ValueError("Could not understand input `{0}` (of type `{1}`) ".format(key, type(key)))

        W.history.pop()  # remove WaveformBase history append, to be replaced next
        W.__history_depth__ -= 1
        W._append_history('{0} = {1}[{2}]'.format(W, self, key))

        return W

    def __repr__(self):
        # "The goal of __str__ is to be readable; the goal of __repr__ is to be unambiguous." --- stackoverflow
        rep = super(WaveformModes, self).__repr__()
        rep += "\n# ell_min={0}, ell_max={1}".format(self.ell_min, self.ell_max)
        return rep


    # N.B.: There are additional methods attached to `WaveformModes` in the `waveform_base` file.  These functions
    # cannot be added here, because they depend on `WaveformGrid` objects, which depend on `WaveformModes` objects.
