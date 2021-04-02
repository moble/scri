# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

import os
import inspect
import functools
import warnings
import socket
import datetime
import pprint
import copy
import numpy as np
import quaternion
import scipy.constants as spc
from scipy.interpolate import CubicSpline
from . import *


@jit("void(c16[:,:], f8[:])")
def complex_array_norm(c, s):
    for i in range(len(s)):
        s[i] = 0.0
        for j in range(c.shape[1]):
            s[i] += c[i, j].real ** 2 + c[i, j].imag ** 2
    return


@jit("void(c16[:,:], f8[:])")
def complex_array_abs(c, s):
    for i in range(len(s)):
        s[i] = 0.0
        for j in range(c.shape[1]):
            s[i] += c[i, j].real ** 2 + c[i, j].imag ** 2
        s[i] = np.sqrt(s[i])
    return


def waveform_alterations(func):
    """Temporarily increment history depth safely

    This decorator stores the value of `self.__history_depth__`, then increments it by 1, calls the function,
    returns the history depth to its original value, and then returns the result of the function.  This should be
    used on any member function that could alter the waveform on which it is called, or which could return a new
    altered version of the original.

    Typically, within the function itself, you will want to decrement the depth manually just before appending to the
    history -- which will presumably take place at the end of the function.  You do not need to undo this,
    as the decorator will take care of that part.

    """

    @functools.wraps(func)
    def func_wrapper(self, *args, **kwargs):
        if self.__history_depth__ == 0:
            self._append_history("")
        stored_history_depth = self.__history_depth__
        self.__history_depth__ += 1
        result = func(self, *args, **kwargs)
        self.__history_depth__ = stored_history_depth
        return result

    return func_wrapper


def test_without_assertions(errs, val, msg=""):
    """Replacement for np.testing.assert_

    This function should be able to replace `assert_`, but rather than raising an exception, this just adds a
    description of the problem to the `errors` variable.

    """
    if not val:
        try:
            smsg = msg()
        except TypeError:
            smsg = msg
        errs += [smsg]


def test_with_assertions(errs, val, msg=""):
    np.testing.assert_(val, "Failed assertion:\n\t" + msg)


class _object:
    """Useless class to allow multiple inheritance"""

    def __init__(self, *args, **kwargs):
        super().__init__()


class WaveformBase(_object):
    """Object containing time, frame, and data, along with related information

    This object is just the base object from which these other classes are derived:
      * WaveformModes
      * WaveformGrid
      * WaveformInDetector
      * WaveformInDetectorFT
    For more specific information, see the documentation of those classes.

    Attributes
    ----------
    t : float array
        Time steps corresponding to other data
    frame : quaternion array
        Rotors taking static basis onto decomposition basis
    data : 2-d array of complex or real numbers
        The nature of this data depends on the derived type.  First index is time, second index depends on type.
    history : list of strings
        As far as possible, all functions applied to the object are recorded in the `history` variable.  In fact,
        the object should almost be able to be recreated using the commands in the history list. Commands taking
        large arrays, however, are shortened -- so the data will not be entirely reconstructable.
    version_hist : list of pairs of strings
        Records the git hash and description for any change in the way SpEC outputs waveform data.
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
    WaveformBase objects can be indexed much like a numpy array, where the first dimension gives the time indices,
    and the second gives the data-set indices. This will return another WaveformBase object containing slices of the
    original data.

    It is important to note, however, that as with numpy array slices, slicing a WaveformBase will not typically copy
    the original data; the result will simply be a view into the data.  This means that changing the data in the
    slice can change the data in the original.  If you want to make a copy, you should probably use the copy
    constructor: `W2 = WaveformBase(W1)`. It is also possible to use the standard copy.deepcopy method.

    Also note that the first slice dimension corresponds to the indices of the time, but the second dimension may NOT
    correspond to indices for derived types.  In particular, for `WaveformModes`, the second index corresponds to
    modes, because this type enforces completeness of each ell mode.  For the `WaveformBase` type, however,
    the second index does correspond to the second dimension of the data.

    For example,

    >>> W  = WaveformBase()
    >>> W[10:-20]

    will give all columns in the data, but only at times starting with the
    10th time step, and ending one before the -20th time step.  Meanwhile,

    >>> W[10:-20,2]

    will give the same range of times, but only the second column (unless the subclass overrides this behavior,
    as in `WaveformModes`).  Similarly,

    >>> W[10:-20,2:5]

    will return the same range of times, along with the 2,3,4 columns. Note the lack of 5 column, for consistency
    with python's usual slice syntax.

    >>> W[:,:0]

    will return all time steps, along with all `frame` data, but `data` will be empty (because the `:0` term selects
    everything before the 0th element).  Similarly,

    >>> W[:0,:0]

    is empty of all numerical data.

    """

    __num = 0  # Used to count number of Waveforms created

    def __init__(self, *args, **kwargs):
        """Initializer for WaveformBase object

        WaveformBase objects may be created in two ways.  First, by copying an existing WaveformBase object -- in
        which case the only parameter should be that object.  Second, by passing any of the (writable) attributes as
        keywords.

        In both cases, the last step in initialization is to check the validity of the result.  By default,
        this will raise an exception if the result is not valid.  An additional keyword parameter
        `override_exception_from_invalidity` may be set if this is not desired.  This may be necessary if only some
        of the data can be passed in to the initializer, for example.

        Keyword parameters
        ------------------
        t: float array, empty default
        frame : quaternion array, empty default
        data : 2-d complex array, empty default
        history : list of strings, empty default
            This is the list of strings prepended to the history, an additional line is appended, showing the call to
            this initializer.
        version_hist : list of pairs of strings, empty default
            Remains empty if waveform data is on version 0.
        frameType : int, defaults to 0 (UnknownFrameType)
            See scri.FrameNames for possible values
        dataType : int, defaults to 0 (UnknownDataType)
            See scri.DataNames for possible values
        r_is_scaled_out : bool, defaults to False
            Set to True if the data represented could approach a nonzero value at Scri
        m_is_scaled_out : bool, defaults to False
            Set to True if the data represented are dimensionless and in units where the total mass is 1
        override_exception_from_invalidity: bool, defaults to False
            If True, report any errors, but do not raise them.
        constructor_statement : str, optional
            If this is present, it will replace the default constructor statement added to the history.  It is
            prepended with a string of the form `'{0} = '.format(self)`, which prints the ID of the resulting object
            (unique to this session only).

        """
        original_kwargs = kwargs.copy()
        super().__init__(*args, **kwargs)  # to ensure proper calling in multiple inheritance
        override_exception_from_invalidity = kwargs.pop("override_exception_from_invalidity", False)
        self.__num = type(self).__num
        self.__history_depth__ = 0
        type(self).__num += 1  # Increment class's instance tracker
        if len(args) == 0:
            self.t = kwargs.pop("t", np.empty((0,), dtype=float))
            self.frame = kwargs.pop("frame", np.empty((0,), dtype=np.quaternion))
            self.data = kwargs.pop("data", np.empty((0, 0), dtype=complex))
            # Information about this object
            self.history = kwargs.pop("history", [])
            self.version_hist = kwargs.pop("version_hist", [])
            self.frameType = kwargs.pop("frameType", UnknownFrameType)
            self.dataType = kwargs.pop("dataType", UnknownDataType)
            self.r_is_scaled_out = kwargs.pop("r_is_scaled_out", False)
            self.m_is_scaled_out = kwargs.pop("m_is_scaled_out", False)
            if "constructor_statement" in kwargs:
                self._append_history("{} = {}".format(self, kwargs.pop("constructor_statement")))
            else:
                opts = np.get_printoptions()
                np.set_printoptions(threshold=6)
                self._append_history(
                    "{} = {}(**{})".format(self, type(self).__name__, pprint.pformat(original_kwargs, indent=4))
                )
                np.set_printoptions(**opts)
        elif len(args) == 1 and isinstance(args[0], type(self)):
            other = args[0]
            self.t = np.copy(other.t)
            self.frame = np.copy(other.frame)
            self.data = np.copy(other.data)
            # Information about this object
            self.history = other.history[:]
            self.version_hist = other.version_hist[:]
            self.frameType = other.frameType
            self.dataType = other.dataType
            self.r_is_scaled_out = other.r_is_scaled_out
            self.m_is_scaled_out = other.m_is_scaled_out
            self._append_history(["", "{} = {}({})".format(self, type(self).__name__, other)])
        else:
            raise ValueError(
                "Did not understand input arguments to `{}` constructor.\n".format(type(self).__name__)
                + "Note that explicit data values must be passed as keywords,\n"
                + "whereas objects to be copied must be passed as the sole argument."
            )
        hostname = socket.gethostname()
        cwd = os.getcwd()
        time = datetime.datetime.now().isoformat()
        self.__history_depth__ = 1
        self.ensure_validity(alter=True, assertions=(not override_exception_from_invalidity))
        self.__history_depth__ = 0
        self._append_history([f"hostname = {hostname}", f"cwd = {cwd}", f"datetime = {time}", version_info()], 1)
        if kwargs:
            warning = "\nIn `{}` initializer, unused keyword arguments:\n".format(type(self).__name__)
            warning += pprint.pformat(kwargs, indent=4)
            warnings.warn(warning)

    @waveform_alterations
    def ensure_validity(self, alter=True, assertions=False):
        """Try to ensure that the `WaveformBase` object is valid

        This tests various qualities of the WaveformBase's members that are frequently assumed throughout the code.
        If the optional argument `alter` is `True` (which is the default), this function tries to alter the
        WaveformBase in place to ensure validity.  Note that this is not always possible.  If that is the case,
        an exception may be raised.  For example, if the `t` member is not a one-dimensional array of floats,
        it is not clear what that data should be. Similarly, if the `t` and `data` members have mismatched
        dimensions, there is no way to resolve that automatically.

        Also note that this is almost certainly not be an exhaustive test of all assumptions made in the code.

        If the optional `assertions` argument is `True` (default is `False`), the first test that fails will raise an
        assertion error.

        """
        import numbers

        errors = []
        alterations = []

        if assertions:
            test = test_with_assertions
        else:
            test = test_without_assertions

        # Ensure that the various data are correct and compatible
        test(
            errors,
            isinstance(self.t, np.ndarray),
            "isinstance(self.t, np.ndarray) # type(self.t)={}".format(type(self.t)),
        )
        test(
            errors,
            self.t.dtype == np.dtype(float),
            f"self.t.dtype == np.dtype(float) # self.t.dtype={self.t.dtype}",
        )
        if alter and self.t.ndim == 2 and self.t.shape[1] == 1:
            self.t = self.t[:, 0]
            alterations += ["{0}.t = {0}.t[:,0]".format(self)]
        test(
            errors,
            not self.t.size or self.t.ndim == 1,
            f"not self.t.size or self.t.ndim==1 # self.t.size={self.t.size}; self.t.ndim={self.t.ndim}",
        )
        test(
            errors,
            self.t.size <= 1 or np.all(np.diff(self.t) > 0.0),
            "self.t.size<=1 or np.all(np.diff(self.t)>0.0) "
            "# self.t.size={}; max(np.diff(self.t))={}".format(
                self.t.size, (max(np.diff(self.t)) if self.t.size > 1 else np.nan)
            ),
        )
        test(errors, np.all(np.isfinite(self.t)), "np.all(np.isfinite(self.t))")

        if alter and self.frame is None:
            self.frame = np.empty((0,), dtype=np.quaternion)
            alterations += [f"{self}.frame = np.empty((0,), dtype=np.quaternion)"]
        test(
            errors,
            isinstance(self.frame, np.ndarray),
            "isinstance(self.frame, np.ndarray) # type(self.frame)={}".format(type(self.frame)),
        )
        if alter and self.frame.dtype == np.dtype(float):
            try:  # Might fail because of shape
                self.frame = quaternion.as_quat_array(self.frame)
                alterations += ["{0}.frame = quaternion.as_quat_array({0}.frame)".format(self)]
            except (AssertionError, ValueError):
                pass
        test(
            errors,
            self.frame.dtype == np.dtype(np.quaternion),
            f"self.frame.dtype == np.dtype(np.quaternion) # self.frame.dtype={self.frame.dtype}",
        )
        test(
            errors,
            self.frame.size <= 1 or self.frame.size == self.t.size,
            "self.frame.size<=1 or self.frame.size==self.t.size "
            "# self.frame.size={}; self.t.size={}".format(self.frame.size, self.t.size),
        )
        test(errors, np.all(np.isfinite(self.frame)), "np.all(np.isfinite(self.frame))")

        test(
            errors,
            isinstance(self.data, np.ndarray),
            "isinstance(self.data, np.ndarray) # type(self.data)={}".format(type(self.data)),
        )
        test(errors, self.data.ndim >= 1, f"self.data.ndim >= 1 # self.data.ndim={self.data.ndim}")
        test(
            errors,
            self.data.shape[0] == self.t.shape[0],
            "self.data.shape[0]==self.t.shape[0] "
            "# self.data.shape[0]={}; self.t.shape[0]={}".format(self.data.shape[0], self.t.shape[0]),
        )
        test(errors, np.all(np.isfinite(self.data)), "np.all(np.isfinite(self.data))")

        # Information about this object
        if alter and not self.history:
            self.history = [""]
            alterations += [f"{self}.history = ['']"]
        if alter and isinstance(self.history, str):
            self.history = self.history.split("\n")
            alterations += ["{0}.history = {0}.history.split('\n')".format(self)]
        test(
            errors,
            isinstance(self.history, list),
            "isinstance(self.history, list) # type(self.history)={}".format(type(self.history)),
        )
        test(
            errors,
            isinstance(self.history[0], str),
            "isinstance(self.history[0], str) # type(self.history[0])={}".format(type(self.history[0])),
        )
        test(
            errors,
            isinstance(self.frameType, numbers.Integral),
            "isinstance(self.frameType, numbers.Integral) # type(self.frameType)={}".format(type(self.frameType)),
        )
        test(errors, self.frameType in FrameType, f"self.frameType in FrameType # self.frameType={self.frameType}")
        test(
            errors,
            isinstance(self.dataType, numbers.Integral),
            "isinstance(self.dataType, numbers.Integral) # type(self.dataType)={}".format(type(self.dataType)),
        )
        test(errors, self.dataType in DataType, f"self.dataType in DataType # self.dataType={self.dataType}")
        test(
            errors,
            isinstance(self.r_is_scaled_out, bool),
            "isinstance(self.r_is_scaled_out, bool) # type(self.r_is_scaled_out)={}".format(type(self.r_is_scaled_out)),
        )
        test(
            errors,
            isinstance(self.m_is_scaled_out, bool),
            "isinstance(self.m_is_scaled_out, bool) # type(self.m_is_scaled_out)={}".format(type(self.m_is_scaled_out)),
        )
        test(
            errors,
            isinstance(self.num, numbers.Integral),
            "isinstance(self.num, numbers.Integral) # type(self.num)={}".format(type(self.num)),
        )

        if alterations:
            self._append_history(alterations)
            warnings.warn("The following alterations were made:\n\t" + "\n\t".join(alterations))
        if errors:
            warnings.warn("The following conditions were found to be incorrectly False:\n\t" + "\n\t".join(errors))
            return False

        self.__history_depth__ -= 1
        self._append_history("WaveformBase.ensure_validity" + f"({self}, alter={alter}, assertions={assertions})")

        return True

    @property
    def is_valid(self):
        return self.ensure_validity(alter=False, assertions=False)

    # Data sizes
    @property
    def n_data_sets(self):
        return int(np.prod(self.data.shape[1:]))

    @property
    def n_times(self):
        return self.t.shape[0]

    # Calculate weights
    @property
    def spin_weight(self):
        return SpinWeights[self.dataType]

    @property
    def conformal_weight(self):
        return ConformalWeights[self.dataType] + (-RScaling[self.dataType] if self.r_is_scaled_out else 0)

    @property
    def gamma_weight(self):
        """Non-conformal effect of a boost.

        This factor allows for mass-scaling, for example.  If the waveform describes `r*h/M`, for example,
        then `r` and `h` vary by the conformal weight, which depends on the direction; whereas `M` is a monopole,
        and thus cannot depend on the direction.  Instead, `M` simply obeys the standard formula, scaling with gamma.

        """
        return (MScaling[self.dataType] if self.m_is_scaled_out else 0) + (
            -RScaling[self.dataType] if (self.r_is_scaled_out and self.m_is_scaled_out) else 0
        )

    @property
    def r_scaling(self):
        return RScaling[self.dataType]

    @property
    def m_scaling(self):
        return MScaling[self.dataType]

    # Text descriptions
    @property
    def num(self):
        return self.__num

    @property
    def frame_type_string(self):
        return FrameNames[self.frameType]

    @property
    def data_type_string(self):
        return DataNames[self.dataType]

    @property
    def data_type_latex(self):
        return DataNamesLaTeX[self.dataType]

    @property
    def descriptor_string(self):
        """Create a simple string describing the content of the waveform

        This string will be suitable for file names.  For example, 'rMpsi4' or 'rhOverM'.  It uses the waveform's
        knowledge of itself, so if this is incorrect, the result will be incorrect.

        """
        if self.dataType == UnknownDataType:
            return self.data_type_string
        descriptor = ""
        if self.r_is_scaled_out:
            if RScaling[self.dataType] == 1:
                descriptor = "r"
            elif RScaling[self.dataType] > 1:
                descriptor = "r" + str(RScaling[self.dataType])
        if self.m_is_scaled_out:
            Mexponent = MScaling[self.dataType] - (RScaling[self.dataType] if self.r_is_scaled_out else 0)
            if Mexponent < -1:
                descriptor = descriptor + self.data_type_string + "OverM" + str(-Mexponent)
            elif Mexponent == -1:
                descriptor = descriptor + self.data_type_string + "OverM"
            elif Mexponent == 0:
                descriptor = descriptor + self.data_type_string
            elif Mexponent == 1:
                descriptor = descriptor + "M" + self.data_type_string
            elif Mexponent > 1:
                descriptor = descriptor + "M" + str(Mexponent) + self.data_type_string
        else:
            descriptor = descriptor + self.data_type_string
        return descriptor

    # Data simplifications
    @property
    def data_2d(self):
        return self.data.reshape((self.n_times, self.n_data_sets))

    @property
    def abs(self):
        return np.abs(self.data)

    @property
    def arg(self):
        return np.angle(self.data)

    @property
    def arg_unwrapped(self):
        return np.unwrap(np.angle(self.data), axis=0)

    def norm(self, take_sqrt=False, indices=slice(None, None, None)):
        """L2 norm of the waveform

        The optional arguments say whether to take the square-root of
        the norm at each time, and allow restriction to a slice of the
        data, respectively.

        """
        if indices == slice(None, None, None):
            n = np.empty((self.n_times,), dtype=float)
        else:
            n = np.empty((self.t[indices].shape[0],), dtype=float)
        if take_sqrt:
            complex_array_abs(self.data_2d[indices], n)
        else:
            complex_array_norm(self.data_2d[indices], n)
        return n

    def max_norm_index(self, skip_fraction_of_data=4):
        """Index of time step with largest norm

        The optional argument skips a fraction of the data.  The default is
        4, which means that it only searches the last three-fourths of the
        data for the max.  If 0 or 1 is input, this is ignored, and all the
        data is searched.

        """
        if skip_fraction_of_data == 0 or skip_fraction_of_data == 1:
            indices = slice(None, None, None)
            return np.argmax(self.norm(indices=indices))
        else:
            indices = slice(self.n_times // skip_fraction_of_data, None, None)
            return np.argmax(self.norm(indices=indices)) + (self.n_times // skip_fraction_of_data)

    def max_norm_time(self, skip_fraction_of_data=4):
        """Return time at which largest norm occurs in data

        See `help(max_norm_index)` for explanation of the optional argument.

        """
        return self.t[self.max_norm_index(skip_fraction_of_data=skip_fraction_of_data)]

    def compare(self, w_a, min_time_step=0.005, min_time=-3.0e300):
        """Return a waveform with differences between the two inputs

        This function simply subtracts the data in this waveform from the data
        in Waveform A, and finds the rotation needed to take this frame into frame A.
        Note that the waveform data are stored as complex numbers, rather than as
        modulus and phase.
        """
        from quaternion.means import mean_rotor_in_chordal_metric
        from scri.extrapolation import intersection
        import scri.waveform_modes

        if self.frameType != w_a.frameType:
            warning = (
                "\nWarning:"
                + "\n    This Waveform is in the "
                + self.frame_type_string
                + " frame,"
                + "\n    The Waveform in the argument is in the "
                + w_a.frame_type_string
                + " frame."
                + "\n    Comparing them probably does not make sense.\n"
            )
            warnings.warn(warning)

        if self.n_modes != w_a.n_modes:
            raise Exception(
                "Trying to compare waveforms with mismatched LM data."
                + "\nA.n_modes="
                + str(w_a.n_modes)
                + "\tB.n_modes()="
                + str(self.n_modes)
            )

        new_times = intersection(self.t, w_a.t)

        w_c = scri.waveform_modes.WaveformModes(
            t=new_times,
            data=np.zeros((new_times.shape[0], self.n_modes), dtype=self.data.dtype),
            history=[],
            version_hist=self.version_hist,
            frameType=self.frameType,
            dataType=self.dataType,
            r_is_scaled_out=self.r_is_scaled_out,
            m_is_scaled_out=self.m_is_scaled_out,
            ell_min=self.ell_min,
            ell_max=self.ell_max,
        )

        w_c.history += ["B.compare(A)\n"]
        w_c.history += ["### A.history.str():\n" + "".join(w_a.history)]
        w_c.history += ["### B.history.str():\n" + "".join(self.history)]
        w_c.history += ["### End of old histories from `compare`"]

        # Process the frame, depending on the sizes of the input frames
        if w_a.frame.shape[0] > 1 and self.frame.shape[0] > 1:
            # Find the frames interpolated to the appropriate times
            Aframe = quaternion.squad(w_a.frame, w_a.t, w_c.t)
            Bframe = quaternion.squad(self.frame, self.t, w_c.t)
            # Assign the data
            w_c.frame = Aframe * np.array([np.quaternion.inverse(v) for v in Bframe])
        elif w_a.frame.shape[0] == 1 and self.frame.shape[0] > 1:
            # Find the frames interpolated to the appropriate times
            Bframe = np.quaternion.squad(self.frame, self.t, w_c.t)
            # Assign the data
            w_c.frame.resize(w_c.n_times)
            w_c.frame = w_a.frame[0] * np.array([np.quaternion.inverse(v) for v in Bframe])
        elif w_a.frame.shape[0] > 1 and self.frame.shape[0] == 1:
            # Find the frames interpolated to the appropriate times
            Aframe = np.quaternion.squad(w_a.frame, w_a.t, w_c.t)
            # Assign the data
            w_c.frame.resize(w_c.n_times)
            w_c.frame = Aframe * np.quaternion.inverse(self.frame[0])
        elif w_a.frame.shape[0] == 1 and self.frame.shape[0] == 1:
            # Assign the data
            w_c.frame = np.array(w_a.frame[0] * np.quaternions.inverse(self.frame[0]))
        elif w_a.frame.shape[0] == 0 and self.frame.shape[0] == 1:
            # Assign the data
            w_c.frame = np.array(np.quaternions.inverse(self.frame[0]))
        elif w_a.frame.shape[0] == 1 and self.frame.shape[0] == 1:
            # Assign the data
            w_c.frame = np.array(w_a.frame[0])
        # else, leave the frame data empty

        # If the average frame rotor is closer to -1 than to 1, flip the sign
        if w_c.frame.shape[0] == w_c.n_times:
            R_m = mean_rotor_in_chordal_metric(w_c.frame, w_c.t)
            if quaternion.rotor_chordal_distance(R_m, -quaternion.one) < quaternion.rotor_chordal_distance(
                R_m, quaternion.one
            ):
                w_c.frame = -w_c.frame
        elif w_c.frame.shape[0] == 1:
            if quaternion.rotor_chordal_distance(w_c.frame[0], -quaternion.one) < quaternion.rotor_chordal_distance(
                w_c.frame[0], quaternion.one
            ):
                w_c.frame[0] = -w_c.frame[0]

        # Now loop over each mode filling in the waveform data
        for AMode in range(w_a.n_modes):
            # Assume that all the ell,m data are the same, but not necessarily in the same order
            BMode = self.index(w_a.LM[AMode][0], w_a.LM[AMode][1])
            # Initialize the interpolators for this data set
            # (Can't just re-view here because data are not contiguous)
            splineReA = CubicSpline(w_a.t, w_a.data[:, AMode].real)
            splineImA = CubicSpline(w_a.t, w_a.data[:, AMode].imag)
            splineReB = CubicSpline(self.t, self.data[:, BMode].real)
            splineImB = CubicSpline(self.t, self.data[:, BMode].imag)
            # Assign the data from the transition
            w_c.data[:, AMode] = (splineReA(w_c.t) - splineReB(w_c.t)) + 1j * (splineImA(w_c.t) - splineImB(w_c.t))

        return w_c

    @property
    def data_dot(self):
        return CubicSpline(self.t, self.data).derivative()(self.t)

    @property
    def data_ddot(self):
        return CubicSpline(self.t, self.data).derivative(2)(self.t)

    @property
    def data_int(self):
        return CubicSpline(self.t, self.data).antiderivative()(self.t)

    @property
    def data_iint(self):
        return CubicSpline(self.t, self.data).antiderivative(2)(self.t)

    # Data representations
    def _append_history(self, hist, additional_depth=0):
        """Add to the object's history log

        Input may be a single string or list of strings.  Any newlines will be split into separate strings.  Each
        such string is then prepended with a number of `#`s, indicating that the content of that line was called from
        within a member function, or is simply a piece of information relevant to the waveform.  The idea behind this
        is that the history should be -- as nearly as possible --  a script that could be run to reproduce the
        waveform, so the lines beginning with `#` would not be run.

        The number of `#`s is controlled by the object's `__history_depth__` field and the optional input to this
        function; their sum is the number prepended.  The user should never have to deal with this issue,
        but all member functions should increment the `__history_depth__` before calling another member function,
        and decrement it again as necessary before recording itself in the history.  Also, for any lines added just
        for informational purposes (e.g., the hostname, pwd, date, and versions added in `__init__`), this function
        should be called with `1` as the optional argument.

        """
        if not isinstance(hist, list):
            hist = [hist]
        self.history += [
            "# " * (self.__history_depth__ + additional_depth) + hist_line
            for hist_element in hist
            for hist_line in hist_element.split("\n")
        ]

    def __str__(self):
        # "The goal of __str__ is to be readable; the goal of __repr__ is to be unambiguous." --- stackoverflow
        return "{}_{}".format(type(self).__name__, self.num)

    def __repr__(self):
        # "The goal of __str__ is to be readable; the goal of __repr__ is to be unambiguous." --- stackoverflow
        from textwrap import dedent

        opts = np.get_printoptions()
        np.set_printoptions(threshold=6, linewidth=150, precision=6)
        rep = """
         {0}(
             t={1},
             frame={2},
             data={5},
             frameType={6}, dataType={7},
             r_is_scaled_out={8}, m_is_scaled_out={9})  # num = {10}"""
        rep = rep.format(
            type(self).__name__,
            str(self.t).replace("\n", "\n" + " " * 15),
            str(self.frame).replace("\n", "\n" + " " * 19),
            self.history,
            self.version_hist,
            str(self.data).replace("\n", "\n" + " " * 18),
            self.frameType,
            self.dataType,
            self.r_is_scaled_out,
            self.m_is_scaled_out,
            self.num,
        )
        np.set_printoptions(**opts)
        return dedent(rep)

    def __getstate__(self):
        """Get state of object for copying and pickling

        The only nontrivial operation is with quaternions, since they can't
        currently be pickled automatically.  We just view the frame array as
        a float array, and pickle as usual.

        Also, we remove the `num` value, because this will get reset
        properly on creation.

        """
        state = copy.deepcopy(self.__dict__)
        state["frame"] = quaternion.as_float_array(self.frame)
        return state

    def __setstate__(self, state):
        """Set state of object for copying and pickling

        The only nontrivial operation is with quaternions, since they can't
        currently be pickled automatically.  We just view the frame array as
        a float array, and unpickle as usual, then convert the float array
        back to a quaternion array.

        """
        new_num = self.__num
        old_num = state.get("_WaveformBase__num")
        self.__dict__.update(state)
        # Make sure to preserve auto-incremented num
        self.__num = new_num
        self.frame = quaternion.as_quat_array(self.frame)
        self._append_history(f"copied, deepcopied, or unpickled as {self}", 1)
        self._append_history("{} = {}".format(self, f"{self}".replace(str(self.num), str(old_num))))

    @waveform_alterations
    def deepcopy(self):
        """Return a deep copy of the object

        This is just an alias for `copy`, which is deep anyway.

        """
        W = self.copy()
        W.__history_depth__ -= 1
        W._append_history(f"{W} = {self}.deepcopy()")
        return W

    @waveform_alterations
    def copy(self):
        """Return a (deep) copy of the object

        Note that this also copies all members if the object is a subclass.  If you want a forgetful WaveformBase
        object, you can simply use the copy constructor.

        """
        W = type(self)()
        state = copy.deepcopy(self.__dict__)
        state.pop("_WaveformBase__num")
        W.__dict__.update(state)
        W.__history_depth__ -= 1
        W._append_history(f"{W} = {self}.copy()")
        return W

    @waveform_alterations
    def copy_without_data(self):
        """Return a copy of the object, with empty `t`, `frame`, and `data` fields

        Note that subclasses may override this to set some related data members.  For example,
        `WaveformModes.copy_without_data` sets the `ell_min` and `ell_max` fields appropriately.  If you wish to only
        skip `t`, `frame`, and `data`, you can simply use `WaveformBase.copy_without_data(W)`.  The latter is useful
        if, for example, you will only be making changes to those three fields, and want everything else preserved.

        Also note that some slicing operations can achieve similar -- but different -- goals.  For example,
        `w = w[:, :0]` will simply empty `data` and `ells`, without affecting the `time` and `frame`.

        """
        W = type(self)()
        state = copy.deepcopy(self.__dict__)
        state.pop("_WaveformBase__num")
        state.pop("t")
        state.pop("frame")
        state.pop("data")
        W.__dict__.update(state)
        W.__history_depth__ -= 1
        W._append_history(f"{W} = {self}.copy_without_data()")
        return W

    def _allclose(
        self, other, report_all=True, rtol=1e-10, atol=1e-10, compare_history_beginnings=False, exceptions=[]
    ):
        """Check that member data in two waveforms are the same

        For data sets (time, modes, etc.), the numpy function `np.allclose` is used, with the input tolerances.  See
        that function's documentation for more details.  The `*__num` datum is always ignored.  By default,
        the `history` is ignored, though this can be partially overridden -- in which case, the shortest subset of
        the histories is compared for exact equality.  This is probably only appropriate for the case where one
        waveform was created from the other.

        Parameters
        ----------
        other : object
            Another object subclassing WaveformBase to compare
        report_all: bool, optional
            Wait until all attributes have been checked (and reported on) before returning the verdict
        rtol : float, optional
            Relative tolerance to which to compare arrays (see np.allclose), defaults to 1e-10
        atol : float, optional
            Absolute tolerance to which to compare arrays (see np.allclose), defaults to 1e-10
        compare_history_beginnings: bool, optional
            Compare the shortest common part of the `history` fields for equality, defaults to False
        exceptions : list, optional
            Don't compare elements in this list, corresponding to keys in the object's `__dict__`, defaults to []

        """
        equality = True
        if not type(self) == type(other):  # not isinstance(other, self.__class__):
            warnings.warn("\n  (type(self)={}) != (type(other)={})".format(type(self), type(other)))
            equality = False
            if not report_all and not equality:
                return False
        for key, val in self.__dict__.items():
            if key.endswith("__num") or key in exceptions:
                continue
            elif key == "history":
                if compare_history_beginnings:
                    min_length = min(len(self.history), len(other.history))
                    if self.history[:min_length] != other.history[:min_length]:
                        warnings.warn("\n  `history` fields differ")
                        equality = False
            elif key == "version_hist":
                if self.version_hist != other.version_hist:
                    warnings.warn("\n  `version_hist` fields differ")
                    equality = False
            elif isinstance(val, np.ndarray):
                if val.dtype == np.quaternion:
                    if not np.allclose(
                        quaternion.as_float_array(val), quaternion.as_float_array(other.__dict__[key]), rtol, atol
                    ):
                        warnings.warn(f"\n  `{key}` fields differ")
                        equality = False
                elif not np.allclose(val, other.__dict__[key], rtol, atol):
                    warnings.warn(f"\n  `{key}` fields differ")
                    equality = False
            else:
                if not val == other.__dict__[key]:
                    warnings.warn(
                        "\n  (self.{0}={1}) != (other.{0}={2}) fields differ".format(key, val, other.__dict__[key])
                    )
                    equality = False
            if not report_all and not equality:
                return False
        return equality

    # Slicing
    @waveform_alterations
    def __getitem__(self, key):
        """Extract subsets of the data efficiently

        See the docstring of the WaveformBase class for examples.

        """
        W = WaveformBase.copy_without_data(self)

        # Remove trivial tuple structure first
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]

        # Now figure out which type of return is desired
        if isinstance(key, tuple) and 2 <= len(key) <= self.n_data_sets:
            # Return a subset of the data from a subset of times
            W.t = self.t[key[0]]
            W.frame = self.frame[key[0]]
            W.data = self.data[key]
        elif isinstance(key, slice) or isinstance(key, int):
            # Return complete data from a subset of times (key is slice), or
            # return complete data from a single instant in time (key is int)
            W.t = self.t[key]
            W.frame = self.frame[key]
            W.data = self.data[key]
        else:
            raise ValueError("Could not understand input `{}` (of type `{}`) ".format(key, type(key)))

        W.__history_depth__ -= 1
        W._append_history(f"{W} = {self}[{key}]")

        return W

    @waveform_alterations
    def interpolate(self, tprime):
        """Interpolate the frame and data onto the new set of time steps

        Note that only `t`, `frame`, and `data` are changed in this function.  If there is a corresponding data set
        in a subclass, for example, the subclass must override this function to set that data set -- though this
        function should probably be called to handle the ugly stuff.

        """
        # Copy the information fields, but not the data
        W = WaveformBase.copy_without_data(self)

        W.t = np.copy(tprime)
        W.frame = quaternion.squad(self.frame, self.t, W.t)
        W.data = np.empty((W.n_times,) + self.data.shape[1:], dtype=self.data.dtype)
        W.data_2d[:] = CubicSpline(self.t, self.data_2d.view(float))(W.t).view(complex)
        W.__history_depth__ -= 1
        W._append_history(f"{W} = {self}.interpolate({tprime})")
        return W

    @waveform_alterations
    def SI_units(self, current_unit_mass_in_solar_masses, distance_from_source_in_megaparsecs=100):
        """Assuming current quantities are in geometric units, convert to SI units

        This function assumes that the `dataType`, `r_is_scaled_out`, and `m_is_scaled_out` attributes are correct,
        then scales the amplitude and time data appropriately so that the data correspond to data that could be
        observed from a source with the given total mass at the given distance.

        Note that the curvature scalars will have units of s^-2, rather than the arguably more correct m^-2.  This
        seems to be more standard in numerical relativity.  The result can be divided by `scipy.constants.c**2`
        to give units of m^-2 if desired.

        Parameters
        ----------
        current_unit_mass_in_solar_masses : float
            Mass of the system in the data converted to solar masses
        distance_from_source_in_megaparsecs : float, optional
            Output will be waveform as observed from this distance, default=100 (Mpc)

        """
        if not self.r_is_scaled_out:
            warning = (
                "\nTrying to convert to SI units, the radius is supposedly not scaled out.\n"
                + "This seems to suggest that the data may already be in some units..."
            )
            warnings.warn(warning)
        if not self.m_is_scaled_out:
            warning = (
                "\nTrying to convert to SI units, the mass is supposedly not scaled out.\n"
                + "This seems to suggest that the data may already be in some units..."
            )
            warnings.warn(warning)

        M_in_meters = current_unit_mass_in_solar_masses * m_sun_in_meters  # m
        M_in_seconds = M_in_meters / speed_of_light  # s
        R_in_meters = distance_from_source_in_megaparsecs * (1e6 * parsec_in_meters)  # m
        R_over_M = R_in_meters / M_in_meters  # [dimensionless]

        # The radius scaling `r_scaling` is the number of factors of the dimensionless quantity `R_over_M` required
        # to keep the waveform asymptotically constant.  So, for example, h and Psi4 both have `r_scaling=1`.  The
        # mass scaling `m_scaling` is the number of factors of `M_in_meters` required to make the waveform
        # dimensionless, and does not account for the factors of mass in the radius scale.  The Newman-Penrose
        # quantities are curvature quantities, so they have dimensions 1/m^2, and thus have `m_scaling=2`.
        if self.r_is_scaled_out:
            if self.m_is_scaled_out:
                amplitude_scaling = (R_over_M ** -self.r_scaling) * (M_in_meters ** -self.m_scaling)
            else:
                amplitude_scaling = R_over_M ** -self.r_scaling
        else:
            if self.m_is_scaled_out:
                amplitude_scaling = M_in_meters ** -self.m_scaling
            else:
                amplitude_scaling = 1.0

        # Copy the information fields, but not the data
        W = WaveformBase.copy_without_data(self)

        if self.m_is_scaled_out:
            W.t = M_in_seconds * self.t  # s
        else:
            W.t = np.copy(self.t)  # supposedly already in the appropriate units...
        W.frame = np.copy(self.frame)
        W.data = amplitude_scaling * self.data

        W.m_is_scaled_out = False
        W.r_is_scaled_out = False

        W.__history_depth__ -= 1
        W._append_history(
            "{} = {}.SI_units(current_unit_mass_in_solar_masses={}, "
            "distance_from_source_in_megaparsecs={})".format(
                W, self, current_unit_mass_in_solar_masses, distance_from_source_in_megaparsecs
            )
        )

        return W
