import copy
import numpy as np
from scipy.interpolate import CubicSpline
import spherical_functions


class ModesTimeSeries(spherical_functions.Modes):
    """Object to store SWSH modes as functions of time

    This class subclasses the spinsfast.Modes class, but also tracks corresponding time values,
    allowing this class to have extra methods for interpolation, as well as differentiation and
    integration in time.

    NOTE: The time array is not copied; this class merely keeps a reference to the original time
    array.  If you change that array *in place* outside of this class, it changes inside of this
    class as well.  You can, of course, change the variable you used to label that array to point to
    some other quantity without affecting the time array stored in this class.

    """
    def __new__(cls, input_array, *args, **kwargs):
        if len(args) > 2:
            raise ValueError("Only one positional argument may be passed")
        if len(args) == 1:
            kwargs['time'] = args[0]
        metadata = copy.copy(getattr(input_array, '_metadata', {}))
        metadata.update(**kwargs)
        input_array = np.asanyarray(input_array).view(complex)
        time = metadata.get('time', None)
        if time is None:
            raise ValueError('Time data must be specified as part of input array or as constructor parameter')
        time = np.asarray(time).view(float)
        if time.ndim != 1:
            raise ValueError(f"Input time array must have exactly 1 dimension; it has {time.ndim}.")
        if input_array.ndim == 0:
            input_array = input_array[np.newaxis, np.newaxis]
        elif input_array.ndim == 1:
            input_array = input_array[np.newaxis, :]
        elif input_array.shape[-2] != time.shape[0] and input_array.shape[-2] != 1:
            raise ValueError("Second-to-last axis of input array must have size 1 or same size as time array.\n            "
                             +f"Their shapes are {input_array.shape} and {time.shape}, respectively.")
        obj = sf.Modes(input_array, **kwargs).view(cls)
        obj._metadata['time'] = time
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        super().__array_finalize__(obj)
        if 'time' not in self._metadata:
            self._metadata['time'] = None

    @property
    def time(self):
        return self._metadata['time']

    @time.setter
    def time(self, new_time):
        self._metadata['time'][:] = new_time
        return self.time

    u = time

    t = time

    def interpolate(self, new_time, derivative_order=0, out=None):
        new_time = np.asarray(new_time)
        if new_time.ndim != 1:
            raise ValueError(f"New time array must have exactly 1 dimension; it has {new_time.ndim}.")
        new_shape = self.shape[:-2] + (new_time.size, self.shape[-1])
        if out is not None:
            out = np.asarray(out)
            if out.shape != new_shape:
                raise ValueError(f"Output array should have shape {new_shape} for consistency with new time array and modes array")
            if out.dtype != np.complex:
                raise ValueError(f"Output array should have dtype `complex`; it has dtype {out.dtype}")
        result = out or np.empty(new_shape, dtype=complex)
        if derivative_order > 3:
            raise ValueError(f"{type(self)} interpolation uses CubicSpline, and cannot take a derivative of order {derivative_order}")
        spline = CubicSpline(self.u, self.view(np.ndarray), axis=-2)
        if derivative_order < 0:
            spline = spline.antiderivative(-derivative_order)
        elif 0 < derivative_order <= 3:
            spline = spline.derivative(derivative_order)
        result[:] = spline(new_time)
        metadata = self._metadata.copy()
        metadata['time'] = new_time
        return type(self)(result, **metadata)

    def antiderivative(self, antiderivative_order=1):
        """Integrate modes with respect to time"""
        return self.interpolate(self.time, derivative_order=-antiderivative_order)

    def derivative(self, derivative_order=1):
        """Differentiate modes with respect to time"""
        return self.interpolate(self.time, derivative_order=derivative_order)

    @property
    def dot(self):
        """Differentiate modes once with respect to time"""
        return self.derivative()

    @property
    def ddot(self):
        """Differentiate modes twice with respect to time"""
        return self.derivative(2)

    @property
    def int(self):
        """Integrate modes once with respect to time"""
        return self.antiderivative()

    @property
    def iint(self):
        """Integrate modes twice with respect to time"""
        return self.antiderivative(2)
