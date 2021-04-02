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
            kwargs["time"] = args[0]
        metadata = copy.copy(getattr(input_array, "_metadata", {}))
        metadata.update(**kwargs)
        input_array = np.asanyarray(input_array).view(complex)
        time = metadata.get("time", None)
        if time is None:
            raise ValueError("Time data must be specified as part of input array or as constructor parameter")
        time = np.asarray(time).view(float)
        if time.ndim != 1:
            raise ValueError(f"Input time array must have exactly 1 dimension; it has {time.ndim}.")
        if input_array.ndim == 0:
            input_array = input_array[np.newaxis, np.newaxis]
        elif input_array.ndim == 1:
            input_array = input_array[np.newaxis, :]
        elif input_array.shape[-2] != time.shape[0] and input_array.shape[-2] != 1:
            raise ValueError(
                "Second-to-last axis of input array must have size 1 or same size as time array.\n            "
                f"Their shapes are {input_array.shape} and {time.shape}, respectively."
            )
        obj = spherical_functions.Modes(input_array, **kwargs).view(cls)
        obj._metadata["time"] = time
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        super().__array_finalize__(obj)
        if "time" not in self._metadata:
            self._metadata["time"] = None

    @property
    def time(self):
        return self._metadata["time"]

    @time.setter
    def time(self, new_time):
        self._metadata["time"][:] = new_time
        return self.time

    @property
    def n_times(self):
        return self.time.size

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
                raise ValueError(
                    f"Output array should have shape {new_shape} for consistency with new time array and modes array"
                )
            if out.dtype != complex:
                raise ValueError(f"Output array should have dtype `complex`; it has dtype {out.dtype}")
        result = out or np.empty(new_shape, dtype=complex)
        if derivative_order > 3:
            raise ValueError(
                f"{type(self)} interpolation uses CubicSpline, and cannot take a derivative of order {derivative_order}"
            )
        spline = CubicSpline(self.u, self.view(np.ndarray), axis=-2)
        if derivative_order < 0:
            spline = spline.antiderivative(-derivative_order)
        elif 0 < derivative_order <= 3:
            spline = spline.derivative(derivative_order)
        result[:] = spline(new_time)
        metadata = self._metadata.copy()
        metadata["time"] = new_time
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

    @property
    def LM(self):
        return spherical_functions.LM_range(self.ell_min, self.ell_max)

    @property
    def eth_GHP(self):
        """Raise spin-weight with GHP convention"""
        return self.eth / np.sqrt(2)

    @property
    def ethbar_GHP(self):
        """Lower spin-weight with GHP convention"""
        return self.ethbar / np.sqrt(2)

    def grid_multiply(self, mts, **kwargs):
        """Compute mode weights of the product of two functions

        This will compute the values of `self` and `mts` on a grid, multiply the grid
        values together, and then return the mode coefficients of the product.  This
        takes less time and memory compared to the `SWSH_modes.Modes.multiply()`
        function, at the risk of introducing aliasing effects if `working_ell_max` is
        too small.

        Parameters
        ----------
        self: ModesTimeSeries
            One of the quantities to multiply.
        mts: ModesTimeSeries
            The quantity to multiply with 'self'.
        working_ell_max: int, optional
            The value of ell_max to be used to define the computation grid. The
            number of theta points and the number of phi points are set to
            2*working_ell_max+1. Defaults to (self.ell_max + mts.ell_max).
        output_ell_max: int, optional
            The value of ell_max in the output mts object. Defaults to self.ell_max.

        """
        import spinsfast
        import spherical_functions as sf
        from spherical_functions import LM_index

        output_ell_max = kwargs.pop("output_ell_max", self.ell_max)
        working_ell_max = kwargs.pop("working_ell_max", self.ell_max + mts.ell_max)
        n_theta = n_phi = 2 * working_ell_max + 1

        if self.n_times != mts.n_times or not np.equal(self.t, mts.t).all():
            raise ValueError("The time series of objects to be multiplied must be the same.")

        # Transform to grid representation
        self_grid = spinsfast.salm2map(
            self.ndarray, self.spin_weight, lmax=self.ell_max, Ntheta=n_theta, Nphi=n_phi
        )
        mts_grid = spinsfast.salm2map(
            mts.ndarray, mts.spin_weight, lmax=mts.ell_max, Ntheta=n_theta, Nphi=n_phi
        )

        product_grid = self_grid * mts_grid
        product_spin_weight = self.spin_weight + mts.spin_weight

        # Transform back to mode representation
        product = spinsfast.map2salm(product_grid, product_spin_weight, lmax=working_ell_max)

        # Convert product ndarray to a ModesTimeSeries object
        product = product[:, : LM_index(output_ell_max, output_ell_max, 0) + 1]
        product = ModesTimeSeries(
            sf.SWSH_modes.Modes(
                product,
                spin_weight=product_spin_weight,
                ell_min=0,
                ell_max=output_ell_max,
                multiplication_truncator=max,
            ),
            time=self.t,
        )
        return product
