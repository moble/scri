import numpy as np
from scipy.interpolate import CubicSpline
import spinsfast


class ModesTimeSeries(spinsfast.Modes):
    """Object to store SWSH modes as functions of time

    This class subclasses the spinsfast.Modes class, but also tracks corresponding time values,
    allowing this class to have extra methods for interpolation, as well as differentiation and
    integration in time.

    NOTE: The time array is not copied; this class merely keeps a reference to the original time
    array.  If you change that array *in place* outside of this class, it changes inside of this
    class as well.  You can, of course, change the variable you used to label that array to point to
    some other quantity without affecting the time array stored in this class.

    """
    def __new__(cls, input_array, time, *args, **kwargs):
        input_array = np.asarray(input_array).view(complex)
        time = np.asarray(time).view(float)
        if input_array.ndim<2:
            raise ValueError(f"Input array must have at least 2 dimensions; it has {input_array.ndim}.")
        if time.ndim != 1:
            raise ValueError(f"Input time array must have exactly 1 dimension; it has {time.ndim}.")
        if input_array.shape[-2] != time.shape:
            raise ValueError("Second-to-last axis of input array must have same size as time array.\n            "
                             +f"Their shapes are {input_array.shape} and {time.shape}, respectively.")
        obj = super(cls, self).__new__(input_array, *args, **kwargs)
        obj._metadata['time'] = time
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        super(cls, self).__array_finalize__(self, obj)
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
        s = CubicSpline(self.u, self.view(np.ndarray), axis=-2)
        if derivative_order < 0:
            s = s.antiderivative(-derivative_order)
        elif 0 < derivative_order <= 3:
            s = s.derivative(derivative_order)
        result[:] = s(new_time)
        return type(self)(result, self.time, s=self.s, ell_max=self.ell_max)

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


class BondiData(object):
    def __init__(self, *args, **kwargs):
        pass

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

    @classmethod
    def from_initial_values(cls, time, ell_max=8, sigma=0.0, sigmadot=0.0, sigmaddot=0.0, psi2=0.0, psi1=0.0, psi0=0.0):
        """Construct Bondi data from sigma as a function of time and optional initial values

        The initial-value formulation for Bondi gauge is determined by these relations:

            \dot{\Psi_0} = -\eth\Psi_1 + 3\sigma \Psi_2
            \dot{\Psi_1} = -\eth\Psi_2 + 2\sigma \Psi_3
            \dot{\Psi_2} = -\eth\Psi_3 + \sigma \Psi_4
            \Psi_3 = \eth \dot{\bar{\sigma}}
            \Psi_4 = - \ddot{\bar{\sigma}}

        We also have a constraint on the initial value of Psi2:

            Im[\Psi_2] = -Im[\eth^2\bar{\sigma} + \sigma \dot{\bar{\sigma}}]

        """
        import functools
        import numpy as np
        import spherical_functions as sf
        from scipy.interpolate import CubicSpline
        from quaternion.calculus import spline_evaluation, spline_indefinite_integral

        d = BondiData()
        d._time = time.copy()
        shape = [6, d.n_times, sf.LM_total_size(0, ell_max)]
        data = np.zeros(shape, dtype=complex)
        d._psi0 = ModesTimeSeries(data[0], d.time, s=2, ell_max=ell_max)
        d._psi1 = ModesTimeSeries(data[1], d.time, s=1, ell_max=ell_max)
        d._psi2 = ModesTimeSeries(data[2], d.time, s=0, ell_max=ell_max)
        d._psi3 = ModesTimeSeries(data[3], d.time, s=-1, ell_max=ell_max)
        d._psi4 = ModesTimeSeries(data[4], d.time, s=-2, ell_max=ell_max)
        d._sigma = ModesTimeSeries(data[5], d.time, s=2, ell_max=ell_max)
        if np.ndim(sigma) == 2:
            # Assume this gives complete data, as a function of time and angle.
            # If this is true, ignore sigmadot and sigmaddot.
            d.sigma = sigma
            sigmadot = d.sigma.dot  # spline_evaluation(sigma, d.time, axis=0, spline_degree=4, derivative_order=1)
            sigmaddot = d.sigma.ddot  # spline_evaluation(sigma, d.time, axis=0, spline_degree=5, derivative_order=2)
        elif np.ndim(sigma) == 1:
            # Assume this is just the angular dependence, which will be taken as constant in time.
            # If this is true, assumes sigmadot and sigmaddot are constants in time, and just
            # integrates them.
            d.sigma = sigma + d.time * (sigmadot + d.time * (sigmaddot / 2))
        else:
            raise ValueError(f"Input `sigma` must have 1 or 2 dimensions; it has {np.ndim(sigma)}")
        d.psi4 = -sigmaddot.bar
        d.psi3 = sigmadot.bar.eth
        raise NotImplementedError("Adjust the initial value of psi2 to satisfy the mass-aspect condition.")
        d.psi2 = (-d.psi3.eth +     d.sigma * d.psi4).int + psi2
        d.psi1 = (-d.psi2.eth + 2 * d.sigma * d.psi3).int + psi1
        d.psi0 = (-d.psi1.eth + 3 * d.sigma * d.psi2).int + psi0
        return d

    @property
    def mass_aspect(self):
        return self.psi2 + self.sigma.bar.eth.eth + self.sigma * self.sigma.bar.dot

    def bondi_constraints(self, lhs=True, rhs=True):
        """Compute Bondi-gauge constraint equations

        Bondi gauge establishes some relations that the data must satisfy:

            \dot{\Psi0} = -\eth\Psi_1 + 3\sigma \Psi_2
            \dot{\Psi1} = -\eth\Psi_2 + 2\sigma \Psi_3
            \dot{\Psi2} = -\eth\Psi_3 + \sigma \Psi_4
            \Psi_3 = \eth \dot{\bar{\sigma}}
            \Psi_4 = - \ddot{\bar{\sigma}}
            Im[\Psi_2] = -Im[\eth^2\bar{\sigma} + \sigma \dot{\bar{\sigma}}]

        This function returns a 6-tuple of 2-tuples, corresponding to these 6 equations and their
        left- and right-hand sides.

        """
        return (
            self.bianchi_0(lhs, rhs),
            self.bianchi_1(lhs, rhs),
            self.bianchi_2(lhs, rhs),
            self.constraint_3(lhs, rhs),
            self.constraint_4(lhs, rhs),
            self.constraint_mass_aspect(lhs, rhs),
        )

    @property
    def bondi_violations(self):
        """Compute violations of Bondi-gauge constraints

        Bondi gauge establishes some relations that the data must satisfy:

            \dot{\Psi0} = -\eth\Psi_1 + 3\sigma \Psi_2
            \dot{\Psi1} = -\eth\Psi_2 + 2\sigma \Psi_3
            \dot{\Psi2} = -\eth\Psi_3 + \sigma \Psi_4
            \Psi_3 = \eth \dot{\bar{\sigma}}
            \Psi_4 = - \ddot{\bar{\sigma}}
            Im[\Psi_2] = -Im[\eth^2\bar{\sigma} + \sigma \dot{\bar{\sigma}}]

        This function returns a tuple of 6 arrays, corresponding to these 6 equations, in which the
        right-hand side is subtracted from the left-hand side.  No norms are taken.

        """
        constraints = self.bondi_constraints(True, True)
        return (lhs-rhs for (lhs, rhs) in constraints)

    @property
    def bondi_violation_norms(self):
        """Compute norms of violations of Bondi-gauge conditions

        Bondi gauge establishes some relations that the data must satisfy:

            \dot{\Psi0} = -\eth\Psi_1 + 3\sigma \Psi_2
            \dot{\Psi1} = -\eth\Psi_2 + 2\sigma \Psi_3
            \dot{\Psi2} = -\eth\Psi_3 + \sigma \Psi_4
            \Psi_3 = \eth \dot{\bar{\sigma}}
            \Psi_4 = - \ddot{\bar{\sigma}}
            Im[\Psi_2] = -Im[\eth^2\bar{\sigma} + \sigma \dot{\bar{\sigma}}]

        This function returns a tuple of 6 arrays, corresponding to the norms of these 6 equations,
        in which the right-hand side is subtracted from the left-hand side, and then the squared
        magnitude of that result is integrated over the sphere.  No integration is performed over
        time.

        """
        violations = self.bondi_violations
        return (v.norm() for v in violations)

    def bianchi_0(self, lhs=True, rhs=True):
        """Return the left- and/or right-hand sides of the Psi0 component of the Bianchi identity

        In Bondi coordinates, the Bianchi identities simplify, resulting in this expression (among
        others) for the time derivative of Psi0:

            \dot{\Psi0} = -\eth\Psi_1 + 3\sigma \Psi_2

        Parameters
        ==========
        lhs: bool [defaults to True]
            If True, return the left-hand side of the equation above
        rhs: bool [defaults to True]
            If True, return the right-hand side of the equation above

        If both parameters are True, a tuple with elements (lhs_value, rhs_value) is returned;
        otherwise just the requested value is returned.

        """
        if lhs:
            lhs_value = self.psi0.dot
        if rhs:
            rhs_value = -self.psi1.eth + 3 * self.sigma * self.psi2
        if lhs and rhs:
            return (lhs_value, rhs_value)
        elif lhs:
            return lhs_value
        elif rhs:
            return rhs_value

    def bianchi_1(self, lhs=True, rhs=True):
        """Return the left- and/or right-hand sides of the Psi1 component of the Bianchi identity

        In Bondi coordinates, the Bianchi identities simplify, resulting in this expression (among
        others) for the time derivative of Psi1:

            \dot{\Psi1} = -\eth\Psi_2 + 2\sigma \Psi_3

        Parameters
        ==========
        lhs: bool [defaults to True]
            If True, return the left-hand side of the equation above
        rhs: bool [defaults to True]
            If True, return the right-hand side of the equation above

        If both parameters are True, a tuple with elements (lhs_value, rhs_value) is returned;
        otherwise just the requested value is returned.

        """
        if lhs:
            lhs_value = self.psi1.dot
        if rhs:
            rhs_value = -self.psi2.eth + 2 * self.sigma * self.psi3
        if lhs and rhs:
            return (lhs_value, rhs_value)
        elif lhs:
            return lhs_value
        elif rhs:
            return rhs_value

    def bianchi_2(self, lhs=True, rhs=True):
        """Return the left- and/or right-hand sides of the Psi2 component of the Bianchi identity

        In Bondi coordinates, the Bianchi identities simplify, resulting in this expression (among
        others) for the time derivative of Psi2:

            \dot{\Psi2} = -\eth\Psi_3 + \sigma \Psi_4

        Parameters
        ==========
        lhs: bool [defaults to True]
            If True, return the left-hand side of the equation above
        rhs: bool [defaults to True]
            If True, return the right-hand side of the equation above

        If both parameters are True, a tuple with elements (lhs_value, rhs_value) is returned;
        otherwise just the requested value is returned.

        """
        if lhs:
            lhs_value = self.psi2.dot
        if rhs:
            rhs_value = -self.psi3.eth + self.sigma * self.psi4
        if lhs and rhs:
            return (lhs_value, rhs_value)
        elif lhs:
            return lhs_value
        elif rhs:
            return rhs_value

    def constraint_3(self, lhs=True, rhs=True):
        """Return the left- and/or right-hand sides of the Psi3 expression in Bondi gauge

        In Bondi coordinates, the value of Psi3 is given by a time derivative and an angular
        derivative of the (conjugate) shear:

            \Psi3 = \eth \dot{\bar{\sigma}}

        Parameters
        ==========
        lhs: bool [defaults to True]
            If True, return the left-hand side of the equation above
        rhs: bool [defaults to True]
            If True, return the right-hand side of the equation above

        If both parameters are True, a tuple with elements (lhs_value, rhs_value) is returned;
        otherwise just the requested value is returned.

        """
        if lhs:
            lhs_value = self.psi3
        if rhs:
            rhs_value = self.sigma.bar.dot.eth
        if lhs and rhs:
            return (lhs_value, rhs_value)
        elif lhs:
            return lhs_value
        elif rhs:
            return rhs_value

    def constraint_4(self, lhs=True, rhs=True):
        """Return the left- and/or right-hand sides of the Psi4 expression in Bondi gauge

        In Bondi coordinates, the value of Psi4 is given by two time derivatives of the (conjugate)
        shear:

            \Psi4 = - \ddot{\bar{\sigma}}

        Parameters
        ==========
        lhs: bool [defaults to True]
            If True, return the left-hand side of the equation above
        rhs: bool [defaults to True]
            If True, return the right-hand side of the equation above

        If both parameters are True, a tuple with elements (lhs_value, rhs_value) is returned;
        otherwise just the requested value is returned.

        """
        if lhs:
            lhs_value = self.psi4
        if rhs:
            rhs_value = -self.sigma.bar.ddot
        if lhs and rhs:
            return (lhs_value, rhs_value)
        elif lhs:
            return lhs_value
        elif rhs:
            return rhs_value

    def constraint_mass_aspect(self, lhs=True, rhs=True):
        """Return the left- and/or right-hand sides of the mass-aspect reality condition in Bondi gauge

        In Bondi coordinates, the Bondi mass aspect is always real, resulting in this relationship:

            \Im[\Psi2] = - \Im[\eth^2\bar{\sigma} + \sigma \dot{\bar{\sigma}}]

        Parameters
        ==========
        lhs: bool [defaults to True]
            If True, return the left-hand side of the equation above
        rhs: bool [defaults to True]
            If True, return the right-hand side of the equation above

        If both parameters are True, a tuple with elements (lhs_value, rhs_value) is returned;
        otherwise just the requested value is returned.

        """
        if lhs:
            lhs_value = np.imag(self.psi2.view(np.ndarray))
        if rhs:
            rhs_value = -np.imag(self.sigma.bar.eth.eth + self.sigma * self.sigma.bar.dot)
        if lhs and rhs:
            return (lhs_value, rhs_value)
        elif lhs:
            return lhs_value
        elif rhs:
            return rhs_value
