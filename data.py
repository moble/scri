

class BondiData(object):
    def __init__(self, *args, **kwargs):
        pass

    @property
    def time_axis(self):
        """Return the dimension of self.data along which time varies"""
        return 0

    @property
    def datatype_axis(self):
        """Return the dimension of self.data along which the data type varies"""
        return 1

    @property
    def mode_axis(self):
        """Return the dimension of self.data along which modes are represented"""
        return 2

    @property
    def n_times(self):
        return self.u.size

    @property
    def n_modes(self):
        return self.data.shape[self.mode_axis]

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, uprm):
        self._u[:] = uprm
        return self._u

    @property
    def sigma_slice(self):
        s = [slice(None), slice(None), slice(None)]
        s[self.datatype_axis] = 0
        return s

    @property
    def sigma(self):
        return self.data[self.sigma_slice]

    @sigma.setter
    def sigma(self, sigmaprm):
        self.data[self.sigma_slice] = sigmaprm
        return self.data[self.sigma_slice]

    @property
    def psi4_slice(self):
        s = [slice(None), slice(None), slice(None)]
        s[self.datatype_axis] = 1
        return s

    @property
    def psi4(self):
        return self.data[self.psi4_slice]

    @psi4.setter
    def psi4(self, psi4prm):
        self.data[self.psi4_slice] = psi4prm
        return self.data[self.psi4_slice]

    @property
    def psi3_slice(self):
        s = [slice(None), slice(None), slice(None)]
        s[self.datatype_axis] = 2
        return s

    @property
    def psi3(self):
        return self.data[self.psi3_slice]

    @psi3.setter
    def psi3(self, psi3prm):
        self.data[self.psi3_slice] = psi3prm
        return self.data[self.psi3_slice]

    @property
    def psi2_slice(self):
        s = [slice(None), slice(None), slice(None)]
        s[self.datatype_axis] = 3
        return s

    @property
    def psi2(self):
        return self.data[self.psi2_slice]

    @psi2.setter
    def psi2(self, psi2prm):
        self.data[self.psi2_slice] = psi2prm
        return self.data[self.psi2_slice]

    @property
    def psi1_slice(self):
        s = [slice(None), slice(None), slice(None)]
        s[self.datatype_axis] = 4
        return s

    @property
    def psi1(self):
        return self.data[self.psi1_slice]

    @psi1.setter
    def psi1(self, psi1prm):
        self.data[self.psi1_slice] = psi1prm
        return self.data[self.psi1_slice]

    @property
    def psi0_slice(self):
        s = [slice(None), slice(None), slice(None)]
        s[self.datatype_axis] = 5
        return s

    @property
    def psi0(self):
        return self.data[self.psi0_slice]
        
    @psi0.setter
    def psi0(self, psi0prm):
        self.data[self.psi0_slice] = psi0prm
        return self.data[self.psi0_slice]

    @classmethod
    def from_initial_values(cls, u, ell_max=8, sigma=0.0, sigmadot=0.0, sigmaddot=0.0, psi2=0.0, psi1=0.0, psi0=0.0):
        """
        The initial-value formulation for Bondi gauge

        \Psi_4 = - \ddot{\bar{\sigma}}
        \Psi_3 = \eth \dot{\bar{\sigma}}
        Im[\Psi_2 + \eth^2\bar{\sigma} + \sigma \dot{\bar{\sigma}}] = 0
        \dot{\Psi2} = -\eth\Psi_3 + \sigma \Psi_4
        \dot{\Psi1} = -\eth\Psi_2 + 2\sigma \Psi_3
        \dot{\Psi0} = -\eth\Psi_1 + 3\sigma \Psi_2

        """
        import functools
        import numpy as np
        import spherical_functions as sf
        from scipy.interpolate import CubicSpline
        from quaternion.calculus import spline_evaluation, spline_indefinite_integral
        def broadcasts(*arrays):
            try:
                np.nditer(arrays)
                return True
            except ValueError:
                return False
        d = BondiData()
        d.u = u.copy()
        shape = [0, 0, 0]
        shape[d.time_axis] = d.n_times
        shape[d.datatype_axis] = 6
        shape[d.mode_axis] = sf.LM_total_size(0, ell_max)
        d.data = np.zeros(shape, dtype=complex)
        if np.ndim(sigma) == 2:
            # Assume this gives complete data, as a function of time and angle
            # If this is true, ignore sigmadot and sigmaddot
            d.sigma = sigma
            sigmadot = spline_evaluation(sigma, d.u, axis=0, spline_degree=4, derivative_order=1)
            sigmaddot = spline_evaluation(sigma, d.u, axis=0, spline_degree=5, derivative_order=2)
        elif np.ndim(sigma) == 1:
            # Assume this is just the angular dependence, which will be taken as constant in time
            # If this is true, check for sigmadot and sigmaddot
            d.sigma = sigma + d.u * (sigmadot + d.u * (sigmaddot / 2))
        else:
            raise ValueError(f"Input `sigma` must have 1 or 2 dimensions; it has {np.ndim(sigma)}")
        multiply_sigma = functools.partial(sf.multiply, d.sigma, 0, ell_max, 2)
        raise NotImplementedError("Need to implement conjugate and eth.")
        d.psi4 = -sf.conjugate(sigmaddot, s=2, ell_min=0, ell_max=ell_max)
        d.psi3 = eth(sf.conjugate(sigmadot, s=2, ell_min=0, ell_max=ell_max), s=-2)
        raise NotImplementedError("Adjust the initial value of psi2 to satisfy the mass-aspect condition.")
        d.psi2 = spline_indefinite_integral(-eth(d.psi3, s=-1) + multiply_sigma(d.psi4, 0, ell_max, -2)[0][..., :d.n_modes], d.u) + psi2
        d.psi1 = spline_indefinite_integral(-eth(d.psi2, s=0) + 2 * multiply_sigma(d.psi3, 0, ell_max, -1)[0][..., :d.n_modes], d.u) + psi1
        d.psi0 = spline_indefinite_integral(-eth(d.psi1, s=1) + 3 * multiply_sigma(d.psi2, 0, ell_max, 0)[0][..., :d.n_modes], d.u) + psi0
        return d


    @property
    def bondi_violations(self):
        """Compute violations of Bondi-gauge conditions

        The initial-value formulation for Bondi gauge establishes some
        relations that the data must satisfy:

        \Psi_4 = - \ddot{\bar{\sigma}}
        \Psi_3 = \eth \dot{\bar{\sigma}}
        Im[\Psi_2 + \eth^2\bar{\sigma} + \sigma \dot{\bar{\sigma}}] = 0
        \dot{\Psi2} = -\eth\Psi_3 + \sigma \Psi_4
        \dot{\Psi1} = -\eth\Psi_2 + 2\sigma \Psi_3
        \dot{\Psi0} = -\eth\Psi_1 + 3\sigma \Psi_2

        This function returns 6 arrays, corresponding to these 6
        equations, in which the right-hand side is subtracted from the
        left-hand side.  No norms are taken.

        """
        raise NotImplementedError()
