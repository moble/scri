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

    Given these expressions, and the value of sigma as a function of time and direction, we can find
    the values of sigma and all the Weyl components for all time -- the Bondi data.  This function
    implements that algorithm.

    Parameters
    ==========
    cls: class
        Class to construct the AsymptoticBondiData object.  This function will also be used as a
        classmethod in that class, in which case this parameter will be passed automatically by
        calling this function as a method.
    time: array_like
        Times at which to compute the Bondi data.  Must be 1-dimensional.
    ell_max: int
        Maximum ell value to be stored in the data
    sigma: 0.0 or array_like [defaults to 0.0]
        This represents the value of sigma as a function of time and direction.  The input quantity
        must be able to broadcast against an array of shape (n_times, LM_total_size(0, ell_max)).
        If this object is 0- or 1-dimensional, it is assumed to be constant in time; only in this
        case are `sigmadot` and `sigmaddot` used.
    sigmadot: 0.0 or array_like [defaults to 0.0]
        This represents the time-derivative of sigma as a function of time and direction.  Note that
        this is ignored if `sigma` is a 2-dimensional array; instead, this quantity is computed by
        differentiating a spline of `sigma.  This must be 0- or 1-dimensional, and is assumed to
        represent the derivative at time 0.  (Or, if `sigmaddot` is 0, then the constant derivative
        at any time.)
    sigmaddot: 0.0 or array_like [defaults to 0.0]
        Just like `sigmadot`, except for the second derivative.
    psi2: 0.0 or array_like [defaults to 0.0]
        This represents the initial value of the psi2 field.  Its imaginary part is determined by
        the condition that the mass aspect must be real-valued, so any imaginary part in the input
        is discarded.  If array_like, this parameter must be 1-dimensional and have size
        LM_total_size(0, ell_max).
    psi1: 0.0 or array_like [defaults to 0.0]
        This represents the initial value of the psi1 field.  If array_like, this parameter must be
        1-dimensional and have size LM_total_size(0, ell_max).
    psi0: 0.0 or array_like [defaults to 0.0]
        This represents the initial value of the psi0 field.  If array_like, this parameter must be
        1-dimensional and have size LM_total_size(0, ell_max).

    """
    import numpy as np
    from .. import ModesTimeSeries

    def asany_atleast2d_complex(a):
        a = np.asanyarray(a) + 0j
        while np.ndim(a) < 2:
            a = a[np.newaxis, ...]
        return a

    psi2 = asany_atleast2d_complex(psi2)
    psi1 = asany_atleast2d_complex(psi1)
    psi0 = asany_atleast2d_complex(psi0)

    # Construct the empty container
    abd = cls(time, ell_max, multiplication_truncator=max)

    # Evaluate sigma and derivatives
    if np.ndim(sigma) == 0 or np.ndim(sigma) == 1:
        # Assume this is just the angular dependence, which will be taken as constant in time.
        # If this is true, assumes sigmadot and sigmaddot are constants in time, and just
        # integrates them.
        # sigma = asany_atleast2d_complex(sigma)
        # sigmadot = asany_atleast2d_complex(sigmadot)
        # sigmaddot = asany_atleast2d_complex(sigmaddot)
        # abd.sigma = sigma
        # abd.sigma = abd.sigma + abd.time * (sigmadot + abd.time * (sigmaddot / 2))
        sigma = ModesTimeSeries(sigma+0j, abd.time, spin_weight=2)
        sigmadot = ModesTimeSeries(sigmadot+0j, abd.time, spin_weight=2)
        sigmaddot = ModesTimeSeries(sigmaddot+0j, abd.time, spin_weight=2)
        abd.sigma = sigma + abd.time * (sigmadot + abd.time * (sigmaddot / 2))
    elif np.ndim(sigma) == 2:
        # Assume this gives complete data, as a function of time and angle.
        # If this is true, ignore sigmadot and sigmaddot.
        abd.sigma = sigma
        sigmadot = abd.sigma.dot
        sigmaddot = abd.sigma.ddot
    else:
        raise ValueError(f"Input `sigma` must have 1 or 2 dimensions; it has {np.ndim(sigma)}")

    # Adjust the initial value of psi2 to satisfy the mass-aspect condition
    sigma_initial = abd.sigma[..., 0, :]
    sigma_bar_dot_initial = abd.sigma.bar.dot[..., 0, :]
    psi2 = (
        ModesTimeSeries(psi2, abd.time, spin_weight=0).real
        - (sigma_initial.bar.eth.eth + sigma_initial * sigma_bar_dot_initial).imag
    )

    # Compute the Weyl components
    abd.psi4 = -sigmaddot.bar
    abd.psi3 = sigmadot.bar.eth
    abd.psi2 = (-abd.psi3.eth +     abd.sigma * abd.psi4).int + psi2
    abd.psi1 = (-abd.psi2.eth + 2 * abd.sigma * abd.psi3).int + psi1
    abd.psi0 = (-abd.psi1.eth + 3 * abd.sigma * abd.psi2).int + psi0

    return abd
