@classmethod
def from_initial_values(cls, time, ell_max=8, sigma0=0.0, sigmadot0=0.0, sigmaddot0=0.0, psi2=0.0, psi1=0.0, psi0=0.0):
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
    sigma0: 0.0 or array_like [defaults to 0.0]
        This represents the value of sigma as a function of time and direction.  The input quantity
        must be able to broadcast against an array of shape (n_times, LM_total_size(0, ell_max)).
        If this object is 0- or 1-dimensional, it is assumed to be the value at time u=0; only in
        this case are `sigmadot0` and `sigmaddot0` used.
    sigmadot0: 0.0 or array_like [defaults to 0.0]
        This represents the time-derivative of sigma as a function of time and direction.  Note that
        this is ignored if `sigma0` is a 2-dimensional array; instead, this quantity is computed by
        differentiating a spline of `sigma0`.  This must be 0- or 1-dimensional, and is assumed to
        represent the derivative at time 0.  (Or, if `sigmaddot0` is 0, then the constant derivative
        at any time.)
    sigmaddot0: 0.0 or array_like [defaults to 0.0]
        Just like `sigmadot0`, except for the second derivative.  Note that this represents the
        second derivative only at time u=0; for all other times we expand as a Taylor series if this
        is used.
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

    def adjust_psi2_imaginary_part(psi2, abd):
        # Adjust the initial value of psi2 to satisfy the mass-aspect condition
        sigma_initial = abd.sigma[..., 0, :]
        sigma_bar_dot_initial = abd.sigma.bar.dot[..., 0, :]
        return (
            ModesTimeSeries(psi2, abd.time, spin_weight=0).real
            - (sigma_initial.bar.eth.eth + sigma_initial * sigma_bar_dot_initial).imag
        )

    psi2 = asany_atleast2d_complex(psi2)
    psi1 = asany_atleast2d_complex(psi1)
    psi0 = asany_atleast2d_complex(psi0)

    # Construct the empty container
    abd = cls(time, ell_max, multiplication_truncator=max)

    # Evaluate sigma and derivatives
    if np.ndim(sigma0) == 0 or np.ndim(sigma0) == 1:
        # Assume this is just the angular dependence, which will be taken as constant in time.  If
        # this is true, assumes sigmadot0 and sigmaddot0 are constants in time, and just integrates
        # them.
        u = abd.time
        ð = lambda x: x.eth
        conjugate = lambda x: x.bar
        sigma0 = ModesTimeSeries(sigma0+0j, u, spin_weight=2, multiplication_truncator=max)
        sigmadot0 = ModesTimeSeries(sigmadot0+0j, u, spin_weight=2, multiplication_truncator=max)
        sigmaddot0 = ModesTimeSeries(sigmaddot0+0j, u, spin_weight=2, multiplication_truncator=max)
        abd.sigma = sigma0 + u * (sigmadot0 + u * (sigmaddot0 / 2))
        # ψ₄ = -∂²σ̄/∂u²
        abd.psi4 = -sigmaddot0.bar
        # ψ₃ = ð ∂σ̄/∂u
        ψ30 = sigmadot0.bar.eth
        ψ31 = sigmaddot0.bar.eth
        abd.psi3 = ψ30 + u * ψ31
        # ψ₂ = ∫ (-ðψ₃ + σψ₄) du
        #    = -ð²σ̄ + ∫ σ du ψ₄
        #    = -ð²σ̄ + (uσ₀ + u²σ₁/2 + u³σ₂/6) ψ₄
        #    = -ð²σ̄₀ + u(σ₀ψ₄-ð²σ̄₁) + u²(σ₁ψ₄-ð²σ̄₂)/2 + u³(σ₂ψ₄/6)
        ψ20 = -sigma0.bar.eth.eth + adjust_psi2_imaginary_part(psi2, abd)
        ψ21 = sigma0 * abd.psi4 - sigmadot0.bar.eth.eth
        ψ22 = (sigmadot0 * abd.psi4 - sigmaddot0.bar.eth.eth) / 2
        ψ23 = sigmaddot0 * abd.psi4 / 6
        abd.psi2 = ψ20 + u * (ψ21 + u * (ψ22 + u * ψ23))
        # ψ₁ = ∫ (-ðψ₂ + σψ₃) du
        #    = -ð(uψ₂₀+u²ψ₂₁/2+u³ψ₂₂/3+u⁴ψ₂₃/4) + ∫ (σ₀+uσ₁+u²σ₂/2)(ψ₃₀+uψ₃₁) du
        #    = -ð(uψ₂₀+u²ψ₂₁/2+u³ψ₂₂/3+u⁴ψ₂₃/4) + ∫ (σ₀ψ₃₀+uσ₁ψ₃₀+u²σ₂ψ₃₀/2+uσ₀ψ₃₁+u²σ₁ψ₃₁+u³σ₂ψ₃₁/2) du
        #    = -ð(uψ₂₀+u²ψ₂₁/2+u³ψ₂₂/3+u⁴ψ₂₃/4) + ∫ (σ₀ψ₃₀+u(σ₁ψ₃₀+σ₀ψ₃₁)+u²(σ₂ψ₃₀/2+σ₁ψ₃₁)+u³(σ₂ψ₃₁/2)) du
        #    = -uðψ₂₀-u²ðψ₂₁/2-u³ðψ₂₂/3-u⁴ðψ₂₃/4 + uσ₀ψ₃₀+u²(σ₁ψ₃₀+σ₀ψ₃₁)/2+u³(σ₂ψ₃₀/2+σ₁ψ₃₁)/3+u⁴(σ₂ψ₃₁/2)/4
        #    = u(σ₀ψ₃₀-ðψ₂₀) + u²(σ₁ψ₃₀+σ₀ψ₃₁-ðψ₂₁)/2 + u³(σ₂ψ₃₀/2+σ₁ψ₃₁-ðψ₂₂)/3 + u⁴(σ₂ψ₃₁/2-ðψ₂₃)/4
        ψ10 = ModesTimeSeries(psi1, u, spin_weight=1, multiplication_truncator=max)
        ψ11 = sigma0 * ψ30 - ψ20.eth
        ψ12 = (sigmadot0 * ψ30 + sigma0 * ψ31 - ψ21.eth) / 2
        ψ13 = (sigmaddot0 * ψ30 / 2 + sigmadot0 * ψ31 - ψ22.eth) / 3
        ψ14 = (sigmaddot0 * ψ31 / 2 - ψ23.eth) / 4
        abd.psi1 = ψ10 + u * (ψ11 + u * (ψ12 + u * (ψ13 + u * ψ14)))
        # ψ₀ = ∫ (-ðψ₁ + σψ₂) du
        #    = -ð(uψ₁₀+u²ψ₁₁/2+u³ψ₁₂/3+u⁴ψ₁₃/4+u⁵ψ₁₄/5) + ∫ (σ₀+uσ₁+u²σ₂/2)(ψ₂₀+uψ₂₁+u²ψ₂₂+u³ψ₂₃) du
        #    = -ð(uψ₁₀+u²ψ₁₁/2+u³ψ₁₂/3+u⁴ψ₁₃/4+u⁵ψ₁₄/5) + ∫ (σ₀ψ₂₀+uσ₁ψ₂₀+u²σ₂ψ₂₀/2+uσ₀ψ₂₁+u²σ₁ψ₂₁+u³σ₂ψ₂₁/2+u²σ₀ψ₂₂+u³σ₁ψ₂₂+u⁴σ₂ψ₂₂/2) du
        #    = -ð(uψ₁₀+u²ψ₁₁/2+u³ψ₁₂/3+u⁴ψ₁₃/4+u⁵ψ₁₄/5) + ∫ (σ₀ψ₂₀+u(σ₁ψ₂₀+σ₀ψ₂₁)+u²(σ₂ψ₂₀/2+σ₁ψ₂₁+σ₀ψ₂₂)+u³(σ₂ψ₂₁/2+σ₁ψ₂₂)+u⁴(σ₂ψ₂₂/2)) du
        #    = -uðψ₁₀-u²ðψ₁₁/2-u³ðψ₁₂/3-u⁴ðψ₁₃/4-u⁵ðψ₁₄/5 + uσ₀ψ₂₀+u²(σ₁ψ₂₀+σ₀ψ₂₁)/2+u³(σ₂ψ₂₀/2+σ₁ψ₂₁+σ₀ψ₂₂)/3+u⁴(σ₂ψ₂₁/2+σ₁ψ₂₂)/4+u⁵(σ₂ψ₂₂/2)/5
        #    = u(σ₀ψ₂₀-ðψ₁₀) + u²(σ₁ψ₂₀+σ₀ψ₂₁-ðψ₁₁)/2 + u³(σ₂ψ₂₀/2+σ₁ψ₂₁+σ₀ψ₂₂-ðψ₁₂)/3 + u⁴(σ₂ψ₂₁/2+σ₁ψ₂₂-ðψ₁₃)/4 + u⁵(σ₂ψ₂₂/2 -ðψ₁₄/5)/5
        ψ00 = ModesTimeSeries(psi0, u, spin_weight=2, multiplication_truncator=max)
        ψ01 = sigma0 * ψ20 - ψ10.eth
        ψ02 = (sigmadot0 * ψ20 + sigma0 * ψ21 - ψ11.eth) / 2
        ψ03 = (sigmaddot0 * ψ20 / 2 + sigmadot0 * ψ21 + sigma0 * ψ22 - ψ12.eth) / 3
        ψ04 = (sigmaddot0 * ψ21 / 2 + sigmadot0 * ψ22 + sigma0 * ψ23 - ψ13.eth) / 4
        ψ05 = (sigmaddot0 * ψ22 / 2 + sigmadot0 * ψ23 - ψ14.eth) / 5
        ψ06 = sigmaddot0 * ψ23 / 12
        abd.psi0 = ψ00 + u * (ψ01 + u * (ψ02 + u * (ψ03 + u * (ψ04 + u * (ψ05 + u * ψ06)))))
    elif np.ndim(sigma0) == 2:
        # Assume this gives complete data, as a function of time and angle.
        # If this is true, ignore sigmadot0 and sigmaddot0.
        abd.sigma = sigma0
        # ψ₄ = -∂²σ̄/∂u²
        abd.psi4 = -abd.sigma.ddot.bar
        # ψ₃ = ð ∂σ̄/∂u
        abd.psi3 = abd.sigma.dot.bar.eth
        # ψ₂ = ∫ (-ðψ₃ + σψ₄) du
        abd.psi2 = (-abd.psi3.eth +     abd.sigma * abd.psi4).int + adjust_psi2_imaginary_part(psi2, abd)
        # ψ₁ = ∫ -ðψ₂ + σψ₃ du
        abd.psi1 = (-abd.psi2.eth + 2 * abd.sigma * abd.psi3).int + psi1
        # ψ₀ = ∫ -ðψ₁ + σψ₂ dt
        abd.psi0 = (-abd.psi1.eth + 3 * abd.sigma * abd.psi2).int + psi0
    else:
        raise ValueError(f"Input `sigma0` must have 1 or 2 dimensions; it has {np.ndim(sigma0)}")

    # Compute the Weyl components

    return abd
