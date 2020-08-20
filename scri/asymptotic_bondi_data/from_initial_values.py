@classmethod
def from_initial_values(cls, time, ell_max=8, sigma0=0.0, sigmadot0=0.0, sigmaddot0=0.0, psi2=0.0, psi1=0.0, psi0=0.0):
    """Construct Bondi data from sigma as a function of time and optional initial values

    The initial-value formulation for Bondi gauge is determined by these relations:

        ψ̇₀ = ðψ₁ + 3 σ ψ₂
        ψ̇₁ = ðψ₂ + 2 σ ψ₃
        ψ̇₂ = ðψ₃ + 1 σ ψ₄
        ψ₃ = -∂ðσ̄/∂u
        ψ₄ = -∂²σ̄/∂u²

    We also have a constraint on the initial value of ψ₂:

        Im[ψ₂] = -Im[ð²σ̄ + σ ∂σ̄/∂u]

    Given these expressions, and the value of sigma as a function of time and
    direction, we can find the values of sigma and all the Weyl components for all
    time -- the Bondi data.  This function implements that algorithm.

    Note that the interpretation of the input mode weights depends on the dimension
    of `sigma0`.  If it has dimension 0 or 1, the problem is integrated exactly,
    and the rest of the inputs are assumed to represent the corresponding values at
    time u=0.  If it has dimension 2, `sigma0` is interpreted as a function of
    time, and the rest of the problem is integrated numerically via splines, with
    `psi2`, `psi1`, and `psi0` representing the corresponding values at time
    u=`time[0]` (the first element of the input `time` array); `sigmadot0` and
    `sigmaddot0` are ignored in this case.

    Parameters
    ==========
    cls: class
        Class to construct the AsymptoticBondiData object.  This function will also
        be used as a classmethod in that class, in which case this parameter will
        be passed automatically by calling this function as a method.
    time: array_like
        Times at which to compute the Bondi data.  Must be 1-dimensional.
    ell_max: int
        Maximum ell value to be stored in the data
    sigma0: 0.0 or array_like [defaults to 0.0]
        This represents the value of sigma as a function of time and direction, or
        simply its initial value (see discussion above).  The input quantity must
        broadcast against an array of shape (n_times, LM_total_size(0, ell_max)).
    sigmadot0: 0.0 or array_like [defaults to 0.0]
        This represents the time-derivative of sigma as a function of time and
        direction.  Note that this is ignored if `sigma0` is a 2-dimensional array;
        instead, this quantity is computed by differentiating a spline of `sigma0`.
        This must be 0- or 1-dimensional, and is assumed to represent the
        derivative at time 0.  (Or, if `sigmaddot0` is 0, then the constant
        derivative at any time.)
    sigmaddot0: 0.0 or array_like [defaults to 0.0]
        Just like `sigmadot0`, except for the second derivative.  Note that this
        represents the second derivative only at time u=0; for all other times we
        expand as a Taylor series if this is used.
    psi2: 0.0 or array_like [defaults to 0.0]
        This represents the initial value of the psi2 field.  Its imaginary part is
        determined by the condition that the mass aspect must be real-valued, so
        any imaginary part in the input is discarded.  If array_like, this
        parameter must be 1-dimensional and have size LM_total_size(0, ell_max).
    psi1: 0.0 or array_like [defaults to 0.0]
        This represents the initial value of the psi1 field.  If array_like, this
        parameter must be 1-dimensional and have size LM_total_size(0, ell_max).
    psi0: 0.0 or array_like [defaults to 0.0]
        This represents the initial value of the psi0 field.  If array_like, this
        parameter must be 1-dimensional and have size LM_total_size(0, ell_max).

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
    if np.ndim(sigma0) == 0 or np.ndim(sigma0) == 1:
        # Assume this is just the angular dependence, which will be taken as
        # constant in time.  If this is true, assumes sigmadot0 and sigmaddot0
        # are constants in time, and just integrates them.
        u = abd.time
        ð = lambda x: x.eth_GHP
        conjugate = lambda x: x.bar
        σ_0 = ModesTimeSeries(asany_atleast2d_complex(sigma0), u, spin_weight=2, multiplication_truncator=max)
        σ_1 = ModesTimeSeries(asany_atleast2d_complex(sigmadot0), u, spin_weight=2, multiplication_truncator=max)
        σ_2 = ModesTimeSeries(asany_atleast2d_complex(sigmaddot0 / 2), u, spin_weight=2, multiplication_truncator=max)
        abd.sigma = u * (u * σ_2 + σ_1) + σ_0
        # ψ₄ = -∂²σ̄/∂u²
        ψ4_0 = -2 * conjugate(σ_2)
        abd.psi4 = ψ4_0
        # ψ₃ = -ð ∂σ̄/∂u
        ψ3_0 = -ð(conjugate(σ_1))
        ψ3_1 = -2 * ð(conjugate(σ_2))
        abd.psi3 = u * ψ3_1 + ψ3_0
        # ψ₂ = ∫ (ðψ₃ + σψ₄) du
        ψ2_0 = (
            ModesTimeSeries(psi2, u, spin_weight=0, multiplication_truncator=max).real
            - 1j * (σ_0.bar.eth_GHP.eth_GHP + σ_0 * σ_1.bar).imag
        )
        ψ2_1 = σ_0 * ψ4_0 + ð(ψ3_0)
        ψ2_2 = (σ_1 * ψ4_0 + ð(ψ3_1)) / 2
        ψ2_3 = (1 / 3) * σ_2 * ψ4_0
        abd.psi2 = u * (u * (u * ψ2_3 + ψ2_2) + ψ2_1) + ψ2_0
        # ψ₁ = ∫ (ðψ₂ + 2σψ₃) du
        ψ1_0 = ModesTimeSeries(psi1, u, spin_weight=1, multiplication_truncator=max)
        ψ1_1 = 2 * σ_0 * ψ3_0 + ð(ψ2_0)
        ψ1_2 = σ_0 * ψ3_1 + σ_1 * ψ3_0 + ð(ψ2_1) / 2
        ψ1_3 = (2 * σ_1 * ψ3_1 + 2 * σ_2 * ψ3_0 + ð(ψ2_2)) / 3
        ψ1_4 = (2 * σ_2 * ψ3_1 + ð(ψ2_3)) / 4
        abd.psi1 = u * (u * (u * (u * ψ1_4 + ψ1_3) + ψ1_2) + ψ1_1) + ψ1_0
        # ψ₀ = ∫ (ðψ₁ + 3σψ₂) du
        ψ0_0 = ModesTimeSeries(psi0, u, spin_weight=2, multiplication_truncator=max)
        ψ0_1 = 3 * σ_0 * ψ2_0 + ð(ψ1_0)
        ψ0_2 = (3 * σ_0 * ψ2_1 + 3 * σ_1 * ψ2_0 + ð(ψ1_1)) / 2
        ψ0_3 = σ_0 * ψ2_2 + σ_1 * ψ2_1 + σ_2 * ψ2_0 + ð(ψ1_2) / 3
        ψ0_4 = (3 * σ_0 * ψ2_3 + 3 * σ_1 * ψ2_2 + 3 * σ_2 * ψ2_1 + ð(ψ1_3)) / 4
        ψ0_5 = (3 * σ_1 * ψ2_3 + 3 * σ_2 * ψ2_2 + ð(ψ1_4)) / 5
        ψ0_6 = σ_2 * ψ2_3 / 2
        abd.psi0 = u * (u * (u * (u * (u * (u * ψ0_6 + ψ0_5) + ψ0_4) + ψ0_3) + ψ0_2) + ψ0_1) + ψ0_0
    elif np.ndim(sigma0) == 2:
        # Assume this gives complete data, as a function of time and angle.
        # If this is true, ignore sigmadot0 and sigmaddot0.
        def adjust_psi2_imaginary_part(psi2, abd):
            # Adjust the initial value of psi2 to satisfy the mass-aspect condition
            sigma_initial = abd.sigma[..., 0, :]
            sigma_bar_dot_initial = abd.sigma.bar.dot[..., 0, :]
            return (
                ModesTimeSeries(psi2, abd.time, spin_weight=0).real
                - 1j * (sigma_initial.bar.eth_GHP.eth_GHP + sigma_initial * sigma_bar_dot_initial).imag
            )

        abd.sigma = sigma0
        # ψ₄ = -∂²σ̄/∂u²
        abd.psi4 = -abd.sigma.ddot.bar
        # ψ₃ = -ð ∂σ̄/∂u
        abd.psi3 = -abd.sigma.dot.bar.eth_GHP
        # ψ₂ = ∫ (ðψ₃ + σψ₄) du
        abd.psi2 = (abd.psi3.eth_GHP + abd.sigma * abd.psi4).int + adjust_psi2_imaginary_part(psi2, abd)
        # ψ₁ = ∫ ðψ₂ + σψ₃ du
        abd.psi1 = (abd.psi2.eth_GHP + 2 * abd.sigma * abd.psi3).int + psi1
        # ψ₀ = ∫ ðψ₁ + σψ₂ dt
        abd.psi0 = (abd.psi1.eth_GHP + 3 * abd.sigma * abd.psi2).int + psi0
    else:
        raise ValueError(f"Input `sigma0` must have 1 or 2 dimensions; it has {np.ndim(sigma0)}")

    return abd
