def conformal_factor(v, ell_max, exponent=1, **kwargs):
    """Compute the conformal factor for velocity v as Modes object
    
    The conformal factor (traditionally denoted `k`) is given by

        k = 1 / (gamma * (1 - v . r))

    where gamma=1/sqrt(1 - v.v) is the usual Lorentz factor, and r is the direction to the observation point.
    
    Parameters
    ==========
    v: array_like
        One-dimensional array of 3 floats describing the velocity
    ell_max: int
        Largest ell value in the output.  For exponent<1, there is no information beyond ell=-exponent.
    exponent: int [defaults to 1]
        Return Modes object for k**exponent.

    """
    import spinsfast
    v = np.array(v, dtype=float)
    if np.ndim(v) != 1 or v.size != 3:
        raise ValueError(f"Input `v` must be one-dimensional array of size 3; it has shape {v.shape}")
    def v_dot_r(theta, phi):
        return v[0]*math.sin(theta)*math.cos(phi) + v[1]*math.sin(theta)*math.sin(phi) + v[2]*math.cos(theta)
    gamma = 1/math.sqrt(1 - v[0]**2 - v[1]**2 - v[2]**2)
    n_theta = 2 * ell_max + 1
    n_phi = n_theta
    theta_phi = sf.theta_phi(n_theta, n_phi)
    k_grid = [
        [
            (gamma * (1 - v_dot_r(theta, phi)))**(-exponent)
            for theta, phi in tp
        ]
        for tp in theta_phi
    ]
    k_modes = spinsfast.map2salm(k_grid, 0, ell_max)
    return sf.Modes(k_modes, spin_weight=0, ell_max=ell_max, **kwargs)


def transform(self, boost_velocity, supertranslation):
    # ðk = ð((1/k)**-1) = -(1/k)**-2 * ð(1/k) = - k**2 * ð(1/k)
    # u' = k(u-α)
    # ðu' = ðk(u-α) - k ðα
    # ðu'/k = ðk * (u - α) / k - ðα
    #       = - ð(1/k) * k * (u - α) - ðα
    #
    # Note that 1/k has power up to ell=1, so accuracy to ell_max can be maintained by keeping
    # ell_max+1 modes in ðu', and then truncating the product of ðu' and 1/k at ell_max.
    ð = lambda m: m.eth
    k = conformal_factor(boost_velocity, self.ell_max, multiplication_truncator=lambda tup: self.ell_max+1)
    one_over_k = conformal_factor(boost_velocity, 1, -1, multiplication_truncator=lambda tup: self.ell_max)
    u = sf.Modes(sf.constant_as_ell_0_mode(self.u), spin_weight=0, ell_max=0, multiplication_truncator=max)
    α = sf.Modes(supertranslation, spin_weight=0, multiplication_truncator=max)
    one_over_k_cubed = conformal_factor(boost_velocity, 3, -3, multiplication_truncator=max)
    uprime = k*(u-α)
    ðuprime = ð(uprime)
    ðuprime._metadata['multiplication_truncator'] = lambda tup: self.ell_max
    ðuprime_over_k = ðuprime * one_over_k

    # We'll also need this below, but there's no sense keeping extra ell values
    uprime = uprime.truncate_ell(self.ell_max)

    # To avoid aliasing in the nested products below, we now need to keep 2*ell_max with each product
    ðuprime_over_k._metadata['multiplication_truncator'] = lambda tup: 2 * self.ell_max

    # The correct phase is actually produced by evaluating the functions as
    # SWSHs on the quaternions.  Let's keep it around anyway, just for form's
    # sake, so that these equations can look like they do in the paper.
    exp_i_λ = 1

    # Since these multiplications are so slow, we really need to squeeze every
    # bit of performance out.  The usual way to evaluate a polynomial is by
    # Horner's method, so I'll try that for now.  But it may make sense to
    # split these up more, to use in-place multiplications and additions.
    psi0prime = exp_i_λ**2 * one_over_k_cubed * (
        self.psi0
        + ðuprime_over_k * (
            -4*self.psi1
            + ðuprime_over_k * (
                6*self.psi2
                + ðuprime_over_k * (
                    -4*self.psi3
                    + ðuprime_over_k * (
                        self.psi4
                    )
                )
            )
        )
    ).truncate_ell(self.ell_max)

    psi1prime = exp_i_λ * one_over_k_cubed * (
        self.psi1
        + ðuprime_over_k * (
            -3*self.psi2
            + ðuprime_over_k * (
                3*self.psi3
                + ðuprime_over_k * (
                    self.psi4
                )
            )
        )
    ).truncate_ell(self.ell_max)

    psi2prime = one_over_k_cubed * (
        self.psi2
        + ðuprime_over_k * (
            -2*self.psi3
            + ðuprime_over_k * (
                self.psi4
            )
        )
    ).truncate_ell(self.ell_max)

    psi3prime = exp_i_λ**-1 * one_over_k_cubed * (
        self.psi3
        - ðuprime_over_k * (
            self.psi4
        )
    ).truncate_ell(self.ell_max)

    psi4prime = (exp_i_λ**-2 * one_over_k_cubed * self.psi4).truncate_ell(self.ell_max)

    sigmaprime = (exp_i_λ**2 * one_over_k * (sigma - ð(ð(α)))).truncate_ell(self.ell_max)
    
    raise NotImplementedError(f"Evaluate on the new grid, interpolate to the new times, and transform back to SWSHs")

