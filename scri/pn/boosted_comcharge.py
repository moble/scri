import numpy as np


def PN_charges(m, ν):
    """
    Determine a tuple of callables for a specific input of mass (m)
    and symmetric mass ratio (ν).

    Parameters
    ----------
    m: float, real
        Total mass of the binary.
    ν: float, real
        Symmetric mass ratio of the binary.

    Returns
    -------
    energy_pn : callable
        Returns PN approximation for energy up to 2PN order
    angular_momentum_pn : callable
        Returns PN approximation for angular momentum up to 3PN order
    G_mag_pn : callable
        Returns PN approximation for the magnitude of CoM charge up to
        leading PN order
    orbital_phase : callable
        Returns the orbital phase from the (2,1) mode of the strain.
        See notes for convention used.

    All returned callables accept an `ABD` object as an input
    parameter and return their respective quantities evaluated at the
    same time steps as the `ABD` object.  The PN expressions for
    energy and angular momentum are taken from Eqs. (337) and (338) of
    Blanchet's Living Review (2014) <https://arxiv.org/abs/1310.1528>.
    The PN expression for CoM charge is derived in Khairnar et al.
    <add_arXiv_link>.  The tuple of these callables can be passed as
    the `Gargsfun` keyword argument to `map_to_superrest_frame`.
    These callables are then used to determine the Gargs parameters to
    `transformation_from_CoM_charge`.  They are called as
    
        Gargs = [func(abd) for func in Gargsfun] if Gargsfun else None
        
    within the `com_transformation_to_map_to_superrest_frame`
    function.

    Notes
    ------
    Our conventions for defining the orbital phase differ from the
    standard conventions used in PN theory.  This stems from the fact
    that SpEC uses h_{ab} to define the metric perturbation while PN
    theory uses h^{ab} for the metric perturbation, which results in
    h^{PN}_{l,m} = − h^{NR}_{l,m}.  The leading order PN expression
    for the h_{2,1} mode is (Eq. 492 of
    <https://arxiv.org/abs/1310.1528>) h^{PN}_{21} = (2 G ν m x)/R *
    \sqrt(16 \pi/5) * ((1/3) 𝒾 Δ x^{1/2} + O(x^{3/2})).  Because of
    the 𝒾 factor we get the orbital phase numerically as ψ = -
    arg(-h^{NR}_{2,1}) + π/2.
    """

    def compute_x(abd):
        """Helper function to extract PN parameter x."""
        h = abd.h
        ω_mag = np.linalg.norm(h.angular_velocity(), axis=1)
        x = (m * ω_mag) ** (2.0 / 3.0)
        return x

    def energy_pn(abd):
        x = compute_x(abd)
        E = m - (m * ν * x) / 2 * (1 + (-3 / 4 - ν / 12) * x + (-27 / 8 + 19 / 8 * ν - ν**2 / 24) * x**2)
        return E

    def angular_momentum_pn(abd):
        x = compute_x(abd)
        J_mag = (m**2 * ν * x ** (-1 / 2)) * (
            1
            + (3 / 2 + ν / 6) * x
            + (-27 / 8 + 19 / 8 * ν - ν**2 / 24) * x**2
            + (135 / 16 + (-6889 / 144 + 41 / 24 * np.pi**2) * ν + 31 / 24 * ν**2 + 7 / 1296 * ν**3) * x**3
        )
        return J_mag

    def G_mag_pn(abd):
        x = compute_x(abd)
        G_mag = -(1142 / 105) * x ** (5 / 2) * m**2 * np.sqrt(1 - 4 * ν) * ν**2
        return G_mag

    def orbital_phase(abd):
        h = abd.h
        h_21 = h.data[:, h.index(2, 1)]
        ψ = (-1) * np.unwrap(np.angle(-h_21) - np.pi / 2)
        return ψ

    return energy_pn, angular_momentum_pn, G_mag_pn, orbital_phase


def analytical_CoM_func(θ, t, E, J_mag, G_mag, ψ):
    """
    Compute a model time series for boosted center-of-mass (CoM)
    charge using PN expressions of energy, angular momentum, and CoM
    charge, along with the phase computed from the h_{21} mode.

    This model timeseries is derived in Khairnar et al.
    <add_arXiv_link>.  It serves as a fitting function that can be
    passed as the `Gfun` keyword argument to the
    `map_to_superrest_frame` function.  All the arguments are computed
    over the window used for fixing the frame.

    Parameters
    ----------
    θ: ndarray, real, shape(8,)
        Parameters of the model.
        - θ[0:3] : components of the boost velocity
        - θ[3:6] : components of the spatial translation
        - θ[6:]  : two additional fit parameters referred to as the
                   nuisance parameters in Khairnar et al.
                   <add_arXiv_link>.
    t: ndarray, real
        Time array corresponding to the window over which the frame
        fixing is performed.
    E: ndarray, real
        PN approximation for the energy computed over the fitting
        window.
    J_mag: ndarray, real
        PN approximation for the magnitude of angular momentum
        computed over the fitting window.
    G_mag: ndarray, real
        PN approximation for the magnitude of the CoM charge computed
        over the fitting window.
    ψ: ndarray, real
        Unwrapped orbital phase obtained from the (2,1) mode of the
        strain over the fitting window.  See PN_charges for
        appropriate conventions.

    Returns
    -------
    G: ndarray, real, shape(..., 3)
        Model time series of the boosted center-of-mass charge.
    """
    β = θ[0:3][np.newaxis]
    X = θ[3:6][np.newaxis]
    α1, α2 = θ[6:]

    # Get the orbital direction vectors, and the angular momentum vector.
    cosψ, sinψ = np.cos(ψ), np.sin(ψ)
    nvec = np.array([cosψ, sinψ, 0 * ψ]).T
    λvec = np.array([-sinψ, cosψ, 0 * ψ]).T
    J = np.array([0 * t, 0 * t, J_mag]).T

    G = (
        α1 * (G_mag / E)[:, np.newaxis] * λvec
        + α2 * (G_mag / E)[:, np.newaxis] * nvec
        - np.cross(β, J / E[:, np.newaxis])
        - (t[:, np.newaxis] @ β)
        + X
    )

    return G
