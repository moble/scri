import numpy as np

def PN_charges(m, ν):
    """ 
    Determine a tuple of callables for passing to
    com_transformation_map_to_superrest_frame function for a specific input of
    mass (m) and symmetric mass ratio (ν).
    
    Parameters
    ----------
    m: float, real
        Total mass of the binary.
    ν: float, real
        Symmetric mass ratio of the binary.

    Returns
    -------
    (energy_pn, angular_momentum_pn, com_charge_pn, phase): tuple of callables
        energy_pn is the PN series of energy up to 2PN order.
        angular_momentum_pn is the PN series of angular momentum up to 3PN order.
        com_charge_pn is the PN series of CoM charge up to leading order.
        orbital_phase is the phase obtained from the (2,1) mode of the strain.
        
        Our conventions for defining the orbital phase differ from the standard
        conventions used in PN theory. This stems from the fact that SpEC uses
        h_{ab} to define the metric perturbation while PN theory uses h^{ab} for
        the metric perturbation, which results in h^{NR}_{lm} = − h^{PN}_{lm}.
        Also, the π/2 phase difference is due to the leading order complex phase
        of h_{21} mode from PN theory.

    All of the callables accept abd object as a parameter.
    """

    def compute_x(abd):
        """Helper function to extract PN parameter x."""
        h = abd.h
        ω_mag = np.linalg.norm(h.angular_velocity(), axis=1)
        x = (m * ω_mag) ** (2. / 3.)
        
        return x

    def energy_pn(abd):
        x = compute_x(abd)
        E = m - (m * ν * x)/2 * (1 + (-3/4 - ν/12)* x + (-27/8 + 19/8 * ν - ν**2/24) * x**2)
        return E

    def angular_momentum_pn(abd):
        x = compute_x(abd)
        J_mag = (m**2 * ν * x**(-1/2)) * (1 + (3/2 + ν/6)* x + (-27/8 + 19/8 * ν - ν**2/24) * x**2 
                 + (135/16 + (-6889/144 + 41/24 * np.pi**2)* ν + 31/24 * ν**2 + 7/1296 * ν**3) * x**3)
        return J_mag

    def G_mag_pn(abd):
        x = compute_x(abd)
        G_mag = -(1142/105) * x**(5/2) * m**2 * np.sqrt(1 - 4*ν) * ν**2
        return G_mag

    def orbital_phase(abd):
        h = abd.h
        h_21 = h.data[:, h.index(2,1)]
        ψ = (-1)* np.unwrap(np.angle(-h_21) - np.pi/2)
        return ψ

    return energy_pn, angular_momentum_pn, G_mag_pn, orbital_phase


def analytical_CoM_func(θ, t, E, J_mag, G_mag, ψ):
    """
    Computes the boosted center-of-mass-charge using PN expressions of other charges.
    
    Parameters
    ----------
    θ: list[floats]
        Parameters of the model.
        θ[0:3] are the components of boost_velocity, θ[3:6] are components of
        the spatial translation (l=1 supertranslation), and there can be any
        extra fit parameters desired (but their fitted values are ignored).
    t: ndarray, real
        Time array corresponding to the size of the center-of-mass charge.
    E: ndarray, real
        Array corresponding to the energy.
    J_mag: ndarray, real
        Array corresponding to the magnitude of angular momentum.
    G_mag: ndarray, real
        Array corresponding to the magnitude of the center-of-mass charge.
    ψ: ndarray, real
        Orbital phase obtained from the (2,1) mode of the strain.

    Returns
    -------
    G: ndarray, real, shape(..., 3)
        Model timeseries of the boosted center-of-mass charge.
    """
    β = θ[0:3][None]
    X = θ[3:6][None]
    α1, α2 = θ[6:]
    
    # Get the orbital direction vectors, and the angular momentum vector.
    cosψ, sinψ = np.cos(ψ), np.sin(ψ)
    nvec = np.stack((cosψ, sinψ, 0*ψ),axis=1)
    λvec = np.stack((-sinψ, cosψ,0*ψ),axis=1)
    J = np.array([0*t,0*t,J_mag]).T
    
    G = α1 * (G_mag/E)[:, None] * λvec + α2 * (G_mag/E)[:, None] * nvec - np.cross( β, J/E[:,None]) - t[:,None] @ β + X
    
    return G