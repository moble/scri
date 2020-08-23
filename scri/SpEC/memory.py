import numpy as np
import spherical_functions as sf
from spherical_functions import LM_index as lm
from ..modes_time_series import ModesTimeSeries


def create_ModesTimeSeries_from_WaveformModes(h_wm):
    h_mts = ModesTimeSeries(
        sf.SWSH_modes.Modes(
            h_wm.data,
            spin_weight=h_wm.spin_weight,
            ell_min=h_wm.ell_min,
            ell_max=h_wm.ell_max,
            multiplication_truncator=max,
        ),
        time=h_wm.t,
    )
    return h_mts


def D_Op(h_mts, UseLaplacian=False):
    """ Differential operator as defined in
    equation () of arXiv: (update when paper is finished)
    """

    h = h_mts.copy()
    s = h_mts.view(np.ndarray)

    def D_Op_value(ell, UseLaplacian=False):
        if ell < 2:
            return 0
        else:
            if not UseLaplacian:
                return 1.0 / (0.125 * (-ell * (ell + 1)) * ((-ell * (ell + 1)) + 2.0))
            else:
                return 1.0 / (0.125 * (-ell * (ell + 1)) ** 2.0 * ((-ell * (ell + 1)) + 2.0))

    for ell in range(h.ell_min, h.ell_max + 1):
        h[..., h.index(ell, -ell) : h.index(ell, ell) + 1] = (
            D_Op_value(ell, UseLaplacian) * s[..., h.index(ell, -ell) : h.index(ell, ell) + 1]
        )
    return h


def mem_mass_aspect_contribution(h, Psi2, news=None):
    """Calculate the Bondi mass aspect contribution to the 
    electric part of the strain according to 
    equation () of arXiv: (update when paper is finished)
    
    Parameters
    ----------
    h : WaveformModes
        WaveformModes object corresponding to the strain
    Psi2 : WaveformModes
        WaveformModes object corresponding to Psi2
    news : WaveformModes [default is h.data_dot]
        WaveformModes object corresponding to the news;
        if the default is None, then compute this as the
        time-derivative of the strain

    Returns
    -------
    Bondi_M : WaveformModes
        Bondi mass aspect contribution to the strain

    """

    if news == None:
        news = h.copy()
        news.data = h.data_dot
    h_mts = create_ModesTimeSeries_from_WaveformModes(h)
    Psi2_mts = create_ModesTimeSeries_from_WaveformModes(Psi2)
    news_mts = create_ModesTimeSeries_from_WaveformModes(news)

    Bondi_M_mts = h_mts.copy()
    Bondi_M_mts = (
        0.5 * D_Op(-(Psi2_mts + 0.25 * 1.0j * h_mts.eth.eth.imag + 0.25 * (news_mts * h_mts.bar))).ethbar.ethbar
    )
    # "* -1.0j" since we want the imaginary part, not the imaginary component

    Bondi_M = h.copy()
    Bondi_M.data = np.array(Bondi_M_mts[:, lm(2, -2, Bondi_M_mts.ell_min) :])

    return Bondi_M


def mem_energy_flux_contribution(h, news=None, start_time=None):
    """Calculate the energy flux contribution to the 
    electric part of the strain according to 
    equation () of arXiv: (update when paper is finished)

    Parameters
    ----------
    h : WaveformModes
        WaveformModes object corresponding to the strain
    news : WaveformModes [default is h.data_dot]
        WaveformModes object corresponding to the news;
        if the default is None, then compute this as the
        time-derivative of the strain
    start_time : double [default is h.t[0]]
        double corresponding to the time at which the
        energy flux integral should begin


    Returns
    -------
    E_flux : WaveformModes
        Energy flux contribution to the strain

    """

    if news == None:
        news = h.copy()
        news.data = h.data_dot
    news_mts = create_ModesTimeSeries_from_WaveformModes(news)

    E_flux_mts = news_mts.copy()
    E_flux_mts = 0.25 * (news_mts * news_mts.bar).int
    E_flux_mts = 0.5 * D_Op(E_flux_mts).ethbar.ethbar

    if not start_time == None:
        start_time_idx = np.argmin(abs(h.t - start_time))
        E_flux_mts = E_flux_mts - E_flux_mts[start_time_idx, :]

    E_flux = h.copy()
    E_flux.data = np.array(E_flux_mts[:, lm(2, -2, E_flux_mts.ell_min) :])

    return E_flux


def mem_angular_momentum_aspect_contribution(h, Psi1, news=None):
    """Calculate the angular momentum aspect contribution to the 
    magnetic part of the strain according to
    equation () of arXiv: (update when paper is finished)

    Parameters
    ----------
    h : WaveformModes
        WaveformModes object corresponding to the strain
    Psi1 : WaveformModes
        WaveformModes object corresponding to Psi1
    news : WaveformModes [default is h.data_dot]
        WaveformModes object corresponding to the news;
        if the default is None, then compute this as the
        time-derivative of the strain

    Returns
    -------
    Bondi_Ndot : WaveformModes
        Bondi angular momentum aspect contribution to the strain

    """

    if news == None:
        news = h.copy()
        news.data = h.data_dot
    h_mts = create_ModesTimeSeries_from_WaveformModes(h)
    Psi1_mts = create_ModesTimeSeries_from_WaveformModes(Psi1)
    news_mts = create_ModesTimeSeries_from_WaveformModes(news)

    Bondi_Ndot_mts = h_mts.copy()
    Bondi_Ndot_mts = (
        0.5
        * 1.0j
        * D_Op((2.0 * Psi1_mts - 0.25 * (h_mts.bar * h_mts.eth)).ethbar.dot.imag, UseLaplacian=True).ethbar.ethbar
    )
    # "* -1.0j" since we want the imaginary part, not the imaginary component

    Bondi_Ndot = h.copy()
    Bondi_Ndot.data = np.array(Bondi_Ndot_mts[:, lm(2, -2, Bondi_Ndot_mts.ell_min) :])

    return Bondi_Ndot


def mem_angular_momentum_flux_contribution(h, news=None):
    """Calculate the angular momentum flux contribution to the 
    magnetic part of the strain according to 
    equation () of arXiv: (update when paper is finished)

    Parameters
    ----------
    h : WaveformModes
        WaveformModes object corresponding to the strain
    news : WaveformModes [default is h.data_dot]
        WaveformModes object corresponding to the news;
        if the default is None, then compute this as the
        time-derivative of the strain

    Returns
    -------
    Jdot_flux : WaveformModes
        Angular momentum flux contribution to the strain

    """

    if news == None:
        news = h.copy()
        news.data = h.data_dot
    h_mts = create_ModesTimeSeries_from_WaveformModes(h)
    news_mts = create_ModesTimeSeries_from_WaveformModes(news)

    Jdot_flux_mts = h_mts.copy()
    Jdot_flux_mts = (
        0.5
        * 1.0j
        * D_Op(
            0.125
            * (
                3.0 * h_mts * news_mts.bar.ethbar
                - 3.0 * news_mts * h_mts.bar.ethbar
                + news_mts.bar * h_mts.ethbar
                - h_mts.bar * news_mts.ethbar
            ).eth.imag,
            UseLaplacian=True,
        ).ethbar.ethbar
    )
    # "* -1.0j" since we want the imaginary part, not the imaginary component

    Jdot_flux = h.copy()
    Jdot_flux.data = np.array(Jdot_flux_mts[:, lm(2, -2, Jdot_flux_mts.ell_min) :])

    return Jdot_flux


def add_memory(h, start_time=None):
    """Calculate the strain with the electric component of
    the null memory added on, i.e., the contribution from the energy flux

    Parameters
    ----------
    h : WaveformModes
        WaveformModes object corresponding to the strain
    start_time : double [default is h.t[0]]
        double corresponding to the time at which the
        energy flux integral should begin

    Returns
    -------
    h_with_mem : WaveformModes
        WaveformMOdes object corresponding to the strain with electric memory

    """

    h_with_mem = h.copy()
    h_with_mem.data = h.data + mem_energy_flux_contribution(h, start_time=start_time).data

    return h_with_mem


def BMS_strain(h, Psi2, Psi1, news=None, start_time=None, match_time=None):
    """Calculate what the strain should be according to the BMS balance laws, i.e.,
    equations () of arXiv: (update when paper is finished)

    Parameters
    ----------
    h : WaveformModes
        WaveformModes object corresponding to the strain
    Psi2 : WaveformModes
        WaveformModes object corresponding to Psi2
    Psi1 : WaveformModes
        WaveformModes object corresponding to Psi1
    news : WaveformModes [default is h.data_dot]
        WaveformModes object corresponding to the news;
        if the default is None, then compute this as the
        time-derivative of the strain
    start_time : double [default is h.t[0]]
        double corresponding to the time at which the
        energy flux integral should begin
    match_time : double [default is None]
        double corresponding to the time at which the
        strain and BMS strain should match

    Returns
    -------
    (h_BMS, Constraint, M, E, Ndot, Jdot) : ntuple of WaveformModes
        ntuple of WaveformModes objects corresponding to the strain from the BMS balance laws,
        the BMS constraint, i.e., equation () of arXiv: (update when paper is finished),
        and the contributions from the Bondi mass aspect, the energy flux,
        the Bondi angular momentum aspect, and the angular momentum flux

    """

    h_BMS = h.copy()
    M = mem_mass_aspect_contribution(h, Psi2, news=news)
    E = mem_energy_flux_contribution(h, news=news, start_time=start_time)
    Ndot = mem_angular_momentum_aspect_contribution(h, Psi1, news=news)
    Jdot = mem_angular_momentum_flux_contribution(h, news=news)
    h_BMS.data = M.data + E.data + Ndot.data + Jdot.data

    if not match_time == None:
        match_time_idx = np.argmin(abs(h.t - match_time))
        h_BMS.data = h_BMS.data + (h.data[match_time_idx, lm(2, -2, h.ell_min) :] - h_BMS.data[match_time_idx, :])

    Constraint = h_BMS.copy()
    Constraint.data = h.data[:, lm(2, -2, h.ell_min) :] - h_BMS.data

    return (h_BMS, Constraint, M, E, Ndot, Jdot)
