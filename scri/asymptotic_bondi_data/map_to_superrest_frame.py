import copy
import warnings
import numpy as np

import sxs
import scri
import spinsfast
import spherical_functions as sf
from spherical_functions import LM_index
from scri.asymptotic_bondi_data.bms_charges import charge_vector_from_aspect

import quaternion
from quaternion.calculus import derivative

from scipy.integrate import simpson
from scipy.interpolate import CubicSpline


def MT_to_WM(h_mts, sxs_version=False, dataType=scri.h):
    """Convert a ModesTimeSeries object to a scri or a sxs WaveformModes object.

    Parameters
    ----------
    h_mts: ModesTimesSeries
        ModesTimeSeries object to be converted to WaveformModes object.
    sxs_version: bool, default is False
        If True (False), then return the sxs (scri) WaveformModes object. Default is False.
    dataType: int, default is 7 (scri.h)
        Data type of the WaveformModes object, e.g., scri.h or scri.hdot. Default is 7, i.e., scri.h
    """
    if not sxs_version:
        h = scri.WaveformModes(
            t=h_mts.t,
            data=np.array(h_mts)[:, sf.LM_index(abs(h_mts.s), -abs(h_mts.s), 0) :],
            ell_min=abs(h_mts.s),
            ell_max=h_mts.ell_max,
            frameType=scri.Inertial,
            dataType=dataType,
        )
        h.r_is_scaled_out = True
        h.m_is_scaled_out = True
        return h
    else:
        h = sxs.WaveformModes(
            input_array=np.array(h_mts)[:, LM_index(abs(h_mts.s), -abs(h_mts.s), 0) :],
            time=h_mts.t,
            time_axis=0,
            modes_axis=1,
            ell_min=abs(h_mts.s),
            ell_max=h_mts.ell_max,
            spin_weight=h_mts.s,
        )
        return h


def WM_to_MT(h_wm):
    """Convert a WaveformModes object to a ModesTimeSeries object.

    Parameters
    ----------
    h_wm: WaveformModes
        WaveformModes object to be converted to ModesTimeSeries object.
    """
    h_mts = scri.ModesTimeSeries(
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


def 𝔇(h, ell_max):
    """Differential operator 𝔇 acting on spin-weight s=0 function.

    This is equivalent to ð^{2}\bar{ð}^{2}.

    It is explicitly defined in Eqs (16) and (17) of https://doi.org/10.1063/1.532646.

    """
    h_with_operator = h.copy()
    for ell in range(0, ell_max + 1):
        if ell < 2:
            value = 0
        else:
            value = ((ell + 2) * (ell + 1) * (ell) * (ell - 1)) / 4.0
        h_with_operator[LM_index(ell, -ell, 0) : LM_index(ell, ell, 0) + 1] *= value

    return h_with_operator


def 𝔇inverse(h, ell_max):
    """Inverse of differential operator 𝔇 acting on spin-weight s=0 function."""
    h_with_operator = h.copy()
    for ell in range(0, ell_max + 1):
        if ell < 2:
            value = 0
        else:
            value = 4.0 / ((ell + 2) * (ell + 1) * (ell) * (ell - 1))
        h_with_operator[LM_index(ell, -ell, 0) : LM_index(ell, ell, 0) + 1] *= value

    return h_with_operator


def compute_bondi_rest_mass_and_conformal_factor(PsiM, ell_max):
    """Compute the Bondi rest mass and conformal factor K.

    Compute the Bondi rest mass and the conformal factor K from
    the Moreschi supermomentum to help obtain the Moreschi supermomentum
    in a supertranslated frame. These are defined in Eqs. (14) and (15) of
    https://doi.org/10.1063/1.532646.

    Parameters
    ----------
    PsiM: ModesTimeSeries
        Moreschi supermomentum computed from AsymptoticBondiData object.
    ell_max: int
        Maximum ell to use when converting data from SWSHs to data on the spherical grid.
    """
    # This comes from Eq. (1), note that charge_vector_from aspect divides by np.sqrt(4 * np.pi)
    P = -charge_vector_from_aspect(PsiM)

    r_vector_modes_on_Grid = []
    for L, M in [(0, 0), (1, -1), (1, 0), (1, +1)]:
        r_vector_modes = np.zeros((ell_max + 1) ** 2)
        r_vector_modes[LM_index(L, M, 0)] = 1
        r_vector_modes_on_Grid.append(spinsfast.salm2map(r_vector_modes, 0, ell_max, 2 * ell_max + 1, 2 * ell_max + 1))

    # This comes from Eq. (12), note that charge_vector_from aspect divides by np.sqrt(4 * np.pi)
    r_vector = 4 * np.pi * charge_vector_from_aspect(np.array(r_vector_modes_on_Grid).transpose()).transpose()

    # These come from Eqs. (14) and (15)
    if len(PsiM.shape) > 1:
        M_Grid = np.sqrt(P[:, 0] ** 2 - (P[:, 1] ** 2 + P[:, 2] ** 2 + P[:, 3] ** 2))

        K_Grid = M_Grid[:, None, None] / (
            np.tensordot(P[:, 0], r_vector[0], axes=0)
            - (
                np.tensordot(P[:, 1], r_vector[1], axes=0)
                + np.tensordot(P[:, 2], r_vector[2], axes=0)
                + np.tensordot(P[:, 3], r_vector[3], axes=0)
            )
        )
    else:
        M_Grid = np.sqrt(P[0] ** 2 - (P[1] ** 2 + P[2] ** 2 + P[3] ** 2))

        K_Grid = M_Grid / (P[0] * r_vector[0] - (P[1] * r_vector[1] + P[2] * r_vector[2] + P[3] * r_vector[3]))

    return M_Grid, K_Grid


def compute_Moreschi_supermomentum(abd, alpha, ell_max):
    """Compute the Moreschi supermomentum in a supertranslated frame.

    The transformation of the Moreschi supermomentum can be found in
    Eq (9) of https://doi.org/10.1063/1.532646.

    Parameters
    ----------
    PsiM: ModesTimeSeries
        Moreschi supermomentum computed from AsymptoticBondiData object.
    alpha: ndarray, complex, shape (Ntheta, Nphi)
        Supertranslation to apply to Moreschi supermomentum.
        This should be stored as an array of spherical harmonic modes.
    ell_max: int
        Maximum ell to use when converting data from SWSHs to data on the spherical grid.
    """
    PsiM = abd.supermomentum("Moreschi")
    M_Grid, K_Grid = compute_bondi_rest_mass_and_conformal_factor(np.array(PsiM), ell_max)

    PsiM_Grid = PsiM.grid().real
    PsiM_Grid_interp = sf.SWSH_grids.Grid(np.zeros((2 * ell_max + 1, 2 * ell_max + 1), dtype=float), spin_weight=0)
    K_Grid_interp = sf.SWSH_grids.Grid(np.zeros((2 * ell_max + 1, 2 * ell_max + 1), dtype=float), spin_weight=0)

    # interpolate these functions onto the new retarded time
    for i in range(2 * ell_max + 1):
        for j in range(2 * ell_max + 1):
            alpha_i_j = alpha[i, j]
            PsiM_Grid_interp[i, j] = CubicSpline(PsiM.t, PsiM_Grid[:, i, j])(alpha_i_j)

            K_Grid_interp[i, j] = CubicSpline(PsiM.t, K_Grid[:, i, j])(alpha_i_j)

    # compute the supertranslation term
    alpha_modes = spinsfast.map2salm(alpha.view(np.ndarray), 0, ell_max)
    D_alpha_modes = 𝔇(alpha_modes, ell_max)
    D_alpha_modes_Grid = sf.SWSH_grids.Grid(
        spinsfast.salm2map(D_alpha_modes.view(np.ndarray), 0, ell_max, 2 * ell_max + 1, 2 * ell_max + 1), spin_weight=0
    )

    # transform the Moreschi supermomentum
    PsiM_Grid_interp = (PsiM_Grid_interp - D_alpha_modes_Grid) / K_Grid_interp**3

    PsiM_interp = spinsfast.map2salm(PsiM_Grid_interp.view(np.ndarray), 0, ell_max)

    return PsiM_interp


def compute_alpha_perturbation(PsiM, M_Grid, K_Grid, ell_max):
    """From the Moreschi supermomentum transformation law,
    compute the supertranslation that maps to the superrest frame.

    This equation can be found in Eq (10) of
    https://doi.org/10.1063/1.532646.

    Parameters
    ----------
    PsiM: ModesTimeSeries
        Moreschi supermomentum computed from AsymptoticBondiData object.
    M_Grid: ndarray, complex, shape (..., Ntheta, Nphi)
        Bondi mass on the spherical grid.
    K_Grid: ndarray, complex, shape (..., Ntheta, Nphi)
        Conformal factor on the spherical grid.
    ell_max: int
        Maximum ell to use when converting data from SWSHs to data on the spherical grid.
    """
    PsiM_Grid = spinsfast.salm2map(PsiM.view(np.ndarray), 0, ell_max, 2 * ell_max + 1, 2 * ell_max + 1)
    PsiM_plus_M_K3_Grid = PsiM_Grid + M_Grid * K_Grid**3
    PsiM_plus_M_K3 = spinsfast.map2salm(PsiM_plus_M_K3_Grid.view(np.ndarray), 0, ell_max)

    alpha = 𝔇inverse(PsiM_plus_M_K3, ell_max)

    return spinsfast.salm2map(alpha, 0, ell_max, 2 * ell_max + 1, 2 * ell_max + 1).real


def supertranslation_to_map_to_superrest_frame(
    abd, target_PsiM=None, N_itr_max=10, rel_err_tol=1e-12, ell_max=12, print_conv=False
):
    """Determine the supertranslation needed to map an abd object to the superrest frame.

    This is found through an iterative solve; e.g., compute the supertranslation needed to minimize
    the Moreschi supermomentum according to Eq (10) of https://doi.org/10.1063/1.532646,
    transform the Moreschi supermomentum, and repeat until the supertranslation converges.

    Parameters
    ----------
    abd: AsymptoticBondiData
        AsymptoticBondiData object from which the Moreschi supermomentum will be computed.
    target_PsiM: ModesTimeSeries, defaults to None
        Target Moreschi supermomentum, e.g., the PN supermomentum which we may map the NR supermomentum to.
        Default is None, which equates to the target supermomentum being zero.
    N_itr_max: int, defaults to 10
        Maximum numebr of iterations to perform. Default is 10.
    rel_err_tol: float, defaults to 1e-12
        Minimum relativie error tolerance between transformation iterations. Default is 1e-12.
    ell_max: int, defaults to 12
        Maximum ell to use when converting data from SWSHs to data on the spherical grid. Default is 12.
    print_conv: bool, defaults to False
        Whether or not to print the termination criterion. Default is False.
    """
    alpha_Grid = np.zeros((2 * ell_max + 1, 2 * ell_max + 1), dtype=float)

    itr = 0
    rel_err = np.inf
    rel_errs = []
    while itr < N_itr_max and not rel_err < rel_err_tol:
        prev_alpha_Grid = sf.SWSH_grids.Grid(alpha_Grid.copy(), spin_weight=0)

        PsiM = compute_Moreschi_supermomentum(abd, alpha_Grid, ell_max)

        M_Grid, K_Grid = compute_bondi_rest_mass_and_conformal_factor(np.array(PsiM), ell_max)

        if target_PsiM != None:
            target_PsiM_Grid = WM_to_MT(target_PsiM).grid().real
            target_PsiM_Grid_interp = sf.SWSH_grids.Grid(
                np.zeros((2 * target_PsiM.ell_max + 1, 2 * target_PsiM.ell_max + 1), dtype=float), spin_weight=0
            )
            for i in range(2 * target_PsiM.ell_max + 1):
                for j in range(2 * target_PsiM.ell_max + 1):
                    alpha_i_j = prev_alpha_Grid[i, j]
                    target_PsiM_Grid_interp[i, j] = CubicSpline(target_PsiM.t, target_PsiM_Grid[:, i, j])(alpha_i_j)
            M_Grid = -target_PsiM_Grid_interp.view(np.ndarray)

        alpha_Grid += compute_alpha_perturbation(PsiM, M_Grid, K_Grid, ell_max)

        rel_err = (
            spinsfast.map2salm(
                (
                    abs(sf.SWSH_grids.Grid(alpha_Grid.copy(), spin_weight=0) - prev_alpha_Grid)
                    / abs(sf.SWSH_grids.Grid(alpha_Grid.copy(), spin_weight=0))
                ).view(np.ndarray),
                0,
                ell_max,
            )[LM_index(0, 0, 0)]
            / np.sqrt(4 * np.pi)
        ).real
        rel_errs.append(rel_err)

        itr += 1

    if print_conv:
        if not itr < N_itr_max:
            print(
                f"supertranslation: maximum number of iterations reached; the min error was {min(np.array(rel_errs).flatten())}."
            )
        else:
            print(f"supertranslation: tolerance achieved in {itr} iterations!")

    supertranslation = spinsfast.map2salm(alpha_Grid.view(np.ndarray), 0, ell_max)
    supertranslation[0:4] = 0

    return supertranslation, rel_errs


def transformation_from_CoM_charge(G, t):
    """Obtain the space translation and boost velocity from the center-of-mass charge.

    This is defined in Eq (18) of https://journals.aps.org/prd/abstract/10.1103/PhysRevD.104.024051.

    Parameters
    ----------
    G: ndarray, real, shape (..., 3)
        Center-of-mass charge.
    t: ndarray, real
        Time array corresponding to the size of the center-of-mass charge.
    """
    polynomial_fit = np.polyfit(t, G, deg=1)

    CoM_transformation = {
        "space_translation": polynomial_fit[1],
        "boost_velocity": polynomial_fit[0]
    }

    return CoM_transformation


def com_transformation_to_map_to_superrest_frame(
    abd, N_itr_max=10, rel_err_tol=1e-12, ell_max=None, space_translation=True, boost_velocity=True, print_conv=False
):
    """Determine the space translation and boost needed to map an abd object to the superrest frame.

    These are found through an iterative solve; e.g., compute the transformations needed to minimize
    the center-of-mass charge, transform the abd object, and repeat until the transformations converge.

    This function can also be used to find just the space translation or the boost velocity, rather than both.

    Parameters
    ----------
    abd: AsymptoticBondiData
        AsymptoticBondiData object from which the Moreschi supermomentum will be computed.
    N_itr_max: int, defaults to 10
        Maximum numebr of iterations to perform. Default is 10.
    rel_err_tol: float, defaults to 1e-12
        Minimum relativie error tolerance between transformation iterations. Default is 1e-12.
    ell_max: int, defaults to 12
        Maximum ell to use when converting data from SWSHs to data on the spherical grid. Default is 12.
    space_translation: bool, defaults to True
        Whether or not to return the space translation.
    boost_velocity: bool, defaults to True
        Whether or not to return the boost velocity.
    print_conv: bool, defaults to False
        Whether or not to print the termination criterion. Default is False.
    """
    if not space_translation and not boost_velocity:
        raise ValueError("space_translation and boost_velocity cannot both be False.")

    CoM_transformation = {
        "space_translation": np.zeros(3),
        "boost_velocity": np.zeros(3)
    }

    itr = 0
    rel_err = np.array(2 * [np.inf])
    rel_errs = []
    while itr < N_itr_max and not (rel_err < rel_err_tol).sum() == int(space_translation) + int(boost_velocity):
        prev_CoM_transformation = copy.deepcopy(CoM_transformation)

        if itr == 0:
            abd_prime = abd.copy()
        else:
            abd_prime = abd.transform(
                space_translation=CoM_transformation["space_translation"],
                boost_velocity=CoM_transformation["boost_velocity"],
            )

        G_prime = abd_prime.bondi_CoM_charge() / abd_prime.bondi_four_momentum()[:, 0, None]

        new_CoM_transformation = transformation_from_CoM_charge(G_prime, abd_prime.t)
        for transformation in CoM_transformation:
            if not space_translation:
                if transformation == "space_translation":
                    continue
            if not boost_velocity:
                if transformation == "boost_velocity":
                    continue
            CoM_transformation[transformation] += new_CoM_transformation[transformation]

        for i, transformation in enumerate(CoM_transformation):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
                err = np.sqrt(
                    abs(
                        np.dot(
                            CoM_transformation[transformation] - prev_CoM_transformation[transformation],
                            CoM_transformation[transformation] - prev_CoM_transformation[transformation],
                        )
                        / np.dot(CoM_transformation[transformation], CoM_transformation[transformation])
                    )
                )

                if err == np.inf:
                    err = 1
                rel_err[i] = err

        rel_errs.append(copy.deepcopy(rel_err))

        itr += 1

    transformation_name = "CoM"
    if not space_translation:
        transformation_name = "boost"
        transformation = CoM_transformation["boost_velocity"]
        rel_errs = np.array(rel_errs)[:, 1:]
    elif not boost_velocity:
        transformation_name = "space_translation"
        transformation = CoM_transformation["space_translation"]
        rel_errs = np.array(rel_errs)[:, :1]
    else:
        transformation = CoM_transformation
        rel_errs = np.array(rel_errs)

    if print_conv:
        if not itr < N_itr_max:
            print(
                f"{transformation_name}: maximum number of iterations reached; the min error was {min(rel_errs.flatten())}."
            )
        else:
            print(f"{transformation_name}: tolerance achieved in {itr} iterations!")

    return transformation, rel_errs


def rotation_from_spin_charge(chi, t):
    """Obtain the rotation from the remnant BH's spin vector.

    This finds the rotation that aligns the z-component of the spin vector with the z-axis.

    Parameters
    ----------
    chi: ndarray, real, shape (..., 3)
        Remnant BH's spin vector.
    t: ndarray, real
        Time array corresponding to the size of the spin vector.
    """
    chi_f = quaternion.quaternion(*chi[np.argmin(abs(t))]).normalized()
    return (1 - chi_f * quaternion.z).normalized()


def rotation_from_omega_vectors(omega, target_omega, t=None):
    """Obtain the rotation from the two angular velocity vectors.

    This finds the rotation that best aligns two angular velocity vectors over the time interval t.

    Parameters
    ----------
    omega: ndarray, real, shape (..., 3)
        Angular velocity vector.
    target_omega: ndarray, real, shape (..., 3)
        Target angular velocity vector.
    t: ndarray, real
        Time array corresponding to the size of the spin vector. Default is None.
    """
    q = quaternion.optimal_alignment_in_Euclidean_metric(omega, target_omega, t=None)
    return q


def rotation_to_map_to_superrest_frame(
    abd, target_h=None, N_itr_max=10, rel_err_tol=[1e-12, 1e-5], ell_max=None, print_conv=False
):
    """Determine the rotation needed to map an abd object to the superrest frame.

    This is found through an iterative solve; e.g., compute the transformation needed to align
    the spin vector charge with the z-axis or align the angular velocity with a target angular velocity,
    transform the abd object, and repeat until the transformation converges.

    Note that the angular momentum charge is aligned with either the
    positive or negative z-axis, depending on which it is initially closest to.

    If target_h is not None, then instead find the rotation that aligns the
    angular velocity vector of the abd object to the angular velocity vector
    of the target_h input.

    Parameters
    ----------
    abd: AsymptoticBondiData
        AsymptoticBondiData object from which the Moreschi supermomentum will be computed.
    target_h: WaveformModes
        WaveformModes object from which the target angular velocity will be computed. Default is None.
    N_itr_max: int, defaults to 10
        Maximum number of iterations to perform. Default is 10.
    rel_err_tol: array_like, defaults to [1e-12, 1e-5]
        First value is minimum relative error tolerance between transformation iterations; second value is the
        minimum relative error tolerance between the NR angular velocity and the target angular velocity.
        Default is [1e-12, 1e-5].
    ell_max: int, defaults to 12
        Maximum ell to use when converting data from SWSHs to data on the spherical grid. Default is 12.
    print_conv: bool, defaults to False
        Whether or not to print the termination criterion. Default is False.
    """
    rotation = quaternion.quaternion(1, 0, 0, 0)

    # if there is no target_h, just map the spin charge to the z-axis;
    # otherwise, align the angular momentum fluxes over the full window.
    if target_h == None:
        itr = 0
        rel_err = np.inf
        rel_errs = []
        rel_errs_omega = []
        while itr < N_itr_max and not rel_err < rel_err_tol[0]:
            prev_rotation = copy.deepcopy(rotation)

            if itr == 0:
                abd_prime = abd.copy()
            else:
                abd_prime = abd.transform(frame_rotation=rotation.components)

            chi_prime = abd_prime.bondi_dimensionless_spin()

            rotation = rotation_from_spin_charge(chi_prime, abd_prime.t) * rotation

            rel_err = quaternion.rotation_intrinsic_distance(rotation, prev_rotation)

            rel_errs.append(rel_err)

            itr += 1

        if print_conv:
            if not itr < N_itr_max:
                print(
                    f"rotation: maximum number of iterations reached; the min error was {min(np.array(rel_errs).flatten())}."
                )
            else:
                print(f"rotation: tolerance achieved in {itr} iterations!")

        return rotation.components, rel_errs
    else:
        target_news = MT_to_WM(WM_to_MT(target_h).dot, dataType=scri.hdot)

        target_omega = target_news.angular_velocity()
        target_omega = target_omega / np.linalg.norm(target_omega, axis=1)[:, None]

        itr = 0
        rel_err = np.inf
        rel_err_omega = np.inf
        rel_errs = []
        rel_errs_omega = []
        while itr < N_itr_max and not (rel_err < rel_err_tol[0] or rel_err_omega < rel_err_tol[1]):
            prev_rotation = copy.deepcopy(rotation)

            if itr == 0:
                abd_prime = abd.copy()
            else:
                abd_prime = abd.transform(frame_rotation=rotation.components)
            news = MT_to_WM(2.0 * abd_prime.sigma.bar.dot, dataType=scri.hdot)

            omega = news.angular_velocity()
            omega = omega / np.linalg.norm(omega, axis=1)[:, None]

            rotation = rotation_from_omega_vectors(omega, target_omega, news.t) * rotation

            rel_err = quaternion.rotor_intrinsic_distance(rotation, prev_rotation)
            rel_err_omega = simpson(np.linalg.norm(omega - target_omega, axis=1), news.t) / simpson(
                np.linalg.norm(target_omega, axis=1), news.t
            )

            rel_errs.append([rel_err, rel_err_omega])

            itr += 1

    if print_conv:
        if not itr < N_itr_max:
            print(
                f"rotation: maximum number of iterations reached; the min error was {min(np.array(rel_errs).flatten())}."
            )
        else:
            print(f"rotation: tolerance achieved in {itr} iterations!")

    return rotation.components, rel_errs


def time_translation(abd, t_0=0):
    """Time translate an abd object.

    This is necessary because creating a copy
    of an abd object and then changing it's time variable does not change
    the time variable of the waveform variables.
    """
    abd_prime = scri.asymptotic_bondi_data.AsymptoticBondiData(abd.t, abd.ell_max)

    abd_prime.sigma = abd.sigma
    abd_prime.psi4 = abd.psi4
    abd_prime.psi3 = abd.psi3
    abd_prime.psi2 = abd.psi2
    abd_prime.psi1 = abd.psi1
    abd_prime.psi0 = abd.psi0

    abd_prime.t -= t_0

    return abd_prime


def map_to_superrest_frame(
    self,
    t_0=0,
    target_h_input=None,
    target_PsiM_input=None,
    N_itr_maxes={
        "supertranslation": 10,
        "com_transformation": 10,
        "rotation": 10,
    },
    rel_err_tols={
        "supertranslation": 1e-12,
        "com_transformation": 1e-12,
        "rotation": [1e-12, 1e-5],
    },
    ell_max=None,
    alpha_ell_max=None,
    padding_time=250,
    print_conv=False,
):
    """Transform an abd object to the superrest frame.

    This computes the transformations necessary to map an abd object to the superrest frame
    by iteratively minimizing various BMS charges at a certain time. This can be performed either
    without a target (in which case it maps to the superrest frame at the input time t_0) or
    with a target (in which case it maps to the frame of the target). For example, one could
    supply a t_0 which is toward the end of the simulation in which case this code would map to
    the superrest frame of the remnant BH. Or, one could supply a target PN strain and supermomentum and
    a t_0 during the early inspiral in which case this code would map to the PN BMS frame.

    This wholly fixes the BMS frame of the abd object up to a
    time translation and a phase rotation.

    Parameters
    ----------
    t_0 : float, optional
        When to map to the superrest frame.
        Default is 0.
    target_h_input : WaveformModes, optional
        Target strain used to constrain the rotation
        freedom via the angular momentum flux.
        Default is aligning to the z-axis.
    target_PsiM_input : WaveformModes, optional
        Target Moreschi supermomentum to map to.
        Default is 0.
    N_itr_maxes : dict, optional
        Maximum number of iterations to perform for each transformation.
        Default is
        N_itr_maxes = {
            'supertranslation':   10,
            'com_transformation': 10,
            'rotation':           10
        }.
    rel_err_tols : dict, optional
        Relative error tolerances for each transformation.
        Default is
        N_itr_maxes = {
            'supertranslation':   1e-12,
            'com_transformation': 1e-12,
            'rotation':           1e-12
        }.
    ell_max : int, optional
        Maximum ell to use for SWSH/Grid transformations.
        Default is self.ell_max.
    alpha_ell_max : int, optional
        Maximum ell of the supertranslation to use.
        Default is self.ell_max.
    padding_time : float, optional
        Amount by which to pad around t_0 to speed up computations, i.e.,
        distance from t_0 in each direction to be included in self.interpolate(...).
        This also determines the range over which certain BMS charges will be computed.
        Default is 100.

    Returns
    -------
    abd_prime : AsymptoticBondiData
        Result of self.transform(...) where the input transformations are
        the transformations found in the transformations dictionary.
    transformations : dict
        Dictionary of transformations and their relative errors whose keys are
            * 'transformations'
                * 'space_translation'
                * 'supertranslation'
                * 'frame_rotation'
                * 'boost_velocity'
            * 'rel_errs'
                (same as above)

    """
    # apply a time translation so that we're mapping
    # to the superrest frame at u = 0
    abd = time_translation(self, t_0)

    if target_h_input != None:
        target_h = target_h_input.copy()
        target_h.t -= t_0
    else:
        target_h = None

    if target_PsiM_input != None:
        target_PsiM = target_PsiM_input.copy()
        target_PsiM.t -= t_0
    else:
        target_PsiM = None

    if ell_max == None:
        ell_max = abd.ell_max

    if alpha_ell_max == None:
        alpha_ell_max = ell_max

    abd_interp = abd.interpolate(
        abd.t[np.argmin(abs(abd.t - (-padding_time))) : np.argmin(abs(abd.t - (+padding_time))) + 1]
    )

    # space_translation
    space_translation, space_rel_errs = com_transformation_to_map_to_superrest_frame(
        abd_interp,
        N_itr_max=N_itr_maxes["com_transformation"],
        rel_err_tol=rel_err_tols["com_transformation"],
        ell_max=ell_max,
        space_translation=True,
        boost_velocity=False,
        print_conv=print_conv,
    )

    # supertranslation
    abd_prime = abd_interp.transform(space_translation=space_translation)

    alpha, alpha_rel_errs = supertranslation_to_map_to_superrest_frame(
        abd_prime,
        target_PsiM,
        N_itr_max=N_itr_maxes["supertranslation"],
        rel_err_tol=rel_err_tols["supertranslation"],
        ell_max=ell_max,
        print_conv=print_conv,
    )

    alpha[0] = 0
    alpha[1:4] = sf.vector_as_ell_1_modes(space_translation)

    # rotation
    abd_prime = abd_interp.transform(supertranslation=alpha[: LM_index(alpha_ell_max, alpha_ell_max, 0) + 1])

    # do interpolation here because .transform() changes the size of the time array
    target_h_interp = None
    if target_h != None:
        target_h_interp = target_h.interpolate(abd_prime.t)
    rotation, rot_rel_errs = rotation_to_map_to_superrest_frame(
        abd_prime,
        target_h_interp,
        N_itr_max=N_itr_maxes["rotation"],
        rel_err_tol=rel_err_tols["rotation"],
        ell_max=ell_max,
        print_conv=print_conv,
    )

    # com_transformation
    abd_prime = abd_interp.transform(
        supertranslation=alpha[: LM_index(alpha_ell_max, alpha_ell_max, 0) + 1], frame_rotation=rotation
    )

    CoM_transformation, CoM_rel_errs = com_transformation_to_map_to_superrest_frame(
        abd_prime,
        N_itr_max=N_itr_maxes["com_transformation"],
        rel_err_tol=rel_err_tols["com_transformation"],
        ell_max=ell_max,
        space_translation=True,
        boost_velocity=True,
        print_conv=print_conv,
    )

    # transform abd
    abd_prime = abd.transform(
        supertranslation=alpha[: LM_index(alpha_ell_max, alpha_ell_max, 0) + 1], frame_rotation=rotation
    )
    abd_prime = abd_prime.transform(
        space_translation=CoM_transformation["space_translation"], boost_velocity=CoM_transformation["boost_velocity"]
    )

    # undo the initial time translation
    abd_prime = time_translation(abd_prime, -t_0)

    alpha[0:4] = 0

    transformations = {
        "transformations": {
            "space_translation": space_translation,
            "supertranslation": alpha,
            "frame_rotation": rotation,
            "CoM_transformation": CoM_transformation,
        },
        "rel_errs": {
            "space_translation": space_rel_errs,
            "supertranslation": alpha_rel_errs,
            "frame_rotation": rot_rel_errs,
            "CoM_transformation": CoM_rel_errs,
        },
    }

    return abd_prime, transformations
