import numpy as np

import sxs
import scri
import spinsfast
import spherical_functions as sf
from spherical_functions import LM_index

from sxs.waveforms.alignment import align2d
from scri.asymptotic_bondi_data.bms_charges import charge_vector_from_aspect

import quaternion
from quaternion.calculus import indefinite_integral as integrate

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
            data=np.array(h_mts)[:, LM_index(abs(h_mts.s), -abs(h_mts.s), 0) :],
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
        h_with_operator[..., LM_index(ell, -ell, 0) : LM_index(ell, ell, 0) + 1] *= value

    return h_with_operator


def 𝔇inverse(h, ell_max):
    """Inverse of differential operator 𝔇 acting on spin-weight s=0 function."""
    h_with_operator = h.copy()
    for ell in range(0, ell_max + 1):
        if ell < 2:
            value = 0
        else:
            value = 4.0 / ((ell + 2) * (ell + 1) * (ell) * (ell - 1))
        h_with_operator[..., LM_index(ell, -ell, 0) : LM_index(ell, ell, 0) + 1] *= value

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


def compute_Moreschi_supermomentum(PsiM, alpha, ell_max):
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
    best_alpha_Grid = np.zeros((2 * ell_max + 1, 2 * ell_max + 1), dtype=float)

    PsiM = abd.supermomentum("Moreschi")

    itr = 0
    rel_err = np.inf
    rel_errs = [np.inf]
    while itr < N_itr_max and not rel_err < rel_err_tol:
        prev_alpha_Grid = sf.SWSH_grids.Grid(alpha_Grid.copy(), spin_weight=0)

        if itr == 0:
            PsiM_interp = compute_Moreschi_supermomentum(PsiM, alpha_Grid, ell_max)

            M_Grid, K_Grid = compute_bondi_rest_mass_and_conformal_factor(np.array(PsiM_interp), ell_max)

            if target_PsiM is not None:
                target_PsiM_Grid = WM_to_MT(target_PsiM).grid().real
                target_PsiM_Grid_interp = sf.SWSH_grids.Grid(
                    np.zeros((2 * target_PsiM.ell_max + 1, 2 * target_PsiM.ell_max + 1), dtype=float), spin_weight=0
                )
                for i in range(2 * target_PsiM.ell_max + 1):
                    for j in range(2 * target_PsiM.ell_max + 1):
                        alpha_i_j = prev_alpha_Grid[i, j]
                        target_PsiM_Grid_interp[i, j] = CubicSpline(target_PsiM.t, target_PsiM_Grid[:, i, j])(alpha_i_j)
                M_Grid = -target_PsiM_Grid_interp.view(np.ndarray)

        alpha_Grid += compute_alpha_perturbation(PsiM_interp, M_Grid, K_Grid, ell_max)

        PsiM_interp = compute_Moreschi_supermomentum(PsiM, alpha_Grid, ell_max)

        M_Grid, K_Grid = compute_bondi_rest_mass_and_conformal_factor(np.array(PsiM_interp), ell_max)

        if target_PsiM is not None:
            target_PsiM_Grid = WM_to_MT(target_PsiM).grid().real
            target_PsiM_Grid_interp = sf.SWSH_grids.Grid(
                np.zeros((2 * target_PsiM.ell_max + 1, 2 * target_PsiM.ell_max + 1), dtype=float), spin_weight=0
            )
            for i in range(2 * target_PsiM.ell_max + 1):
                for j in range(2 * target_PsiM.ell_max + 1):
                    alpha_i_j = prev_alpha_Grid[i, j]
                    target_PsiM_Grid_interp[i, j] = CubicSpline(target_PsiM.t, target_PsiM_Grid[:, i, j])(alpha_i_j)
            M_Grid = -target_PsiM_Grid_interp.view(np.ndarray)

            target_PsiM_interp = spinsfast.map2salm(target_PsiM_Grid_interp.view(np.ndarray), 0, ell_max)

            rel_err = np.linalg.norm(PsiM_interp[4:] - target_PsiM_interp[4:]) / np.linalg.norm(target_PsiM_interp[4:])
        else:
            rel_err = np.linalg.norm(PsiM_interp[4:])

        if rel_err < min(rel_errs):
            best_alpha_Grid = alpha_Grid.copy()
        rel_errs.append(rel_err)

        itr += 1

    if print_conv:
        if not itr < N_itr_max:
            print(
                f"supertranslation: maximum number of iterations reached; the min error was {min(np.array(rel_errs).flatten())}."
            )
        else:
            print(f"supertranslation: tolerance achieved in {itr} iterations!")

    supertranslation = spinsfast.map2salm(best_alpha_Grid.view(np.ndarray), 0, ell_max)
    supertranslation[0:4] = 0

    return scri.bms_transformations.BMSTransformation(supertranslation=supertranslation), rel_errs


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

    CoM_transformation = scri.bms_transformations.BMSTransformation(
        supertranslation=-np.insert(sf.vector_as_ell_1_modes(polynomial_fit[1]), 0, 0),
        boost_velocity=polynomial_fit[0],
        order=["supertranslation", "boost_velocity", "frame_rotation"],
    )

    return CoM_transformation


def com_transformation_to_map_to_superrest_frame(abd, N_itr_max=10, rel_err_tol=1e-12, print_conv=False):
    """Determine the space translation and boost needed to map an abd object to the superrest frame.

    These are found through an iterative solve; e.g., compute the transformations needed to minimize
    the center-of-mass charge, transform the abd object, and repeat until the transformations converge.

    This function can also be used to find just the space translation or the boost velocity, rather than both.

    Parameters
    ----------
    abd: AsymptoticBondiData
        AsymptoticBondiData object from which the Moreschi supermomentum will be computed.
    N_itr_max: int, defaults to 10
        Maximum number of iterations to perform. Default is 10.
    rel_err_tol: float, defaults to 1e-12
        Minimum relativie error tolerance between transformation iterations. Default is 1e-12.
    print_conv: bool, defaults to False
        Whether or not to print the termination criterion. Default is False.
    """
    CoM_transformation = scri.bms_transformations.BMSTransformation()
    best_CoM_transformation = scri.bms_transformations.BMSTransformation()

    itr = 0
    rel_err = np.inf
    rel_errs = [np.inf]
    while itr < N_itr_max and not rel_err < rel_err_tol:
        if itr == 0:
            abd_prime = abd.copy()
            G_prime = abd_prime.bondi_CoM_charge() / abd_prime.bondi_four_momentum()[:, 0, None]

        new_CoM_transformation = transformation_from_CoM_charge(G_prime, abd_prime.t)
        CoM_transformation = (new_CoM_transformation * CoM_transformation).reorder(
            ["supertranslation", "frame_rotation", "boost_velocity"]
        )
        # remove proper supertranslation and frame rotation components
        CoM_transformation.supertranslation[4:] *= 0
        CoM_transformation.frame_rotation = quaternion.quaternion(1, 0, 0, 0)

        abd_prime = abd.transform(
            supertranslation=CoM_transformation.supertranslation,
            frame_rotation=CoM_transformation.frame_rotation.components,
            boost_velocity=CoM_transformation.boost_velocity,
        )

        G_prime = abd_prime.bondi_CoM_charge() / abd_prime.bondi_four_momentum()[:, 0, None]

        rel_err = integrate(np.linalg.norm(G_prime, axis=-1), abd_prime.t)[-1] / (abd_prime.t[-1] - abd_prime.t[0])
        if rel_err < min(rel_errs):
            best_CoM_transformation = CoM_transformation.copy()
        rel_errs.append(rel_err)

        itr += 1

    if print_conv:
        if not itr < N_itr_max:
            print(f"CoM: maximum number of iterations reached; the min error was {min(rel_errs)}.")
        else:
            print(f"CoM: tolerance achieved in {itr} iterations!")

    return best_CoM_transformation, rel_errs


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
    q = (1 - chi_f * quaternion.z).normalized().components
    return scri.bms_transformations.BMSTransformation(frame_rotation=q)


def rotation_from_vectors(vector, target_vector, t=None):
    """Obtain the rotation from the two vectors.

    This finds the rotation that best aligns two vectors over the time interval t.

    Parameters
    ----------
    vector: ndarray, real, shape (..., 3)
        Angular vector.
    target_vector: ndarray, real, shape (..., 3)
        Target vector.
    t: ndarray, real
        Time array corresponding to the size of the spin vector. Default is None.
    """
    q = quaternion.optimal_alignment_in_Euclidean_metric(vector, target_vector, t=t).inverse()

    return scri.bms_transformations.BMSTransformation(frame_rotation=q.components)


def rotation_to_map_to_superrest_frame(abd, target_strain=None, N_itr_max=10, rel_err_tol=1e-12, print_conv=False):
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
    target_strain: WaveformModes, optional
        WaveformModes object from which the target angular velocity will be computed. Default is None.
    N_itr_max: int, defaults to 10
        Maximum number of iterations to perform. Default is 10.
    rel_err_tol: array_like, defaults to 1e-12
        First value is minimum relative error tolerance between transformation iterations; second value is the
        minimum relative error tolerance between the NR angular velocity and the target angular velocity.
        Default is 1e-12.
    print_conv: bool, defaults to False
        Whether or not to print the termination criterion. Default is False.
    """
    rotation_transformation = scri.bms_transformations.BMSTransformation()
    best_rotation_transformation = scri.bms_transformations.BMSTransformation()

    # if there is no target_strain, just map the spin charge to the z-axis;
    # otherwise, align the angular momentum fluxes over the full window.
    if target_strain is not None:
        target_news = MT_to_WM(
            scri.ModesTimeSeries(
                sf.SWSH_modes.Modes(
                    target_strain.data_dot,
                    spin_weight=-2,
                    ell_min=2,
                    ell_max=target_strain.ell_max,
                    multiplication_truncator=max,
                ),
                time=target_strain.t,
            ).dot,
            dataType=scri.hdot,
        )

        target_omega = target_news.angular_velocity()
        target_omega_spline = CubicSpline(
            target_strain.t, target_omega / np.linalg.norm(target_omega, axis=-1)[:, None]
        )

        itr = 0
        rel_err = np.inf
        rel_errs = [np.inf]
        while itr < N_itr_max and not rel_err < rel_err_tol:
            if itr == 0:
                abd_prime = abd.copy()
                news = MT_to_WM(2.0 * abd.sigma.bar.dot, dataType=scri.hdot)
                omega = news.angular_velocity()
                omega = omega / np.linalg.norm(omega, axis=-1)[:, None]

            rotation_transformation = (
                rotation_from_vectors(omega, target_omega_spline(news.t), news.t) * rotation_transformation
            ).reorder(["supertranslation", "frame_rotation", "boost_velocity"])
            # remove supertranslation and CoM components
            rotation_transformation.supertranslation *= 0
            rotation_transformation.boost_velocity *= 0

            abd_prime = abd.transform(frame_rotation=rotation_transformation.frame_rotation.components)

            news = MT_to_WM(2.0 * abd_prime.sigma.bar.dot, dataType=scri.hdot)
            omega = news.angular_velocity()
            omega = omega / np.linalg.norm(omega, axis=-1)[:, None]

            rel_err = integrate(np.linalg.norm(omega - target_omega_spline(news.t), axis=-1), news.t)[-1] / (
                abd_prime.t[-1] - abd_prime.t[0]
            )
            if rel_err < min(rel_errs):
                best_rotation_transformation = rotation_transformation.copy()
            rel_errs.append(rel_err)

            itr += 1
    else:
        itr = 0
        rel_err = np.inf
        rel_errs = [np.inf]
        while itr < N_itr_max and not rel_err < rel_err_tol:
            if itr == 0:
                abd_prime = abd.copy()
                chi_prime = abd_prime.bondi_dimensionless_spin()
                chi_prime = chi_prime / np.linalg.norm(chi_prime, axis=-1)[:, None]

            rotation_transformation = (
                rotation_from_spin_charge(chi_prime, abd_prime.t) * rotation_transformation
            ).reorder(["supertranslation", "frame_rotation", "boost_velocity"])
            # remove supertranslation and CoM components
            rotation_transformation.supertranslation *= 0
            rotation_transformation.boost_velocity *= 0

            abd_prime = abd.transform(frame_rotation=rotation_transformation.frame_rotation.components)

            chi_prime = abd_prime.bondi_dimensionless_spin()
            chi_prime = chi_prime / np.linalg.norm(chi_prime, axis=-1)[:, None]

            rel_err = integrate(np.linalg.norm(chi_prime - [[0, 0, 1]] * abd_prime.t.size, axis=-1), abd_prime.t)[
                -1
            ] / (abd_prime.t[-1] - abd_prime.t[0])
            if rel_err < min(rel_errs):
                best_rotation_transformation = rotation_transformation.copy()
            rel_errs.append(rel_err)

            itr += 1

    if print_conv:
        if not itr < N_itr_max:
            print(
                f"rotation: maximum number of iterations reached; the min error was {min(np.array(rel_errs).flatten())}."
            )
        else:
            print(f"rotation: tolerance achieved in {itr} iterations!")

    return best_rotation_transformation, rel_errs


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


def rotation(abd, phi=0):
    """Rotate an abd object.

    This is faster than using abd.transform().
    """
    q = quaternion.from_rotation_vector(-phi * np.array([0, 0, 1]))

    h = MT_to_WM(2.0 * abd.sigma.bar, False, scri.h)
    Psi4 = MT_to_WM(0.5 * (-np.sqrt(2)) ** 4 * abd.psi4, False, scri.psi4)
    Psi3 = MT_to_WM(0.5 * (-np.sqrt(2)) ** 3 * abd.psi3, False, scri.psi3)
    Psi2 = MT_to_WM(0.5 * (-np.sqrt(2)) ** 2 * abd.psi2, False, scri.psi2)
    Psi1 = MT_to_WM(0.5 * (-np.sqrt(2)) ** 1 * abd.psi1, False, scri.psi1)
    Psi0 = MT_to_WM(0.5 * (-np.sqrt(2)) ** 0 * abd.psi0, False, scri.psi0)

    h.rotate_physical_system(q)
    Psi4.rotate_physical_system(q)
    Psi3.rotate_physical_system(q)
    Psi2.rotate_physical_system(q)
    Psi1.rotate_physical_system(q)
    Psi0.rotate_physical_system(q)

    abd_rot = abd.copy()
    abd_rot.sigma = 0.5 * WM_to_MT(h).bar
    abd_rot.psi4 = 2 * (-1.0 / np.sqrt(2)) ** 4 * WM_to_MT(Psi4)
    abd_rot.psi3 = 2 * (-1.0 / np.sqrt(2)) ** 3 * WM_to_MT(Psi3)
    abd_rot.psi2 = 2 * (-1.0 / np.sqrt(2)) ** 2 * WM_to_MT(Psi2)
    abd_rot.psi1 = 2 * (-1.0 / np.sqrt(2)) ** 1 * WM_to_MT(Psi1)
    abd_rot.psi0 = 2 * (-1.0 / np.sqrt(2)) ** 0 * WM_to_MT(Psi0)

    return abd_rot


def rel_err_for_abd_in_superrest(abd, target_PsiM, target_strain):
    G = abd.bondi_CoM_charge() / abd.bondi_four_momentum()[:, 0, None]
    rel_err_CoM_transformation = integrate(np.linalg.norm(G, axis=-1), abd.t)[-1] / (abd.t[-1] - abd.t[0])

    if target_strain is not None:
        target_news = MT_to_WM(
            scri.ModesTimeSeries(
                sf.SWSH_modes.Modes(
                    target_strain.data_dot,
                    spin_weight=-2,
                    ell_min=2,
                    ell_max=target_strain.ell_max,
                    multiplication_truncator=max,
                ),
                time=target_strain.t,
            ).dot,
            dataType=scri.hdot,
        )

        target_omega = target_news.angular_velocity()
        target_omega_spline = CubicSpline(
            target_strain.t, target_omega / np.linalg.norm(target_omega, axis=-1)[:, None]
        )

        news = MT_to_WM(2.0 * abd.sigma.bar.dot, dataType=scri.hdot)
        omega = news.angular_velocity()
        omega = omega / np.linalg.norm(omega, axis=-1)[:, None]

        rel_err_rotation = integrate(np.linalg.norm(omega - target_omega_spline(news.t), axis=-1), news.t)[-1] / (
            abd.t[-1] - abd.t[0]
        )
    else:
        spin = abd.bondi_dimensionless_spin()
        spin = spin / np.linalg.norm(spin, axis=-1)[:, None]
        rel_err_rotation = integrate(np.linalg.norm(spin - [[0, 0, 1]] * abd.t.size, axis=-1), abd.t)[-1] / (
            abd.t[-1] - abd.t[0]
        )

    if target_PsiM is not None:
        rel_err_PsiM = np.linalg.norm(
            np.array(abd.supermomentum("Moreschi"))[np.argmin(abs(abd.t - 0)), 4:]
            - np.array(target_PsiM.data)[np.argmin(abs(target_PsiM.t - 0)), 4:]
        )
    else:
        rel_err_PsiM = np.linalg.norm(np.array(abd.supermomentum("Moreschi"))[np.argmin(abs(abd.t - 0)), 4:])

    return rel_err_CoM_transformation, rel_err_rotation, rel_err_PsiM


def map_to_superrest_frame(
    self,
    t_0=0,
    target_PsiM_input=None,
    target_strain_input=None,
    padding_time=250,
    N_itr_maxes={
        "superrest": 2,
        "CoM_transformation": 10,
        "rotation": 10,
        "supertranslation": 10,
    },
    rel_err_tols={"CoM_transformation": 1e-12, "rotation": 1e-12, "supertranslation": 1e-12},
    order=["supertranslation", "rotation", "CoM_transformation"],
    ell_max=None,
    alpha_ell_max=None,
    fix_time_phase_freedom=False,
    modes=None,
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
    target_PsiM_input : WaveformModes, optional
        Target Moreschi supermomentum to map to.
        Default is 0.
    target_strain_input : WaveformModes, optional
        Target strain used to constrain the rotation
        freedom via the angular momentum flux.
        Default is aligning to the z-axis.
    padding_time : float, optional
        Amount by which to pad around t_0 to speed up computations, i.e.,
        distance from t_0 in each direction to be included in self.interpolate(...).
        This also determines the range over which certain BMS charges will be computed.
        Default is 250.
    N_itr_maxes : dict, optional
        Maximum number of iterations to perform for each transformation.
        For 'superrest', this is the number of iterations to use for the superrest procedure.
        For the other options, these are the number of iterations to use for finding each individual transformation.
        Default is
        N_itr_maxes = {
            'superrest':          2,
            'CoM_transformation': 10,
            'rotation':           10,
            'supertranslation':   10,
        }.
    rel_err_tols : dict, optional
        Relative error tolerances for each transformation.
        Default is
        rel_err_tols = {
            'CoM_transformation': 1e-12,
            'rotation':           1e-12,
            'supertranslation':   1e-12
        }.
    order : list, optional
        Order in which to solve for the BMS transformations.
        Default is ["rotation", "CoM_transformation", "supertranslation"].
    ell_max : int, optional
        Maximum ell to use for SWSH/Grid transformations.
        Default is self.ell_max.
    alpha_ell_max : int, optional
        Maximum ell of the supertranslation to use.
        Default is self.ell_max.
    fix_time_phase_freedom : bool, optional
        Whether or not to fix the time and phase freedom using a 2d minimization scheme.
        Default is True.
    modes : list, optional
        List of modes to include when performing the 2d alignment.
        Default is every mode.
    print_conv: bool, defaults to False
        Whether or not to print the termination criterion. Default is False.

    Returns
    -------
    abd_prime : AsymptoticBondiData
        Result of self.transform(...) where the input transformations are
        the transformations found in the BMSTransformations object.
    transformations : BMSTransformation
        BMS transformation to map to the target BMS frame.

    """
    abd = self.copy()

    if target_strain_input is not None:
        target_strain = target_strain_input.copy()
        target_strain.t -= t_0
    else:
        target_strain = None

    if target_PsiM_input is not None:
        target_PsiM = target_PsiM_input.copy()
        target_PsiM.t -= t_0
    else:
        target_PsiM = None

    if ell_max is None:
        ell_max = abd.ell_max

    if alpha_ell_max is None:
        alpha_ell_max = ell_max

    abd_interp = abd.interpolate(
        abd.t[np.argmin(abs(abd.t - (t_0 - (padding_time + 200)))) : np.argmin(abs(abd.t - (t_0 + (padding_time + 200)))) + 1]
    )

    # apply a time translation so that we're mapping
    # to the superrest frame at u = 0
    time_translation = scri.bms_transformations.BMSTransformation(supertranslation=[sf.constant_as_ell_0_mode(t_0)])
    BMS_transformation = time_translation * scri.bms_transformations.BMSTransformation().reorder(
        ["supertranslation", "frame_rotation", "boost_velocity"]
    )

    itr = 0
    rel_err = [np.inf, np.inf, np.inf]
    rel_errs = [[np.inf, np.inf, np.inf]]
    best_rel_err = [np.inf, np.inf, np.inf]
    while itr < N_itr_maxes["superrest"]:
        if type(rel_err) == tuple:
            if (
                    rel_err[0] < rel_err_tols["CoM_transformation"]
                    and rel_err[1] < rel_err_tols["rotation"]
                    and rel_err[2] < rel_err_tols["supertranslation"]
            ):
                break
        else:
            pass
            
        if itr == 0:
            abd_interp_prime = abd_interp.transform(
                supertranslation=BMS_transformation.supertranslation,
                frame_rotation=BMS_transformation.frame_rotation.components,
                boost_velocity=BMS_transformation.boost_velocity,
            )

        for transformation in order:
            if transformation == "supertranslation":
                new_transformation, supertranslation_rel_errs = supertranslation_to_map_to_superrest_frame(
                    abd_interp_prime,
                    target_PsiM,
                    N_itr_max=N_itr_maxes["supertranslation"],
                    rel_err_tol=rel_err_tols["supertranslation"],
                    ell_max=ell_max,
                    print_conv=print_conv,
                )
            elif transformation == "rotation":
                new_transformation, rot_rel_errs = rotation_to_map_to_superrest_frame(
                    abd_interp_prime,
                    target_strain=target_strain,
                    N_itr_max=N_itr_maxes["rotation"],
                    rel_err_tol=rel_err_tols["rotation"],
                    print_conv=print_conv,
                )
            elif transformation == "CoM_transformation":
                new_transformation, CoM_rel_errs = com_transformation_to_map_to_superrest_frame(
                    abd_interp_prime,
                    N_itr_max=N_itr_maxes["CoM_transformation"],
                    rel_err_tol=rel_err_tols["CoM_transformation"],
                    print_conv=print_conv,
                )
            elif transformation == "time_phase":
                if target_strain is not None:
                    strain_interp_prime = scri.asymptotic_bondi_data.map_to_superrest_frame.MT_to_WM(
                        2.0 * abd_interp_prime.sigma.bar, dataType=scri.h
                    )

                    rel_err, _, res = align2d(
                        MT_to_WM(WM_to_MT(strain_interp_prime), sxs_version=True),
                        MT_to_WM(WM_to_MT(target_strain), sxs_version=True),
                        0 - padding_time,
                        0 + padding_time,
                        n_brute_force_δt=None,
                        n_brute_force_δϕ=None,
                        include_modes=modes,
                        nprocs=4,
                    )

                    new_transformation = scri.bms_transformations.BMSTransformation(
                        supertranslation=[sf.constant_as_ell_0_mode(res.x[0])],
                        frame_rotation=quaternion.from_rotation_vector(res.x[1] * np.array([0, 0, 1])).components,
                    )

            BMS_transformation = (new_transformation * BMS_transformation).reorder(
                ["supertranslation", "frame_rotation", "boost_velocity"]
            )

            abd_interp_prime = abd_interp.transform(
                supertranslation=BMS_transformation.supertranslation,
                frame_rotation=BMS_transformation.frame_rotation.components,
                boost_velocity=BMS_transformation.boost_velocity,
            )

        if target_strain is not None and order[-1] == "time_phase":
            # rel_err is obtained from align2d, so do nothing
            pass
        else:
            rel_err = rel_err_for_abd_in_superrest(abd_interp_prime, target_PsiM, target_strain)
            
        if np.mean(rel_err) < min([np.mean(r) for r in rel_errs]):
            best_BMS_transformation = BMS_transformation.copy()
            best_rel_err = rel_err
        rel_errs.append(rel_err)

        itr += 1

    if print_conv:
        if not itr < N_itr_maxes["superrest"]:
            print(f"superrest: maximum number of iterations reached; the min error was {best_rel_err}.")
        else:
            print(f"superrest: tolerance achieved in {itr} iterations!")

    # undo the time translation
    best_BMS_transformation = (time_translation.inverse() * best_BMS_transformation).reorder(
        ["supertranslation", "frame_rotation", "boost_velocity"]
    )

    # transform abd
    abd_prime = abd.transform(
        supertranslation=best_BMS_transformation.supertranslation,
        frame_rotation=best_BMS_transformation.frame_rotation.components,
        boost_velocity=best_BMS_transformation.boost_velocity,
    )

    return abd_prime, best_BMS_transformation, best_rel_err
