import numpy as np

import sxs
import scri
import spinsfast
import spherical_functions as sf
from spherical_functions import LM_index

import quaternion
from quaternion.calculus import derivative
from quaternion.calculus import indefinite_integral as integrate

from scipy.interpolate import CubicSpline

from sxs.waveforms.alignment import align2d

from functools import partial

from scri.asymptotic_bondi_data.map_to_superrest_frame import time_translation, rotation, MT_to_WM, WM_to_MT

def rel_err_between_abds(abd1, abd2, t1, t2):
    t_array = abd1.t[np.argmin(abs(abd1.t - t1)) : np.argmin(abs(abd1.t - t2)) + 1]

    abd1_interp = abd1.interpolate(t_array)
    abd2_interp = abd2.interpolate(t_array)

    abd_diff = abd1_interp.copy()
    abd_diff.sigma -= abd2_interp.sigma
    abd_diff.psi0 -= abd2_interp.psi0
    abd_diff.psi1 -= abd2_interp.psi1
    abd_diff.psi2 -= abd2_interp.psi2
    abd_diff.psi3 -= abd2_interp.psi3
    abd_diff.psi4 -= abd2_interp.psi4

    rel_err = 0
    rel_err += integrate(abd_diff.sigma.norm(), abd_diff.t)[-1] / (
        integrate(abd1_interp.sigma.norm(), abd1_interp.t)[-1] or (abd1_interp.t[-1] - abd1_interp.t[0])
    )
    rel_err += integrate(abd_diff.psi0.norm(), abd_diff.t)[-1] / (
        integrate(abd1_interp.psi0.norm(), abd1_interp.t)[-1] or (abd1_interp.t[-1] - abd1_interp.t[0])
    )
    rel_err += integrate(abd_diff.psi1.norm(), abd_diff.t)[-1] / (
        integrate(abd1_interp.psi1.norm(), abd1_interp.t)[-1] or (abd1_interp.t[-1] - abd1_interp.t[0])
    )
    rel_err += integrate(abd_diff.psi2.norm(), abd_diff.t)[-1] / (
        integrate(abd1_interp.psi2.norm(), abd1_interp.t)[-1] or (abd1_interp.t[-1] - abd1_interp.t[0])
    )
    rel_err += integrate(abd_diff.psi3.norm(), abd_diff.t)[-1] / (
        integrate(abd1_interp.psi3.norm(), abd1_interp.t)[-1] or (abd1_interp.t[-1] - abd1_interp.t[0])
    )
    rel_err += integrate(abd_diff.psi4.norm(), abd_diff.t)[-1] / (
        integrate(abd1_interp.psi4.norm(), abd1_interp.t)[-1] or (abd1_interp.t[-1] - abd1_interp.t[0])
    )
    rel_err /= 6

    return rel_err


def map_to_abd_frame(
    self,
    target_abd,
    t_0=0,
    padding_time=250,
    N_itr_maxes={
        "abd": 2,
        "superrest": 2,
        "CoM_transformation": 10,
        "rotation": 10,
        "supertranslation": 10,
    },
    rel_err_tols={"CoM_transformation": 1e-12, "rotation": 1e-12, "supertranslation": 1e-12},
    order=["supertranslation", "rotation", "CoM_transformation"],
    ell_max=None,
    alpha_ell_max=None,
    fix_time_phase_freedom=True,
    nprocs=4,
    print_conv=False,
):
    """Transform an abd object to a target abd object using data at t=0.

    This computes the transformations necessary to map an abd object to a target abd object.
    It uses the function map_to_bms_frame with the target charges computed from the target abd object.

    Parameters
    ----------
    target_abd : AsymptoticBondiData
        Target AsymptoticBondiData to map to.
    t_0: float, optional
        When to map to the target BMS frame.
        Default is 0.
    padding_time : float, optional
        Amount by which to pad around t=0 to speed up computations, i.e.,
        distance from t=0 in each direction to be included in self.interpolate(...).
        This also determines the range over which certain BMS charges will be computed.
        Default is 250.
    N_itr_maxes : dict, optional
        Maximum number of iterations to perform for each transformation.
        For 'abd' , this is the number of iterations to use for the abd procedure.
        For 'superrest', this is the number of iterations to use for the superrest procedure.
        For the other options, these are the number of iterations to use for finding each individual transformation.
        Default is
        N_itr_maxes = {
            'abd':                2,
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
        Default is ["supertranslation", "rotation", "CoM_transformation"].
    ell_max : int, optional
        Maximum ell to use for SWSH/Grid transformations.
        Default is self.ell_max.
    alpha_ell_max : int, optional
        Maximum ell of the supertranslation to use.
        Default is self.ell_max.
    fix_time_phase_freedom : bool, optional
        Whether or not to fix the time and phase freedom using a 2d minimization scheme.
        Default is True.
    nprocs : int, optional
        Number of cpus to use during parallelization for fixing the time and phase freedom.
        Default is 4. 'None' corresponds to the maximum number. '-1' corresponds to no parallelization.
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

    target_strain = scri.asymptotic_bondi_data.map_to_superrest_frame.MT_to_WM(
        2.0 * target_abd.sigma.bar, dataType=scri.h
    )

    time_translation = scri.bms_transformations.BMSTransformation()
    if fix_time_phase_freedom:
        # ensure that they are reasonable close
        energy = abd.bondi_four_momentum()[:, 0]
        target_energy = target_abd.bondi_four_momentum()[:, 0]
        time_translation = scri.bms_transformations.BMSTransformation(
            supertranslation=[
                sf.constant_as_ell_0_mode(
                    abd.t[np.argmin(abs(energy - target_energy[np.argmin(abs(target_abd.t - t_0))]))] - t_0
                )
            ]
        )

    BMS_transformation = (time_translation * scri.bms_transformations.BMSTransformation()).reorder(
        ["supertranslation", "frame_rotation", "boost_velocity"]
    )

    abd_interp = abd.interpolate(
        abd.t[
            np.argmin(abs(abd.t - (t_0 - 1.5 * padding_time))) : np.argmin(abs(abd.t - (t_0 + 1.5 * padding_time))) + 1
        ]
    )

    target_abd_superrest, transformation2, rel_err2 = target_abd.map_to_superrest_frame(
        t_0=t_0,
        padding_time=padding_time,
        N_itr_maxes=N_itr_maxes,
        rel_err_tols=rel_err_tols,
        ell_max=ell_max,
        alpha_ell_max=alpha_ell_max,
        print_conv=print_conv,
        order=order,
    )

    target_strain_superrest = scri.asymptotic_bondi_data.map_to_superrest_frame.MT_to_WM(
        2.0 * target_abd_superrest.sigma.bar, dataType=scri.h
    )

    itr = 0
    rel_err = np.inf
    rel_errs = [np.inf]
    while itr < N_itr_maxes["abd"]:
        if itr == 0:
            abd_interp_prime = abd_interp.transform(
                supertranslation=BMS_transformation.supertranslation,
                frame_rotation=BMS_transformation.frame_rotation.components,
                boost_velocity=BMS_transformation.boost_velocity,
            )

        # find the transformations that map to the superrest frame
        abd_interp_superrest, transformation1, rel_err1 = abd_interp_prime.map_to_superrest_frame(
            t_0=t_0,
            padding_time=padding_time,
            N_itr_maxes=N_itr_maxes,
            rel_err_tols=rel_err_tols,
            ell_max=ell_max,
            alpha_ell_max=alpha_ell_max,
            print_conv=print_conv,
            order=order,
        )

        if fix_time_phase_freedom:
            # 2d align now that we're in the superrest frame
            strain_interp_superrest = scri.asymptotic_bondi_data.map_to_superrest_frame.MT_to_WM(
                2.0 * abd_interp_superrest.sigma.bar, dataType=scri.h
            )

            rel_err, _, res = align2d(
                MT_to_WM(WM_to_MT(strain_interp_superrest), sxs_version=True),
                MT_to_WM(WM_to_MT(target_strain_superrest), sxs_version=True),
                t_0 - padding_time,
                t_0 + padding_time,
                n_brute_force_δt=None,
                n_brute_force_δϕ=None,
                include_modes=None,
                nprocs=nprocs,
            )

            time_phase_transformation = scri.bms_transformations.BMSTransformation(
                supertranslation=[sf.constant_as_ell_0_mode(res.x[0])],
                frame_rotation=quaternion.from_rotation_vector(res.x[1] * np.array([0, 0, 1])).components,
            )
        else:
            time_phase_transformation = scri.bms_transformations.BMSTransformation()

        # compose these transformations in the right order
        BMS_transformation = (
            transformation2.inverse() * (time_phase_transformation * (transformation1 * BMS_transformation))
        ).reorder(["supertranslation", "frame_rotation", "boost_velocity"])

        # obtain the transformed abd object
        abd_interp_prime = abd_interp.transform(
            supertranslation=BMS_transformation.supertranslation,
            frame_rotation=BMS_transformation.frame_rotation.components,
            boost_velocity=BMS_transformation.boost_velocity,
        )

        if fix_time_phase_freedom:
            # find the time/phase transformations
            strain_interp_prime = scri.asymptotic_bondi_data.map_to_superrest_frame.MT_to_WM(
                2.0 * abd_interp_prime.sigma.bar, dataType=scri.h
            )

            rel_err, _, res = align2d(
                MT_to_WM(WM_to_MT(strain_interp_prime), sxs_version=True),
                MT_to_WM(WM_to_MT(target_strain), sxs_version=True),
                t_0 - padding_time,
                t_0 + padding_time,
                n_brute_force_δt=None,
                n_brute_force_δϕ=None,
                include_modes=None,
                nprocs=nprocs,
            )

            time_phase_transformation = scri.bms_transformations.BMSTransformation(
                supertranslation=[sf.constant_as_ell_0_mode(res.x[0])],
                frame_rotation=quaternion.from_rotation_vector(res.x[1] * np.array([0, 0, 1])).components,
            )

            BMS_transformation = (time_phase_transformation * BMS_transformation).reorder(
                ["supertranslation", "frame_rotation", "boost_velocity"]
            )

            # obtain the transformed abd object
            abd_interp_prime = abd_interp.transform(
                supertranslation=BMS_transformation.supertranslation,
                frame_rotation=BMS_transformation.frame_rotation.components,
                boost_velocity=BMS_transformation.boost_velocity,
            )
        else:
            rel_err = rel_err_between_abds(target_abd, abd_interp_prime, t_0 - padding_time, t_0 + padding_time)

        if rel_err < min(rel_errs):
            best_BMS_transformation = BMS_transformation.copy()
            best_rel_err = rel_err
        rel_errs.append(rel_err)

        itr += 1

    if print_conv:
        if not itr < N_itr_maxes["abd"]:
            print(f"BMS: maximum number of iterations reached; the min error was {best_rel_err}.")
        else:
            print(f"BMS: tolerance achieved in {itr} iterations!")

    abd_prime = abd.transform(
        supertranslation=best_BMS_transformation.supertranslation,
        frame_rotation=best_BMS_transformation.frame_rotation.components,
        boost_velocity=best_BMS_transformation.boost_velocity,
    )

    return abd_prime, best_BMS_transformation, best_rel_err
