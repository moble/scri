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

from functools import partial

from scri.asymptotic_bondi_data.map_to_superrest_frame import time_translation, rotation


def cost(δt_δϕ, args):
    modes_A, modes_B, t_reference, δϕ_factor, δΨ_factor, normalization = args

    # Take the sqrt because least_squares squares the inputs...
    diff = integrate(
        np.sum(
            abs(modes_A(t_reference + δt_δϕ[0]) * np.exp(1j * δt_δϕ[1]) ** δϕ_factor * δΨ_factor - modes_B) ** 2, axis=1
        ),
        t_reference,
    )[-1]
    return np.sqrt(diff / normalization)


def align2d(h_A, h_B, t1, t2, n_brute_force_δt=None, n_brute_force_δϕ=None, include_modes=None, nprocs=4):
    """Align waveforms by shifting in time and phase

    This function determines the optimal time and phase offset to apply to `h_A` by minimizing
    the averaged (over time) L² norm (over the sphere) of the difference of the h_Aveforms.

    The integral is taken from time `t1` to `t2`.

    Note that the input waveforms are assumed to be initially aligned at least well
    enough that:

      1) the time span from `t1` to `t2` in the two waveforms will overlap at
         least slightly after the second waveform is shifted in time; and
      2) waveform `h_B` contains all the times corresponding to `t1` to `t2`
         in waveform `h_A`.

    The first of these can usually be assured by simply aligning the peaks prior to
    calling this function:

        h_A.t -= h_A.max_norm_time() - h_B.max_norm_time()

    The second assumption will be satisfied as long as `t1` is not too close to the
    beginning of `h_B` and `t2` is not too close to the end.

    Parameters
    ----------
    h_A : WaveformModes
    h_B : WaveformModes
        Waveforms to be aligned
    t1 : float
    t2 : float
        Beginning and end of integration interval
    n_brute_force_δt : int, optional
        Number of evenly spaced δt values between (t1-t2) and (t2-t1) to sample
        for the initial guess.  By default, this is just the maximum number of
        time steps in the range (t1, t2) in the input waveforms.  If this is
        too small, an incorrect local minimum may be found.
    n_brute_force_δϕ : int, optional
        Number of evenly spaced δϕ values between 0 and 2π to sample
        for the initial guess.  By default, this is 2 * ell_max + 1.
    include_modes: list, optional
        A list containing the (ell, m) modes to be included in the L² norm.
    nprocs: int, optional
        Number of cpus to use.
        Default is 4. 'None' corresponds to the maximum number. '-1' corresponds to no parallelization.

    Returns
    -------
    optimum: OptimizeResult
        Result of scipy.optimize.least_squares
    h_A_prime: WaveformModes
        Resulting waveform after transforming `h_A` using `optimum`

    Notes
    -----
    Choosing the time interval is usually the most difficult choice to make when
    aligning waveforms.  Assuming you want to align during inspiral, the times
    must span sufficiently long that the waveforms' norm (equivalently, orbital
    frequency changes) significantly from `t1` to `t2`.  This means that you
    cannot always rely on a specific number of orbits, for example.  Also note
    that neither number should be too close to the beginning or end of either
    waveform, to provide some "wiggle room".

    Precession generally causes no problems for this function.  In principle,
    eccentricity, center-of-mass offsets, boosts, or other supertranslations could
    cause problems, but this function begins with a brute-force method of finding
    the optimal time offset that will avoid local minima in all but truly
    outrageous situations.  In particular, as long as `t1` and `t2` are separated
    by enough, there should never be a problem.

    """
    from scipy.optimize import least_squares

    import multiprocessing as mp

    h_A_copy = h_A.copy()
    h_B_copy = h_B.copy()

    # Check that (t1, t2) makes sense and is actually contained in both waveforms
    if t2 <= t1:
        raise ValueError(f"(t1,t2)=({t1}, {t2}) is out of order")
    if h_A_copy.t[0] > t1 or h_A_copy.t[-1] < t2:
        raise ValueError(
            f"(t1,t2)=({t1}, {t2}) not contained in h_A_copy.t, which spans ({h_A_copy.t[0]}, {h_A_copy.t[-1]})"
        )
    if h_B_copy.t[0] > t1 or h_B_copy.t[-1] < t2:
        raise ValueError(
            f"(t1,t2)=({t1}, {t2}) not contained in h_B_copy.t, which spans ({h_B_copy.t[0]}, {h_B_copy.t[-1]})"
        )

    # Figure out time offsets to try
    δt_lower = max(t1 - t2, h_A_copy.t[0] - t1)
    δt_upper = min(t2 - t1, h_A_copy.t[-1] - t2)

    # We'll start by brute forcing, sampling time offsets evenly at as many
    # points as there are time steps in (t1,t2) in the input waveforms
    if n_brute_force_δt is None:
        n_brute_force_δt = max(
            sum((h_A_copy.t >= t1) & (h_A_copy.t <= t2)), sum((h_B_copy.t >= t1) & (h_B_copy.t <= t2))
        )
    δt_brute_force = np.linspace(δt_lower, δt_upper, num=n_brute_force_δt)

    if n_brute_force_δϕ is None:
        n_brute_force_δϕ = 2 * h_A_copy.ell_max + 1
    δϕ_brute_force = np.linspace(0, 2 * np.pi, n_brute_force_δϕ, endpoint=False)

    δt_δϕ_brute_force = np.array(np.meshgrid(δt_brute_force, δϕ_brute_force)).T.reshape(-1, 2)

    t_reference = h_B_copy.t[np.argmin(abs(h_B_copy.t - t1)) : np.argmin(abs(h_B_copy.t - t2)) + 1]

    # Remove certain modes, if requested
    ell_max = min(h_A_copy.ell_max, h_B_copy.ell_max)
    if include_modes != None:
        for L in range(2, ell_max + 1):
            for M in range(-L, L + 1):
                if not (L, M) in include_modes:
                    h_A_copy.data[:, LM_index(L, M, h_A_copy.ell_min)] *= 0
                    h_B_copy.data[:, LM_index(L, M, h_B_copy.ell_min)] *= 0

    # Define the cost function
    modes_A = CubicSpline(h_A_copy.t, h_A_copy[:, 2 : ell_max + 1].data)
    modes_B = CubicSpline(h_B_copy.t, h_B_copy[:, 2 : ell_max + 1].data)(t_reference)

    normalization = integrate(CubicSpline(h_B_copy.t, h_B_copy[:, 2 : ell_max + 1].norm())(t_reference), t_reference)[
        -1
    ]
    if normalization == 0:
        normalization = t_reference[-1] - t_reference[0]

    δϕ_factor = np.array([M for L in range(h_A_copy.ell_min, ell_max + 1) for M in range(-L, L + 1)])

    optimums = []
    h_A_primes = []
    for δΨ_factor in [-1, +1]:
        # Optimize by brute force with multiprocessing
        cost_wrapper = partial(cost, args=[modes_A, modes_B, t_reference, δϕ_factor, δΨ_factor, normalization])

        if nprocs == -1:
            cost_brute_force = [cost_wrapper(δt_δϕ) for δt_δϕ in δt_δϕ_brute_force]
        else:
            if nprocs is None:
                nprocs = mp.cpu_count()

            pool = mp.Pool(processes=nprocs)
            cost_brute_force = pool.map(cost_wrapper, δt_δϕ_brute_force)
            pool.close()
            pool.join()

        δt_δϕ = δt_δϕ_brute_force[np.argmin(cost_brute_force)]

        # Optimize explicitly
        optimum = least_squares(cost_wrapper, δt_δϕ, bounds=[(δt_lower, 0), (δt_upper, 2 * np.pi)], max_nfev=50000)
        optimums.append(optimum)

        h_A_prime = h_A.copy()
        h_A_prime.t = h_A.t - optimum.x[0]
        h_A_prime.data = h_A[:, 2 : ell_max + 1].data * np.exp(1j * optimum.x[1]) ** δϕ_factor * δΨ_factor
        h_A_prime.ell_min = 2
        h_A_prime.ell_max = ell_max
        h_A_primes.append(h_A_prime)

    idx = np.argmin(abs(np.array([optimum.cost for optimum in optimums])))

    return h_A_primes[idx], optimums[idx].fun, optimums[idx]


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

    time_translation = scri.bms_transformations.BMSTransformation()
    if fix_time_phase_freedom:
        # ensure that they are reasonable close
        time_translation = scri.bms_transformations.BMSTransformation(
            supertranslation=[
                sf.constant_as_ell_0_mode(
                    abd.t[np.argmax(abd.bondi_four_momentum()[:, 0])]
                    - target_abd.t[np.argmax(target_abd.bondi_four_momentum()[:, 0])]
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
        t_0=t_0, padding_time=padding_time
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
            t_0=t_0, padding_time=padding_time
        )

        # compose these transformations in the right order
        BMS_transformation = (transformation2.inverse() * transformation1 * BMS_transformation).reorder(
            ["supertranslation", "frame_rotation", "boost_velocity"]
        )

        # obtain the transformed abd object
        abd_interp_prime = abd_interp.transform(
            supertranslation=BMS_transformation.supertranslation,
            frame_rotation=BMS_transformation.frame_rotation.components,
            boost_velocity=BMS_transformation.boost_velocity,
        )

        if fix_time_phase_freedom:
            # find the time/phase transformations
            news_interp_prime = scri.asymptotic_bondi_data.map_to_superrest_frame.MT_to_WM(
                2.0 * abd_interp_prime.sigma.bar, dataType=scri.h
            )
            target_news = scri.asymptotic_bondi_data.map_to_superrest_frame.MT_to_WM(
                2.0 * target_abd.sigma.bar, dataType=scri.h
            )

            _, rel_err, res = scri.asymptotic_bondi_data.map_to_abd_frame.align2d(
                news_interp_prime,
                target_news,
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
        if not itr < N_itr_max:
            print(f"BMS: maximum number of iterations reached; the min error was {best_rel_err}.")
        else:
            print(f"BMS: tolerance achieved in {itr} iterations!")

    abd_prime = abd.transform(
        supertranslation=best_BMS_transformation.supertranslation,
        frame_rotation=best_BMS_transformation.frame_rotation.components,
        boost_velocity=best_BMS_transformation.boost_velocity,
    )

    return abd_prime, best_BMS_transformation, best_rel_err
