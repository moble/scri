# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

import math
import numpy as np
import quaternion
import spherical_functions as sf
from .waveform_base import waveform_alterations
from .mode_calculations import corotating_frame, angular_velocity, LLDominantEigenvector
from . import jit, Coprecessing, Coorbital, Corotating, Inertial


@waveform_alterations
def to_coprecessing_frame(W, RoughDirection=np.array([0.0, 0.0, 1.0]), RoughDirectionIndex=None):
    """Transform waveform (in place) to a coprecessing frame

    Parameters
    ==========
    W: waveform
        Waveform object to be transformed in place.
    RoughDirection: 3-array [defaults to np.array([0.0, 0.0, 1.0])]
        Vague guess about the preferred initial axis, to choose the sign of the eigenvectors.
    RoughDirectionIndex: int or None [defaults to None]
        Time index at which to apply RoughDirection guess.

    """
    if RoughDirectionIndex is None:
        RoughDirectionIndex = W.n_times // 8
    dpa = LLDominantEigenvector(W, RoughDirection=RoughDirection, RoughDirectionIndex=RoughDirectionIndex)
    R = np.array([quaternion.quaternion.sqrt(-quaternion.quaternion(0, *q).normalized() * quaternion.z) for q in dpa])
    R = quaternion.minimal_rotation(R, W.t, iterations=3)
    W.rotate_decomposition_basis(R)
    W._append_history(f"{W}.to_coprecessing_frame({RoughDirection}, {RoughDirectionIndex})")
    W.frameType = Coprecessing
    return W


@waveform_alterations
def to_corotating_frame(
    W, R0=quaternion.one, tolerance=1e-12, z_alignment_region=None, return_omega=False, truncate_log_frame=False
):
    """Transform waveform (in place) to a corotating frame

    Parameters
    ==========
    W: waveform
        Waveform object to be transformed in place
    R0: quaternion [defaults to 1]
        Initial value of frame when integrating angular velocity
    tolerance: float [defaults to 1e-12]
        Absolute tolerance used in integration of angular velocity
    z_alignment_region: None or 2-tuple of floats [defaults to None]
        If not None, the dominant eigenvector of the <LL> matrix is aligned with the z axis,
        averaging over this portion of the data.  The first and second elements of the input are
        considered fractions of the inspiral at which to begin and end the average.  For example,
        (0.1, 0.9) would lead to starting 10% of the time from the first time step to the max norm
        time, and ending at 90% of that time.
    return_omega: bool [defaults to False]
        If True, return a 2-tuple consisting of the waveform in the corotating frame (the usual
        returned object) and the angular-velocity data.  That is frequently also needed, so this is
        just a more efficient way of getting the data.
    truncate_log_frame: bool [defaults to False]
        If True, set bits of log(frame) with lower significance than `tolerance` to zero, and use
        exp(truncated(log(frame))) to rotate the waveform.  Also returns `log_frame` along with the
        waveform (and optionally `omega`)

    """
    frame, omega = corotating_frame(
        W, R0=R0, tolerance=tolerance, z_alignment_region=z_alignment_region, return_omega=True
    )
    if truncate_log_frame:
        log_frame = quaternion.as_float_array(np.log(frame))
        power_of_2 = 2 ** int(-np.floor(np.log2(2 * tolerance)))
        log_frame = np.round(log_frame * power_of_2) / power_of_2
        frame = np.exp(quaternion.as_quat_array(log_frame))
    W.rotate_decomposition_basis(frame)
    W._append_history(
        f"{W}.to_corotating_frame({R0}, {tolerance}, {z_alignment_region}, {return_omega}, {truncate_log_frame})"
    )
    W.frameType = Corotating
    if return_omega:
        if truncate_log_frame:
            return (W, omega, log_frame)
        else:
            return (W, omega)
    else:
        if truncate_log_frame:
            return (W, log_frame)
        else:
            return W


@waveform_alterations
def to_inertial_frame(W):
    W.rotate_decomposition_basis(~W.frame)
    W._append_history(f"{W}.to_inertial_frame()")
    W.frameType = Inertial
    return W


def get_alignment_of_decomposition_frame_to_modes(w, t_fid, nHat_t_fid=quaternion.x, ell_max=None):
    """Find the appropriate rotation to fix the attitude of the corotating frame.

    This function simply finds the rotation necessary to align the
    corotating frame to the waveform at the fiducial time, rather than
    applying it.  This is called by `AlignDecompositionFrameToModes`
    and probably does not need to be called directly; see that
    function's documentation for more details.

    Parameters
    ----------
    w: WaveformModes
        Object to be aligned
    t_fid: float
        Fiducial time at which the alignment should happen
    nHat_t_fid: quaternion (optional)
        The approximate direction of nHat at t_fid; defaults to x
    ell_max: int
       Highest ell mode to use in computing the <LL> matrix

    """
    # We seek that R_c such that R_corot(t_fid)*R_c rotates the z axis
    # onto V_f.  V_f measured in this frame is given by
    #     V_f = R_V_f * Z * R_V_f.conjugate(),
    # (note Z rather than z) where R_V_f is found below.  But
    #     Z = R_corot * z * R_corot.conjugate(),
    # so in the (x,y,z) frame,
    #     V_f = R_V_f * R_corot * z * R_corot.conjugate() * R_V_f.conjugate().
    # Now, this is the standard composition for rotating physical
    # vectors.  However, rotation of the basis behaves oppositely, so
    # we want R_V_f as our constant rotation, applied as a rotation of
    # the decomposition basis.  We also want to rotate so that the
    # phase of the (2,2) mode is zero at t_fid.  This can be achieved
    # with an initial rotation.

    if ell_max is None:
        ell_max = w.ell_max

    if w.frameType not in [Coprecessing, Coorbital, Corotating]:
        message = (
            "get_alignment_of_decomposition_frame_to_modes only takes Waveforms in the "
            + "coprecessing, coorbital, or corotating frames.  This Waveform is in the "
            + "'{0}' frame."
        )
        raise ValueError(message.format(w.frame_type_string))

    if w.frame.size != w.n_times:
        message = (
            "get_alignment_of_decomposition_frame_to_modes requires full information about the Waveform's frame."
            + "This Waveform has {0} time steps, but only {1} rotors in its frame."
        )
        raise ValueError(message.format(w.n_times, np.asarray(w.frame).size))

    if t_fid < w.t[0] or t_fid > w.t[-1]:
        message = "The requested alignment time t_fid={0} is outside the range of times in this waveform ({1}, {2})."
        raise ValueError(message.format(t_fid, w.t[0], w.t[-1]))

    # Get direction of angular-velocity vector near t_fid
    i_t_fid = (w.t <= t_fid).nonzero()[0][-1]  # Find the largest index i with t[i-1] <= t_fid
    if i_t_fid < w.t.size - 1:
        i_t_fid += 1
    i1 = 0 if i_t_fid - 5 < 0 else i_t_fid - 5
    i2 = w.t.size if i1 + 11 > w.t.size else i1 + 11
    Region = w[i1:i2, 2].copy().to_inertial_frame()
    omegaHat = quaternion.quaternion(0, *(angular_velocity(Region)[i_t_fid - i1])).normalized()

    # omegaHat contains the components of that vector relative to the
    # inertial frame.  To get its components in this Waveform's
    # (possibly rotating) frame, we need to rotate it by the inverse
    # of this Waveform's `frame` data:
    if w.frame.size > 1:
        R = w.frame[i_t_fid]
        omegaHat = R.inverse() * omegaHat * R
    elif w.frame.size == 1:
        R = w.frame[0]
        omegaHat = R.inverse() * omegaHat * R

    # Interpolate the Waveform to t_fid
    Instant = w[i1:i2].copy().interpolate(np.array([t_fid,]))
    R_f0 = Instant.frame[0]

    # V_f is the dominant eigenvector of <LL>, suggested by O'Shaughnessy et al.
    V_f = quaternion.quaternion(0, *(LLDominantEigenvector(Instant[:, : ell_max + 1])[0])).normalized()
    V_f_aligned = -V_f if np.dot(omegaHat.vec, V_f.vec) < 0 else V_f

    # R_V_f is the rotor taking the Z axis onto V_f
    R_V_f = (-V_f_aligned * quaternion.z).sqrt()
    # INFOTOCERR << omegaHat << "\n"
    #            << V_f << "\n"
    #            << V_f_aligned << "\n"
    #            << R_V_f * Quaternions::zHat * R_V_f.conjugate() << "\n" << std::endl;

    # Now rotate Instant so that its z axis is aligned with V_f
    Instant.rotate_decomposition_basis(R_V_f)

    # Get the phase of the (2,+/-2) modes after rotation
    i_22 = Instant.index(2, 2)
    i_2m2 = Instant.index(2, -2)
    phase_22 = math.atan2(Instant.data[0, i_22].imag, Instant.data[0, i_22].real)
    phase_2m2 = math.atan2(Instant.data[0, i_2m2].imag, Instant.data[0, i_2m2].real)

    # R_eps is the rotation we will be applying on the right-hand side
    R_eps = R_V_f * (quaternion.quaternion(0, 0, 0, (-(phase_22 - phase_2m2) / 8.0))).exp()

    # Without changing anything else (the direction of V_f or the
    # phase), make sure that the rotating frame's XHat axis is more
    # parallel to the input nHat_t_fid than anti-parallel.
    if np.dot(nHat_t_fid.vec, (R_f0 * R_eps * quaternion.x * R_eps.inverse() * R_f0.inverse()).vec) < 0:
        R_eps = R_eps * ((math.pi / 2.0) * quaternion.z).exp()

    return R_eps


@waveform_alterations
def align_decomposition_frame_to_modes(w, t_fid, nHat_t_fid=quaternion.x, ell_max=None):
    """Fix the attitude of the corotating frame.

    The corotating frame is only defined up to some constant rotor
    R_eps; if R_corot is corotating, then so is R_corot*R_eps.  This
    function uses that freedom to ensure that the frame is aligned
    with the Waveform modes at the fiducial time.  In particular, it
    ensures that the Z axis of the frame in which the decomposition
    is done is along the dominant eigenvector of the <LL> matrix
    (suggested by O'Shaughnessy et al.), and the phase of the (2,2)
    mode is zero.

    If ell_max is None (default), all ell modes are used.

    Parameters
    ----------
    w: WaveformModes
        Object to be aligned
    t_fid: float
        Fiducial time at which the alignment should happen
    nHat_t_fid: quaternion (optional)
        The approximate direction of nHat at t_fid; defaults to x
    ell_max: int
       Highest ell mode to use in computing the <LL> matrix

    """

    # Find the appropriate rotation
    R_eps = get_alignment_of_decomposition_frame_to_modes(w, t_fid, nHat_t_fid, ell_max)

    # Record what happened
    command = "{0}.align_decomposition_frame_to_modes({1}, {2}, {3})  # R_eps={4}"
    w._append_history(command.format(w, t_fid, nHat_t_fid, ell_max, R_eps))

    # Now, apply the rotation
    w = w.rotate_decomposition_basis(R_eps)

    return w


@waveform_alterations
def rotate_physical_system(W, R_phys):
    """Rotate a Waveform in place

    This just rotates the decomposition basis by the inverse of the input
    rotor(s).  See `rotate_decomposition_basis`.

    For more information on the analytical details, see
    http://moble.github.io/spherical_functions/SWSHs.html#rotating-swshs

    """
    W = rotate_decomposition_basis(W, ~R_phys)
    W._append_history(f"{W}.rotate_physical_system({R_phys})")
    return W  # Probably no return, but just in case...


@waveform_alterations
def rotate_decomposition_basis(W, R_basis):
    """Rotate a Waveform in place

    This function takes a Waveform object `W` and either a quaternion
    or array of quaternions `R_basis`.  It applies that rotation to
    the decomposition basis of the modes in the Waveform.  The change
    in basis is also recorded in the Waveform's `frame` data.

    For more information on the analytical details, see
    http://moble.github.io/spherical_functions/SWSHs.html#rotating-swshs

    """
    # This will be used in the jitted functions below to store the
    # Wigner D matrix at each time step
    D = np.empty((sf.WignerD._total_size_D_matrices(W.ell_min, W.ell_max),), dtype=complex)

    if isinstance(R_basis, (list, np.ndarray)) and len(R_basis) == 1:
        R_basis = R_basis[0]

    if isinstance(R_basis, (list, np.ndarray)):
        if isinstance(R_basis, np.ndarray) and R_basis.ndim != 1:
            raise ValueError("Input dimension mismatch.  R_basis.shape={1}".format(R_basis.shape))
        if W.n_times != len(R_basis):
            raise ValueError(
                "Input dimension mismatch.  (W.n_times={}) != (len(R_basis)={})".format(W.n_times, len(R_basis))
            )
        _rotate_decomposition_basis_by_series(W.data, quaternion.as_spinor_array(R_basis), W.ell_min, W.ell_max, D)

        # Update the frame data, using right-multiplication
        if W.frame.size:
            if W.frame.shape[0] == 1:
                # Numpy can't currently multiply one element times an array
                W.frame = np.array([W.frame * R for R in R_basis])
            else:
                W.frame = W.frame * R_basis
        else:
            W.frame = np.copy(R_basis)

    # We can't just use an `else` here because we need to process the
    # case where the input was an iterable of length 1, which we've
    # now changed to just a single quaternion.
    if isinstance(R_basis, np.quaternion):
        sf._Wigner_D_matrices(R_basis.a, R_basis.b, W.ell_min, W.ell_max, D)
        tmp = np.empty((2 * W.ell_max + 1,), dtype=complex)
        _rotate_decomposition_basis_by_constant(W.data, W.ell_min, W.ell_max, D, tmp)

        # Update the frame data, using right-multiplication
        if W.frame.size:
            W.frame = W.frame * R_basis
        else:
            W.frame = np.array([R_basis])

    opts = np.get_printoptions()
    np.set_printoptions(threshold=6)
    W.__history_depth__ -= 1
    W._append_history(f"{W}.rotate_decomposition_basis({R_basis})")
    np.set_printoptions(**opts)

    return W


@jit("void(c16[:,:], i8, i8, c16[:], c16[:])")
def _rotate_decomposition_basis_by_constant(data, ell_min, ell_max, D, tmp):
    """Rotate data by the same rotor at each point in time

    `D` is the Wigner D matrix for all the ell values.

    `tmp` is just a workspace used as temporary storage to hold the
    results for each item of data during the sum.

    """
    for i_t in range(data.shape[0]):
        for ell in range(ell_min, ell_max + 1):
            i_data = ell ** 2 - ell_min ** 2
            i_D = sf._linear_matrix_offset(ell, ell_min)

            for i_m in range(2 * ell + 1):
                tmp[i_m] = 0j
            for i_mp in range(2 * ell + 1):
                for i_m in range(2 * ell + 1):
                    tmp[i_m] += data[i_t, i_data + i_mp] * D[i_D + (2 * ell + 1) * i_mp + i_m]
            for i_m in range(2 * ell + 1):
                data[i_t, i_data + i_m] = tmp[i_m]


@jit("void(c16[:,:], c16[:,:], i8, i8, c16[:])")
def _rotate_decomposition_basis_by_series(data, R_basis, ell_min, ell_max, D):
    """Rotate data by a different rotor at each point in time

    `D` is just a workspace, which holds the Wigner D matrices.
    During the summation, it is also used as temporary storage to hold
    the results for each item of data, where the first row in the
    matrix is overwritten with the new sums.

    """
    for i_t in range(data.shape[0]):
        sf._Wigner_D_matrices(R_basis[i_t, 0], R_basis[i_t, 1], ell_min, ell_max, D)
        for ell in range(ell_min, ell_max + 1):
            i_data = ell ** 2 - ell_min ** 2
            i_D = sf._linear_matrix_offset(ell, ell_min)

            for i_m in range(2 * ell + 1):
                new_data_mp = 0j
                for i_mp in range(2 * ell + 1):
                    new_data_mp += data[i_t, i_data + i_mp] * D[i_D + i_m + (2 * ell + 1) * i_mp]
                D[i_D + i_m] = new_data_mp
            for i_m in range(2 * ell + 1):
                data[i_t, i_data + i_m] = D[i_D + i_m]
