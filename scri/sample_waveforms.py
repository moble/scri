# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

import math
import numpy as np
import quaternion
import scri
import spherical_functions as sf
import warnings


def modes_constructor(constructor_statement, data_functor, **kwargs):
    """WaveformModes object filled with data from the input functor

    Additional keyword arguments are mostly passed to the WaveformModes initializer, though some more reasonable
    defaults are provided.

    Parameters
    ----------
    constructor_statement : str
        This is a string form of the function call used to create the object.  This is passed to the WaveformBase
        initializer as the parameter of the same name.  See the docstring for more information.
    data_functor : function
        Takes a 1-d array of time values and an array of (ell, m) values and returns the complex array of data.
    t : float array, optional
        Time values of the data.  Default is `np.linspace(-10., 100., num=1101))`.
    ell_min, ell_max : int, optional
        Smallest and largest ell value present in the data.  Defaults are 2 and 8.

    """
    t = np.array(kwargs.pop("t", np.linspace(-10.0, 100.0, num=1101)), dtype=float)
    frame = np.array(kwargs.pop("frame", []), dtype=np.quaternion)
    frameType = int(kwargs.pop("frameType", scri.Inertial))
    dataType = int(kwargs.pop("dataType", scri.h))
    r_is_scaled_out = bool(kwargs.pop("r_is_scaled_out", True))
    m_is_scaled_out = bool(kwargs.pop("m_is_scaled_out", True))
    ell_min = int(kwargs.pop("ell_min", abs(scri.SpinWeights[dataType])))
    ell_max = int(kwargs.pop("ell_max", 8))
    if kwargs:
        import pprint

        warnings.warn(f"\nUnused kwargs passed to this function:\n{pprint.pformat(kwargs, width=1)}")
    data = data_functor(t, sf.LM_range(ell_min, ell_max))
    w = scri.WaveformModes(
        t=t,
        frame=frame,
        data=data,
        history=["# Called from constant_waveform"],
        frameType=frameType,
        dataType=dataType,
        r_is_scaled_out=r_is_scaled_out,
        m_is_scaled_out=m_is_scaled_out,
        constructor_statement=constructor_statement,
        ell_min=ell_min,
        ell_max=ell_max,
    )
    return w


def constant_waveform(**kwargs):
    """WaveformModes object with constant values in each mode

    Additional keyword arguments are passed to `modes_constructor`.

    """
    if kwargs:
        import pprint

        warnings.warn(f"\nUnused kwargs passed to this function:\n{pprint.pformat(kwargs, width=1)}")

    def data_functor(t, LM):
        data = np.zeros((t.shape[0], LM.shape[0]), dtype=complex)
        for i, m in enumerate(LM[:, 1]):
            data[:, i] = m - 1j * m
        return data

    return modes_constructor(f"constant_waveform(**{kwargs})", data_functor)


def single_mode(ell, m, **kwargs):
    """WaveformModes object with 1 in selected slot and 0 elsewhere

    Additional keyword arguments are passed to `modes_constructor`.

    Parameters
    ----------
    ell, m : int
        The (ell, m) value of the nonzero mode

    """
    if kwargs:
        import pprint

        warnings.warn(f"\nUnused kwargs passed to this function:\n{pprint.pformat(kwargs, width=1)}")

    def data_functor(t, LM):
        data = np.zeros((t.shape[0], LM.shape[0]), dtype=complex)
        data[:, sf.LM_index(ell, m, min(LM[:, 0]))] = 1.0 + 0.0j
        return data

    return modes_constructor(f"single_mode({ell}, {m}, **{kwargs})", data_functor)


def random_waveform(**kwargs):
    """WaveformModes object filled with random data at each time step

    Additional keyword arguments are passed to `modes_constructor`.

    Parameters
    ----------
    uniform_time : bool, optional
        Use uniform, rather than random, time steps.  Default is False.
    begin, end : float, optional
        Initial and final times of the time data.  Default values are -10., 100.
    n_times : int, optional
        Number of time steps in the time data.  Default is 1101
    rotating : bool, optional
        If True, use a `Corotating` frame, with random orientation at each instant.  Default is True.

    """
    np.random.seed(hash("random_waveform") % 4294967294)
    begin = float(kwargs.pop("begin", -10.0))
    end = float(kwargs.pop("end", 100.0))
    n_times = int(kwargs.pop("n_times", 1101))
    if kwargs.pop("uniform_time", False):
        t = np.array(kwargs.pop("t", np.linspace(begin, end, num=n_times)), dtype=float)
    else:
        t = np.sort(np.random.uniform(begin, end, size=n_times))
    rotating = bool(kwargs.pop("rotating", True))

    if kwargs:
        import pprint

        warnings.warn(f"\nUnused kwargs passed to this function:\n{pprint.pformat(kwargs, width=1)}")

    def data_functor(tin, LM):
        data = np.random.normal(size=(tin.shape[0], LM.shape[0], 2)).view(complex)[:, :, 0]
        return data

    if rotating:
        frame = np.array([np.quaternion(*np.random.uniform(-1, 1, 4)).normalized() for t_i in t])
        return modes_constructor(
            f"random_waveform(**{kwargs})", data_functor, t=t, frame=frame, frameType=scri.Corotating
        )
    else:
        return modes_constructor(f"random_waveform(**{kwargs})", data_functor, t=t)


def random_waveform_proportional_to_time(**kwargs):
    """WaveformModes object filled with random data times the time coordinate

    Additional keyword arguments are passed to `modes_constructor`.

    Parameters
    ----------
    uniform_time : bool, optional
        Use uniform, rather than random, time steps.  Default is False.
    begin, end : float, optional
        Initial and final times of the time data.  Default values are -10., 100.
    n_times : int, optional
        Number of time steps in the time data.  Default is 1101
    rotating : bool, optional
        If True, use a `Corotating` frame, with random orientation at each instant.  Default is True.

    """
    np.random.seed(hash("random_waveform_proportional_to_time") % 4294967294)  # Use mod to get in an acceptable range
    begin = float(kwargs.pop("begin", -10.0))
    end = float(kwargs.pop("end", 100.0))
    n_times = int(kwargs.pop("n_times", 1101))
    if kwargs.pop("uniform_time", False):
        t = np.array(kwargs.pop("t", np.linspace(begin, end, num=n_times)), dtype=float)
    else:
        t = np.sort(np.random.uniform(begin, end, size=n_times))
    rotating = bool(kwargs.pop("rotating", True))

    if kwargs:
        import pprint

        warnings.warn(f"\nUnused kwargs passed to this function:\n{pprint.pformat(kwargs, width=1)}")

    def data_functor(tin, LM):
        return np.outer(tin, np.random.normal(size=(LM.shape[0], 2)).view(complex)[:, 0])

    if rotating:
        axis = np.quaternion(0.0, *np.random.uniform(-1, 1, size=3)).normalized()
        omega = 2 * np.pi * 4 / (t[-1] - t[0])
        frame = np.array([np.exp(axis * (omega * t_i / 2)) for t_i in t])
        return modes_constructor(
            f"random_waveform(**{kwargs})", data_functor, t=t, frame=frame, frameType=scri.Corotating
        )
    else:
        return modes_constructor(f"random_waveform(**{kwargs})", data_functor, t=t)


def single_mode_constant_rotation(**kwargs):
    """Return WaveformModes object a single nonzero mode, with phase proportional to time

    The waveform output by this function will have just one nonzero mode.  The behavior of that mode will be fairly
    simple; it will be given by exp(i*omega*t).  Note that omega can be complex, which gives damping.

    Parameters
    ----------
    s : int, optional
        Spin weight of the waveform field.  Default is -2.
    ell, m : int, optional
        The (ell, m) values of the nonzero mode in the returned waveform.  Default value is (abs(s), -abs(s)).
    ell_min, ell_max : int, optional
        Smallest and largest ell values present in the output.  Default values are abs(s) and 8.
    data_type : int, optional
        Default value is whichever psi_n corresponds to the input spin.  It is important to choose these, rather than
        `h` or `sigma` for the analytical solution to translations, which doesn't account for the direct contribution
        of supertranslations (as opposed to the indirect contribution, which involves moving points around).
    t_0, t_1 : float, optional
        Beginning and end of time.  Default values are -20. and 20.
    dt : float, optional
        Time step.  Default value is 0.1.
    omega : complex, optional
        Constant of proportionality such that nonzero mode is exp(i*omega*t).  Note that this can be complex, which
        implies damping.  Default is 0.5.

    """
    s = kwargs.pop("s", -2)
    ell = kwargs.pop("ell", abs(s))
    m = kwargs.pop("m", -ell)
    ell_min = kwargs.pop("ell_min", abs(s))
    ell_max = kwargs.pop("ell_max", 8)
    data_type = kwargs.pop("data_type", scri.DataType[scri.SpinWeights.index(s)])
    t_0 = kwargs.pop("t_0", -20.0)
    t_1 = kwargs.pop("t_1", 20.0)
    dt = kwargs.pop("dt", 1.0 / 10.0)
    t = np.arange(t_0, t_1 + dt, dt)
    n_times = t.size
    omega = complex(kwargs.pop("omega", 0.5))
    data = np.zeros((n_times, sf.LM_total_size(ell_min, ell_max)), dtype=complex)
    data[:, sf.LM_index(ell, m, ell_min)] = np.exp(1j * omega * t)

    if kwargs:
        import pprint

        warnings.warn(f"\nUnused kwargs passed to this function:\n{pprint.pformat(kwargs, width=1)}")

    return scri.WaveformModes(
        t=t,
        data=data,
        ell_min=ell_min,
        ell_max=ell_max,
        frameType=scri.Inertial,
        dataType=data_type,
        r_is_scaled_out=True,
        m_is_scaled_out=True,
    )


def single_mode_proportional_to_time(**kwargs):
    """Return WaveformModes object a single nonzero mode, proportional to time

    The waveform output by this function will have just one nonzero mode.  The behavior of that mode will be
    particularly simple; it will just be proportional to time.

    Parameters
    ----------
    s : int, optional
        Spin weight of the waveform field.  Default is -2.
    ell, m : int, optional
        The (ell, m) values of the nonzero mode in the returned waveform.  Default value is (abs(s), -abs(s)).
    ell_min, ell_max : int, optional
        Smallest and largest ell values present in the output.  Default values are abs(s) and 8.
    data_type : int, optional
        Default value is whichever psi_n corresponds to the input spin.  It is important to choose these, rather than
        `h` or `sigma` for the analytical solution to translations, which doesn't account for the direct contribution
        of supertranslations (as opposed to the indirect contribution, which involves moving points around).
    t_0, t_1 : float, optional
        Beginning and end of time.  Default values are -20. and 20.
    dt : float, optional
        Time step.  Default value is 0.1.
    beta : complex, optional
        Constant of proportionality such that nonzero mode is beta*t.  Default is 1.

    """
    s = kwargs.pop("s", -2)
    ell = kwargs.pop("ell", abs(s))
    m = kwargs.pop("m", -ell)
    ell_min = kwargs.pop("ell_min", abs(s))
    ell_max = kwargs.pop("ell_max", 8)
    data_type = kwargs.pop("data_type", scri.DataType[scri.SpinWeights.index(s)])
    t_0 = kwargs.pop("t_0", -20.0)
    t_1 = kwargs.pop("t_1", 20.0)
    dt = kwargs.pop("dt", 1.0 / 10.0)
    t = np.arange(t_0, t_1 + dt, dt)
    n_times = t.size
    beta = kwargs.pop("beta", 1.0)
    data = np.zeros((n_times, sf.LM_total_size(ell_min, ell_max)), dtype=complex)
    data[:, sf.LM_index(ell, m, ell_min)] = beta * t

    if kwargs:
        import pprint

        warnings.warn(f"\nUnused kwargs passed to this function:\n{pprint.pformat(kwargs, width=1)}")

    return scri.WaveformModes(
        t=t,
        data=data,
        ell_min=ell_min,
        ell_max=ell_max,
        frameType=scri.Inertial,
        dataType=data_type,
        r_is_scaled_out=True,
        m_is_scaled_out=True,
    )


def single_mode_proportional_to_time_supertranslated(**kwargs):
    """Return WaveformModes as in single_mode_proportional_to_time, with analytical supertranslation

    This function constructs the same basic object as the `single_mode_proportional_to_time`, but then applies an
    analytical supertranslation.  The arguments to this function are the same as to the other, with two additions:

    Additional parameters
    ---------------------
    supertranslation : complex array, optional
        Spherical-harmonic modes of the supertranslation to apply to the waveform.  This is overwritten by
         `space_translation` if present.  Default value is `None`.
    space_translation : float array of length 3, optional
        This is just the 3-vector representing the displacement to apply to the waveform.  Note that if
        `supertranslation`, this parameter overwrites it.  Default value is [1.0, 0.0, 0.0].

    """
    s = kwargs.pop("s", -2)
    ell = kwargs.pop("ell", abs(s))
    m = kwargs.pop("m", -ell)
    ell_min = kwargs.pop("ell_min", abs(s))
    ell_max = kwargs.pop("ell_max", 8)
    data_type = kwargs.pop("data_type", scri.DataType[scri.SpinWeights.index(s)])
    t_0 = kwargs.pop("t_0", -20.0)
    t_1 = kwargs.pop("t_1", 20.0)
    dt = kwargs.pop("dt", 1.0 / 10.0)
    t = np.arange(t_0, t_1 + dt, dt)
    n_times = t.size
    beta = kwargs.pop("beta", 1.0)
    data = np.zeros((n_times, sf.LM_total_size(ell_min, ell_max)), dtype=complex)
    data[:, sf.LM_index(ell, m, ell_min)] = beta * t
    supertranslation = np.array(kwargs.pop("supertranslation", np.array([], dtype=complex)), dtype=complex)
    if "space_translation" in kwargs:
        if supertranslation.size < 4:
            supertranslation.resize((4,))
        supertranslation[1:4] = -sf.vector_as_ell_1_modes(kwargs.pop("space_translation"))
    supertranslation_ell_max = int(math.sqrt(supertranslation.size) - 1)
    if supertranslation_ell_max * (supertranslation_ell_max + 2) + 1 != supertranslation.size:
        raise ValueError(f"Bad number of elements in supertranslation: {supertranslation.size}")
    for i, (ellpp, mpp) in enumerate(sf.LM_range(0, supertranslation_ell_max)):
        if supertranslation[i] != 0.0:
            mp = m + mpp
            for ellp in range(ell_min, min(ell_max, (ell + ellpp)) + 1):
                if ellp >= abs(mp):
                    addition = (
                        beta
                        * supertranslation[i]
                        * math.sqrt(((2 * ellpp + 1) * (2 * ell + 1) * (2 * ellp + 1)) / (4 * math.pi))
                        * sf.Wigner3j(ellpp, ell, ellp, 0, -s, s)
                        * sf.Wigner3j(ellpp, ell, ellp, mpp, m, -mp)
                    )
                    if (s + mp) % 2 == 1:
                        addition *= -1
                    data[:, sf.LM_index(ellp, mp, ell_min)] += addition

    if kwargs:
        import pprint

        warnings.warn(f"\nUnused kwargs passed to this function:\n{pprint.pformat(kwargs, width=1)}")

    return scri.WaveformModes(
        t=t,
        data=data,
        ell_min=ell_min,
        ell_max=ell_max,
        frameType=scri.Inertial,
        dataType=data_type,
        r_is_scaled_out=True,
        m_is_scaled_out=True,
    )


def fake_precessing_waveform(
    t_0=-20.0,
    t_1=20_000.0,
    dt=0.1,
    ell_max=8,
    mass_ratio=2.0,
    precession_opening_angle=np.pi / 6.0,
    precession_opening_angle_dot=None,
    precession_relative_rate=0.1,
    precession_nutation_angle=None,
    inertial=True,
):
    """Construct a strain waveform with realistic precession effects.

    This model is intended to be weird enough that it breaks any overly simplistic assumptions about
    waveform symmetries while still being (mostly) realistic.

    This waveform uses only the very lowest-order terms from PN theory to evolve the orbital
    frequency up to a typical constant value (with a smooth transition), and to construct modes that
    have very roughly the correct amplitudes as a function of the orbital frequency.  Modes with
    equal ell values but opposite m values are modulated antisymmetrically, though this modulation
    decays quickly after merger -- roughly as it would behave in a precessing system.  The modes are
    then smoothly transitioned to an exponential decay after merger.  The frame is a simulated
    precession involving the basic orbital rotation precessing about a cone of increasing opening
    angle and nutating about that cone on the orbital time scale, but settling down to a constant
    direction shortly after merger.  (Though there is little precise physical content, these
    features are all found in real waveforms.)  If the input argument `inertial` is `True` (the
    default), the waveform is transformed back to the inertial frame before returning.

    Parameters
    ==========
    t_0: float [defaults to -20.0]
    t_1: float [defaults to 20_000.0]
        The initial and final times in the output waveform.  Note that the merger is placed 100.0
        time units before `t_1`, and several transitions are made before this, so `t_0` must be that
        far ahead of `t_1`.
    dt: float [defaults to 0.1]
        Spacing of output time series.
    ell_max: int [defaults to 8]
        Largest ell value in the output modes.
    mass_ratio: float [defaults to 2.0]
        Ratio of BH masses to use as input to rough approximations for orbital evolution and mode
        amplitudes.
    precession_opening_angle: float [defaults to pi/6]
        Opening angle of the precession cone.
    precession_opening_angle_dot: float [defaults to 2*precession_opening_angle/(t_merger-t_0)]
        Rate at which precession cone opens.
    precession_relative_rate: float [defaults to 0.1]
        Fraction of the magnitude of the orbital angular velocity at which it precesses.
    precession_nutation_angle: float [defaults to precession_opening_angle/10]
        Angle (relative to precession_opening_angle) by which the orbital angular velocity nutates.

    """
    import warnings
    import numpy as np
    import quaternion
    from quaternion.calculus import indefinite_integral
    from .utilities import transition_function, transition_to_constant

    if mass_ratio < 1.0:
        mass_ratio = 1.0 / mass_ratio

    s = -2
    ell_min = abs(s)
    data_type = scri.h

    nu = mass_ratio / (1 + mass_ratio) ** 2
    t = np.arange(t_0, t_1 + 0.99 * dt, dt)
    t_merger = t_1 - 100.0
    i_merger = np.argmin(abs(t - t_merger))
    if i_merger < 20:
        raise ValueError(f"Insufficient space between initial time (t={t_merger}) and merger (t={t_0}).")
    n_times = t.size
    data = np.zeros((n_times, sf.LM_total_size(ell_min, ell_max)), dtype=complex)

    # Get a rough approximation to the phasing through merger
    tau = nu * (t_merger - t) / 5
    with warnings.catch_warnings():  # phi and omega will have NaNs after t_merger for now
        warnings.simplefilter("ignore")
        phi = -4 * tau ** (5 / 8)
        omega = (nu / 2) * tau ** (-3 / 8)

    # Now, transition omega smoothly up to a constant value of 0.25
    omega_transition_width = 5.0
    i1 = np.argmin(np.abs(omega[~np.isnan(omega)] - 0.25))
    i0 = np.argmin(np.abs(t - (t[i1] - omega_transition_width)))
    transition = transition_function(t, t[i0], t[i1])
    zero_point_two_five = 0.25 * np.ones_like(t)
    omega[:i1] = omega[:i1] * (1 - transition[:i1]) + zero_point_two_five[:i1] * transition[:i1]
    omega[i1:] = 0.25

    # Integrate phi after i0 to agree with the new omega
    phi[i0:] = phi[i0] + indefinite_integral(omega[i0:], t[i0:])

    # Construct ringdown-transition function
    ringdown_transition_width = 20
    t0 = t_merger
    i0 = np.argmin(np.abs(t - t_merger))
    i1 = np.argmin(np.abs(t - (t[i0] + ringdown_transition_width)))
    t0 = t[i0]
    t1 = t[i1]
    transition = transition_function(t, t0, t1)
    ringdown = np.ones_like(t)
    ringdown[i0:] = ringdown[i0:] * (1 - transition[i0:]) + 2.25 * np.exp(-(t[i0:] - t_merger) / 11.5) * transition[i0:]

    # Construct frame
    if precession_opening_angle_dot is None:
        precession_opening_angle_dot = 2.0 * precession_opening_angle / (t[i1] - t[0])
    if precession_nutation_angle is None:
        precession_nutation_angle = precession_opening_angle / 10.0
    R_orbital = np.exp(phi * quaternion.z / 2)
    R_opening = np.exp(
        transition_to_constant(precession_opening_angle + precession_opening_angle_dot * t, t, t0, t1)
        * quaternion.x
        / 2
    )
    R_precession = np.exp(transition_to_constant(phi / precession_relative_rate, t, t0, t1) * quaternion.z / 2)
    R_nutation = np.exp(precession_nutation_angle * transition * quaternion.x / 2)
    frame = (
        R_orbital * R_nutation * R_orbital.conjugate() * R_precession * R_opening * R_precession.conjugate() * R_orbital
    )
    frame = frame[0].sqrt().conjugate() * frame  # Just give the initial angle a weird little tweak to screw things up

    # Construct the modes
    x = omega ** (2 / 3)
    modulation = transition_function(t, t[i0], t[i1], 1, 0) * np.cos(phi) / 40.0
    for ell in range(ell_min, ell_max + 1):
        for m in range(-ell, ell + 1):
            data[:, sf.LM_index(ell, m, ell_min)] = pn_leading_order_amplitude(ell, m, x, mass_ratio=mass_ratio) * (
                1 + np.sign(m) * modulation
            )

    # Apply ringdown (mode amplitudes are constant after t_merger)
    data *= ringdown[:, np.newaxis]

    h_corot = scri.WaveformModes(
        t=t,
        frame=frame,
        data=data,
        ell_min=ell_min,
        ell_max=ell_max,
        frameType=scri.Corotating,
        dataType=data_type,
        r_is_scaled_out=True,
        m_is_scaled_out=True,
    )

    if inertial:
        return h_corot.to_inertial_frame()
    else:
        return h_corot


def pn_leading_order_amplitude(ell, m, x, mass_ratio=1.0):
    """Return the leading-order amplitude of r*h/M in PN theory

    These expressions are from Eqs. (330) of Blanchet's Living Review (2014).

    Note that `x` is just the orbital angular velocity to the (2/3) power.

    """
    from scipy.special import factorial, factorial2

    if m < 0:
        return (-1) ** ell * np.conjugate(pn_leading_order_amplitude(ell, -m, x, mass_ratio=mass_ratio))

    if mass_ratio < 1.0:
        mass_ratio = 1.0 / mass_ratio
    nu = mass_ratio / (1 + mass_ratio) ** 2
    X1 = mass_ratio / (mass_ratio + 1)
    X2 = 1 / (mass_ratio + 1)

    def sigma(ell):
        return X2 ** (ell - 1) + (-1) ** ell * X1 ** (ell - 1)

    if (ell + m) % 2 == 0:
        amplitude = (
            (
                (-1) ** ((ell - m + 2) / 2)
                / (2 ** (ell + 1) * factorial((ell + m) // 2) * factorial((ell - m) // 2) * factorial2(2 * ell - 1))
            )
            * np.sqrt(
                (5 * (ell + 1) * (ell + 2) * factorial(ell + m) * factorial(ell - m))
                / (ell * (ell - 1) * (2 * ell + 1))
            )
            * sigma(ell)
            * (1j * m) ** ell
            * x ** (ell / 2 - 1)
        )
    else:
        amplitude = (
            (
                (-1) ** ((ell - m - 1) / 2)
                / (
                    2 ** (ell - 1)
                    * factorial((ell + m - 1) // 2)
                    * factorial((ell - m - 1) // 2)
                    * factorial2(2 * ell + 1)
                )
            )
            * np.sqrt(
                (5 * (ell + 2) * (2 * ell + 1) * factorial(ell + m) * factorial(ell - m))
                / (ell * (ell - 1) * (ell + 1))
            )
            * sigma(ell + 1)
            * 1j
            * (1j * m) ** ell
            * x ** ((ell - 1) / 2)
        )

    return 8 * np.sqrt(np.pi / 5) * nu * x * amplitude


def create_fake_finite_radius_strain_h5file(
    output_file_path="./rh_FiniteRadii_CodeUnits.h5",
    n_subleading=3,
    amp=1.0,
    t_0=0.0,
    t_1=3000.0,
    dt=0.1,
    r_min=100.0,
    r_max=600.0,
    n_radii=24,
    ell_max=8,
    initial_adm_energy=0.99,
    avg_lapse=0.99,
    avg_areal_radius_diff=1.0,
    mass_ratio=1.0,
    precession_opening_angle=0.0,
    **kwargs,
):
    """
    Create an HDF5 file with fake finite-radius GW strain data in the NRAR format, as
    used by the Spectral Einstein Code (SpEC). The asymptotic waveform is created by
    the function scri.sample_waveforms.fake_precessing_waveform and then radius-dependent
    terms are added to it. These subleading artificial "near-zone" effects are simple
    sinusoids. The finite-radius waveform, scaled by a factor of the radius, is then,

        r*h(r) = h_0 + h_1*r**-1 + h_2*r**-2 + ... + h_n*r**-n

    where h_0 is the waveform from scri.sample_waveforms.fake_precessing_waveform() and n is
    chosen by the user.

    Finally, the finite-radius waveforms as output by SpEC uses simulation coordinates for
    radius and time, which do not perfectly parametrize outgoing null rays. This function
    approximates these simulation coordinates so that the data in the resulting HDF5 file
    most nearly mimics that of a SpEC HDF5 output file.

    Parameters
    ==========
    output_file_path: str
        The name and path of the output file. The filename must be "rh_FiniteRadius_CodeUnits.h5"
        if you wish to be able to run scri.extrapolation.extrapolate on it.
    n_subleading: int [defaults to 3]
        The number of subleading, radius-dependent terms to add to the asymptotic waveform.
    amp: float [defaults to 1.0]
        The amplitude of the subleading, radius-dependent terms.
    t_0: float [defaults to 0.0]
    t_1: float [defaults to 3000.0]
        The initial and final times in the output waveform. Note that the merger is placed
        100.0 time units before `t_1`, and several transitions are made before this, so `t_0`
        must be that far ahead of `t_1`.
    dt: float [defaults to 0.1]
        Spacing of output time series.
    r_min: float [defaults to 100.0]
    r_max: float [defaults to 600.0]
    n_radii: float [defaults to 24]
        The minimum and maximum radius, and the number of radii between these values, at which
        to produce a finite-radius waveform. These will be equally spaced in inverse radius.
    ell_max: int [defaults to 8]
        Largest ell value in the output modes.
    initial_adm_energy: float [defaults to 0.99]
        The intial ADM energy as would be computed by the SpEC intitial data solver.
    avg_lapse: float [defaults to 0.99]
        The value of the lapse averaged over a sphere at
    avg_areal_radius_diff: float [defaults to 1.0]
        How much on average the areal radius is larger than the coord radius, may be negative.
    mass_ratio: float [defaults to 1.0]
        Ratio of BH masses to use as input to rough approximations for orbital evolution
        and mode amplitudes.
    precession_opening_angle: float [defaults to 0.0]
        Opening angle of the precession cone. The default results in no precession. If
        this value is non-zero, then the following options from fake_precessing_waveform
        may be supplied as kwagrs:
            * precession_opening_angle_dot
            * precession_relative_rate
            * precession_nutation_angle
        See the help text of fake_precessing_waveform for documentation.

    """
    import h5py
    from scipy.interpolate import CubicSpline

    with h5py.File(output_file_path, "x") as h5file:

        # Set up an H5 group for each radius
        coord_radii = (1 / np.linspace(1 / r_min, 1 / r_max, n_radii)).astype(int)
        groups = [f"R{R:04}.dir" for R in coord_radii]
        for group in groups:
            h5file.create_group(group)

        # Create version history dataset
        # We mimic how SpEC encodes the strings in VersionHist.ver to ensure we
        # don't hit any errors due to the encoding.
        version_hist = [["ef51849550a1d8a5bbdd810c7debf0dd839e86dd", "Overall sign change in complex strain h."]]
        reencoded_version_hist = [
            [entry[0].encode("ascii", "ignore"), entry[1].encode("ascii", "ignore")] for entry in version_hist
        ]
        dset = h5file.create_dataset(
            "VersionHist.ver",
            (len(version_hist), 2),
            maxshape=(None, 2),
            data=reencoded_version_hist,
            dtype=h5py.special_dtype(vlen=bytes),
        )
        dset.attrs.create("Version", len(version_hist), shape=(1,), dtype=np.uint64)

        # Generate the asymptotic waveform
        h0 = scri.sample_waveforms.fake_precessing_waveform(
            t_0=t_0, t_1=t_1, dt=dt, ell_max=ell_max, precession_opening_angle=precession_opening_angle, **kwargs
        )
        n_times = int(h0.t.shape[0] + r_max / dt)
        t_1 += r_max
        all_times = np.linspace(0, t_1, n_times)

        # Set auxiliary datasetsss
        for i in range(len(groups)):
            # Set coordinate radius dataset
            coord_radius = np.vstack((all_times, [coord_radii[i]] * n_times)).T
            dset = h5file.create_dataset(f"{groups[i]}/CoordRadius.dat", data=coord_radius)
            dset.attrs.create("Legend", ["time", "CoordRadius"])

            # Set areal radius dataset
            areal_radius = coord_radius
            areal_radius[:, 1] += avg_areal_radius_diff
            dset = h5file.create_dataset(f"{groups[i]}/ArealRadius.dat", data=areal_radius)
            dset.attrs.create("Legend", ["time", "ArealRadius"])

            # Set initial ADM energy dataset
            seg_start_times = h0.t[:: int(n_times / 20)]
            adm_energy_dset = np.vstack((seg_start_times, [initial_adm_energy] * len(seg_start_times))).T
            dset = h5file.create_dataset(f"{groups[i]}/InitialAdmEnergy.dat", data=adm_energy_dset)
            dset.attrs.create("Legend", ["time", "InitialAdmEnergy"])

            # Set average lapse dataset
            avg_lapse_dset = np.vstack((all_times, [avg_lapse] * n_times)).T
            dset = h5file.create_dataset(f"{groups[i]}/AverageLapse.dat", data=avg_lapse_dset)
            dset.attrs.create("Legend", ["time", "AverageLapse"])

            # Set finite radius data
            R = areal_radius[0, 1]
            tortoise_coord = R + 2 * initial_adm_energy * np.log(R / (2 * initial_adm_energy) - 1)
            # Compute the approximate time in SpEC simulation coordinates
            simulation_time = (h0.t + tortoise_coord) * np.sqrt(1 - 2 * initial_adm_energy / R) / avg_lapse
            for l in range(2, ell_max + 1):
                for m in range(-l, l + 1):
                    index = sf.LM_index(l, m, 2)
                    new_data = h0.data[:, index]
                    for n in range(1, n_subleading + 1):
                        new_data += amp * R ** -n * np.exp((1j * n * 50 * np.pi / h0.n_times) * h0.t) * h0.abs[:, index]
                    new_data = CubicSpline(simulation_time, new_data)(all_times)
                    new_data[(all_times < simulation_time[0]) | (all_times > simulation_time[-1])] = 0.0
                    new_data += (1 + 1j) * 1e-14 * all_times
                    new_dset = np.vstack((all_times, new_data.real, new_data.imag)).T
                    dset = h5file.create_dataset(f"{groups[i]}/Y_l{l}_m{m}.dat", data=new_dset)
                    dset.attrs.create(
                        "Legend",
                        [
                            "time",
                            f"Re[rh]_l{l}_m{m}(R={coord_radii[i]}.00)",
                            f"Im[rh]_l{l}_m{m}(R={coord_radii[i]}.00)",
                        ],
                    )
