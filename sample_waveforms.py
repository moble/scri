# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import

import math
import numpy as np
import quaternion
import scri
import spherical_functions as sf


def modes_constructor(constructor_statement, data_functor, **kwargs):
    """WaveformModes object filled with data from the input functor

    Additional keyword arguments are mostly passed to the WaveformModes initializer, though some more reasonable
    defaults are provided.

    Parameters
    ----------
    constructor_statement : str
        This is a string form of the function call used to create the object.  This is passed to the WaveformBase
        initializer as the parameter of the same name.  See the docstring for more information.
    data_function : function
        Takes a 1-d array of time values and an array of (ell, m) values and returns
    t : float array, optional
        Time values of the data.  Default is `np.linspace(-10., 100., num=1101))`.
    ell_min, ell_max : int, optional
        Smallest and largest ell value present in the data.  Defaults are 2 and 8.

    """
    t = np.array(kwargs.pop('t', np.linspace(-10., 100., num=1101)), dtype=float)
    frame = np.array(kwargs.pop('frame', []), dtype=np.quaternion)
    frameType = int(kwargs.pop('frameType', scri.Inertial))
    dataType = int(kwargs.pop('dataType', scri.h))
    r_is_scaled_out = bool(kwargs.pop('r_is_scaled_out', True))
    m_is_scaled_out = bool(kwargs.pop('m_is_scaled_out', True))
    ell_min = int(kwargs.pop('ell_min', abs(scri.SpinWeights[dataType])))
    ell_max = int(kwargs.pop('ell_max', 8))
    if kwargs:
        import pprint
        warnings.warn("\nUnused kwargs passed to this function:\n{0}".format(pprint.pformat(kwargs, width=1)))
    data = data_functor(t, sf.LM_range(ell_min, ell_max))
    w = scri.WaveformModes(t=t, frame=frame, data=data,
                           history=['# Called from constant_waveform'],
                           frameType=frameType, dataType=dataType,
                           r_is_scaled_out=r_is_scaled_out, m_is_scaled_out=m_is_scaled_out,
                           constructor_statement=constructor_statement,
                           ell_min=ell_min, ell_max=ell_max)
    return w


def constant_waveform(**kwargs):
    """WaveformModes object with constant values in each mode

    Additional keyword arguments are passed to `modes_constructor`.

    """
    if kwargs:
        import pprint
        warnings.warn("\nUnused kwargs passed to this function:\n{0}".format(pprint.pformat(kwargs, width=1)))
    def data_functor(t, LM):
        data = np.empty((t.shape[0], LM.shape[0]), dtype=complex)
        for i, m in enumerate(LM[:, 1]):
            data[:, i] = (m - 1j * m)
        return data
    return modes_constructor('constant_waveform(**{0})'.format(kwargs), data_functor)


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
        warnings.warn("\nUnused kwargs passed to this function:\n{0}".format(pprint.pformat(kwargs, width=1)))
    def data_functor(t, LM):
        data = np.empty((t.shape[0], LM.shape[0]), dtype=complex)
        data[:, sf.LM_index(ell, m, min(LM[:, 0]))] = 1.0 + 0.0j
        return data
    return modes_constructor('single_mode({0}, {1}, **{2})'.format(ell, m, kwargs), data_functor)


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
    np.random.seed(hash('random_waveform') % 4294967294)
    begin = float(kwargs.pop('begin', -10.))
    end = float(kwargs.pop('end', 100.))
    n_times = int(kwargs.pop('n_times', 1101))
    if kwargs.pop('uniform_time', False):
        t = np.array(kwargs.pop('t', np.linspace(begin, end, num=n_times)), dtype=float)
    else:
        t = np.sort(np.random.uniform(begin, end, size=n_times))
    rotating = bool(kwargs.pop('rotating', True))

    if kwargs:
        import pprint
        warnings.warn("\nUnused kwargs passed to this function:\n{0}".format(pprint.pformat(kwargs, width=1)))

    def data_functor(tin, LM):
        data = np.random.normal(size=(tin.shape[0], LM.shape[0], 2)).view(complex)[:, :, 0]
        return data
    if rotating:
        frame = np.array([np.quaternion(*np.random.uniform(-1, 1, 4)).normalized() for t_i in t])
        return modes_constructor('random_waveform(**{0})'.format(kwargs), data_functor, t=t,
                                 frame=frame, frameType=scri.Corotating)
    else:
        return modes_constructor('random_waveform(**{0})'.format(kwargs), data_functor, t=t)


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
    np.random.seed(hash('random_waveform_proportional_to_time') % 4294967294)  # Use mod to get in an acceptable range
    begin = float(kwargs.pop('begin', -10.))
    end = float(kwargs.pop('end', 100.))
    n_times = int(kwargs.pop('n_times', 1101))
    if kwargs.pop('uniform_time', False):
        t = np.array(kwargs.pop('t', np.linspace(begin, end, num=n_times)), dtype=float)
    else:
        t = np.sort(np.random.uniform(begin, end, size=n_times))
    rotating = bool(kwargs.pop('rotating', True))

    if kwargs:
        import pprint
        warnings.warn("\nUnused kwargs passed to this function:\n{0}".format(pprint.pformat(kwargs, width=1)))

    def data_functor(tin, LM):
        return np.outer(tin, np.random.normal(size=(LM.shape[0], 2)).view(complex)[:, 0])
    if rotating:
        axis = np.quaternion(0., *np.random.uniform(-1, 1, size=3)).normalized()
        omega = 2 * np.pi * 4 / (t[-1] - t[0])
        frame = np.array([np.exp(axis * (omega * t_i / 2)) for t_i in t])
        return modes_constructor('random_waveform(**{0})'.format(kwargs), data_functor, t=t,
                                 frame=frame, frameType=scri.Corotating)
    else:
        return modes_constructor('random_waveform(**{0})'.format(kwargs), data_functor, t=t)


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
    s = kwargs.pop('s', -2)
    ell = kwargs.pop('ell', abs(s))
    m = kwargs.pop('m', -ell)
    ell_min = kwargs.pop('ell_min', abs(s))
    ell_max = kwargs.pop('ell_max', 8)
    data_type = kwargs.pop('data_type', scri.DataType[scri.SpinWeights.index(s)])
    t_0 = kwargs.pop('t_0', -20.0)
    t_1 = kwargs.pop('t_1', 20.0)
    dt = kwargs.pop('dt', 1./10.)
    t = np.arange(t_0, t_1+dt, dt)
    n_times = t.size
    omega = complex(kwargs.pop('omega', 0.5))
    data = np.zeros((n_times, sf.LM_total_size(ell_min, ell_max)), dtype=complex)
    data[:, sf.LM_index(ell, m, ell_min)] = np.exp(1j*omega*t)

    if kwargs:
        import pprint
        warnings.warn("\nUnused kwargs passed to this function:\n{0}".format(pprint.pformat(kwargs, width=1)))

    return scri.WaveformModes(t=t, data=data, ell_min=ell_min, ell_max=ell_max,
                              frameType=scri.Inertial, dataType=data_type,
                              r_is_scaled_out=True, m_is_scaled_out=True)


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
    s = kwargs.pop('s', -2)
    ell = kwargs.pop('ell', abs(s))
    m = kwargs.pop('m', -ell)
    ell_min = kwargs.pop('ell_min', abs(s))
    ell_max = kwargs.pop('ell_max', 8)
    data_type = kwargs.pop('data_type', scri.DataType[scri.SpinWeights.index(s)])
    t_0 = kwargs.pop('t_0', -20.0)
    t_1 = kwargs.pop('t_1', 20.0)
    dt = kwargs.pop('dt', 1./10.)
    t = np.arange(t_0, t_1+dt, dt)
    n_times = t.size
    beta = kwargs.pop('beta', 1.0)
    data = np.zeros((n_times, sf.LM_total_size(ell_min, ell_max)), dtype=complex)
    data[:, sf.LM_index(ell, m, ell_min)] = beta*t

    if kwargs:
        import pprint
        warnings.warn("\nUnused kwargs passed to this function:\n{0}".format(pprint.pformat(kwargs, width=1)))

    return scri.WaveformModes(t=t, data=data, ell_min=ell_min, ell_max=ell_max,
                              frameType=scri.Inertial, dataType=data_type,
                              r_is_scaled_out=True, m_is_scaled_out=True)


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
    s = kwargs.pop('s', -2)
    ell = kwargs.pop('ell', abs(s))
    m = kwargs.pop('m', -ell)
    ell_min = kwargs.pop('ell_min', abs(s))
    ell_max = kwargs.pop('ell_max', 8)
    data_type = kwargs.pop('data_type', scri.DataType[scri.SpinWeights.index(s)])
    t_0 = kwargs.pop('t_0', -20.0)
    t_1 = kwargs.pop('t_1', 20.0)
    dt = kwargs.pop('dt', 1./10.)
    t = np.arange(t_0, t_1+dt, dt)
    n_times = t.size
    beta = kwargs.pop('beta', 1.0)
    data = np.zeros((n_times, sf.LM_total_size(ell_min, ell_max)), dtype=complex)
    data[:, sf.LM_index(ell, m, ell_min)] = beta*t
    supertranslation = np.array(kwargs.pop('supertranslation', np.array([], dtype=complex)), dtype=complex)
    if 'space_translation' in kwargs:
        if supertranslation.size < 4:
            supertranslation.resize((4,))
        supertranslation[1:4] = -sf.vector_as_ell_1_modes(kwargs.pop('space_translation'))
    supertranslation_ell_max = int(math.sqrt(supertranslation.size)-1)
    if supertranslation_ell_max * (supertranslation_ell_max + 2) + 1 != supertranslation.size:
        raise ValueError("Bad number of elements in supertranslation: {0}".format(supertranslation.size))
    for i, (ellpp, mpp) in enumerate(sf.LM_range(0, supertranslation_ell_max)):
        if supertranslation[i] != 0.0:
            mp = m+mpp
            for ellp in range(ell_min, min(ell_max, (ell+ellpp))+1):
                if ellp >= abs(mp):
                    addition = (
                        beta
                        * supertranslation[i]
                        * math.sqrt(((2*ellpp+1)*(2*ell+1)*(2*ellp+1))/(4*math.pi))
                        * sf.Wigner3j(ellpp, ell, ellp, 0, -s, s)*sf.Wigner3j(ellpp, ell, ellp, mpp, m, -mp)
                    )
                    if (s+mp)%2 == 1:
                        addition *= -1
                    data[:, sf.LM_index(ellp, mp, ell_min)] += addition

    if kwargs:
        import pprint
        warnings.warn("\nUnused kwargs passed to this function:\n{0}".format(pprint.pformat(kwargs, width=1)))

    return scri.WaveformModes(t=t, data=data, ell_min=ell_min, ell_max=ell_max,
                              frameType=scri.Inertial, dataType=data_type,
                              r_is_scaled_out=True, m_is_scaled_out=True)
