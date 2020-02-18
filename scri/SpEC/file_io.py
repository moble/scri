# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import

import warnings
import numpy as np
from quaternion.numba_wrapper import jit, xrange
import spherical_functions as sf
from .. import (WaveformModes, FrameNames, DataType, DataNames, UnknownDataType, h, hdot, psi4, psi3, psi2, psi1, psi0)
from sxs.metadata import Metadata


def translate_data_types_GWFrames_to_waveforms(d):
    if d < 8:
        return {0: UnknownDataType, 1: h, 2: hdot, 3: psi4, 4: psi3, 5: psi2, 6: psi1, 7: psi0}[d]
    else:
        return DataType[d-8]


def translate_data_types_waveforms_to_GWFrames(d):
    if d in [UnknownDataType, h, hdot, psi4, psi3, psi2, psi1, psi0]:
        return {UnknownDataType: 0, h: 1, hdot: 2, psi4: 3, psi3: 4, psi2: 5, psi1: 6, psi0: 7}[d]
    else:
        return d+8


@jit
def index_is_monotonic(y):
    length = y.size
    monotonic = np.ones_like(y, dtype=np.bool_)
    direction = y[-1] - y[0]
    if direction > 0.0:
        max_value = y[0]
        for i in xrange(1, length):
            if y[i] <= max_value:
                monotonic[i] = False
            else:
                max_value = y[i]
    else:
        min_value = y[0]
        for i in xrange(1, length):
            if y[i] >= min_value:
                monotonic[i] = False
            else:
                min_value = y[i]
    return monotonic


def monotonic_indices(y):
    indices = np.arange(y.size)
    return indices[index_is_monotonic(y)]


def monotonize(y):
    return y[monotonic_indices(y)]


def read_from_h5(file_name, **kwargs):
    """Read data from an H5 file in SXS format

    Note that SXS format is essentially compatible with NRAR format.  The existence of this function is not to be
    taken as the author's endorsement of either SXS or NRAR format.

    Parameters
    ----------
    file_name : str
        Path to H5 file containing the data, optionally including the path within the file itself to the directory
        containing the data.  For example, the standard SXS data with N=2 might be obtained with the file name
        `'rhOverM_Asymptotic_GeometricUnits.h5/Extrapolated_N2.dir'`.

    Keyword parameters
    ------------------
    frameType : int, optional
    dataType : int, optional
    r_is_scaled_out : bool, optional
    m_is_scaled_out : bool, optional
        These four parameters are documented in the docstring of the WaveformBase object.  Note that if any of these
        is present in the H5 file (which is not common) that value will override this argument.  If neither the file
        nor these parameters are present, defaults will be applied, assuming that the frame is inertial, R and M are
        both scaled out, and the data type (hdot, h, psi4, psi4, psi2, psi1, or psi0) can be gleaned from `file_name`.

    """

    import os.path
    import re
    import h5py
    import quaternion

    # This unfortunate concoction is needed to determine the (ell,m) values of the various mode data sets
    pattern_Ylm = re.compile(r"""Y_l(?P<L>[0-9]+)_m(?P<M>[-+0-9]+)\.dat""")

    # Initialize an empty object to be filled with goodies
    w = WaveformModes(constructor_statement='scri.SpEC.read_from_h5("{0}", **{1})'.format(file_name, kwargs))

    # Get an h5py handle to the desired part of the h5 file
    try:
        file_name, root_group = file_name.rsplit('.h5', 1)
        file_name += '.h5'
    except ValueError:
        root_group = ''  # FileName is just a file, not a group in a file
    try:
        f_h5 = h5py.File(file_name, 'r')
    except IOError:
        print("\n`read_from_h5` could not open the file '{0}'\n\n".format(file_name))
        raise
    if root_group:
        f = f_h5[root_group]
    else:
        f = f_h5

    # If it exists, add the metadata file to `w` as an object.  So, for example, the initial spin on
    # object 1 can be accessed as `w.metadata.initial_spin1`.  See the documentation of
    # `sxs.metadata.Metadata` for more details.  And in IPython, tab completion works on the
    # `w.metadata` object.
    try:
        w.metadata = Metadata.from_file(os.path.join(os.path.dirname(file_name), 'metadata'),
                                        ignore_invalid_lines=True, cache_json=False)
    except:
        pass  # Probably couldn't find the metadata.json/metadata.txt file

    try:  # Make sure the h5py.File gets closed, even in the event of an exception

        # Add the old history to the new history, if found
        try:
            try:
                old_history = f['History.txt'][()].decode()
            except AttributeError:
                old_history = f['History.txt'][()]
            w._append_history("", 0)
            w._append_history("<previous_history>", 2)
            w._append_history(old_history, 1)
            w._append_history("</previous_history>", 2)
        except KeyError:
            pass  # Did not find a history

        # Get the frame data, converting to quaternion objects
        try:
            w.frame = quaternion.as_quat_array(f['Frame'])
        except KeyError:
            pass  # There was no frame data

        # Get the descriptive items
        try:
            w.frameType = int(f.attrs['FrameType'])
        except KeyError:
            if 'frameType' in kwargs:
                w.frameType = int(kwargs.pop('frameType'))
            else:
                warning = ("\n`frameType` was not found in '{0}' or the keyword arguments.\n".format(file_name) +
                           "Using default value `{0}`.  You may want to set it manually.\n\n".format(FrameNames[1]))
                warnings.warn(warning)
                w.frameType = 1
        try:
            w.dataType = translate_data_types_GWFrames_to_waveforms(int(f.attrs['DataType']))
        except KeyError:
            if 'dataType' in kwargs:
                w.dataType = int(kwargs.pop('dataType'))
            else:
                found = False
                for type_int, type_name in zip(reversed(DataType), reversed(DataNames)):
                    if type_name.lower() in file_name.lower():
                        found = True
                        w.dataType = type_int
                        warning = ("\n`dataType` was not found in '{0}' or the keyword arguments.\n".format(file_name) +
                                   "Using default value `{0}`.  You may want to set it manually.".format(type_name))
                        warnings.warn(warning)
                        break
                if not found:
                    warning = ("\n`dataType` was not found in '{0}' or the keyword arguments.\n".format(file_name) +
                               "You may want to set it manually.")
                    warnings.warn(warning)
        try:
            w.r_is_scaled_out = bool(f.attrs['RIsScaledOut'])
        except KeyError:
            if 'r_is_scaled_out' in kwargs:
                w.r_is_scaled_out = bool(kwargs.pop('r_is_scaled_out'))
            else:
                warning = ("\n`r_is_scaled_out` was not found in '{0}' or the keyword arguments.\n".format(file_name) +
                           "Using default value `True`.  You may want to set it manually.\n\n")
                warnings.warn(warning)
                w.r_is_scaled_out = True
        try:
            w.m_is_scaled_out = bool(f.attrs['MIsScaledOut'])
        except KeyError:
            if 'm_is_scaled_out' in kwargs:
                w.m_is_scaled_out = bool(kwargs.pop('m_is_scaled_out'))
            else:
                warning = ("\n`m_is_scaled_out` was not found in '{0}' or the keyword arguments.\n".format(file_name) +
                           "Using default value `True`.  You may want to set it manually.\n\n")
                warnings.warn(warning)
                w.m_is_scaled_out = True

        # Get the names of all the data sets in the h5 file, and check for matches
        YLMdata = [data_set for data_set in list(f) for m in [pattern_Ylm.search(data_set)] if m]
        if len(YLMdata) == 0:
            raise ValueError("Couldn't understand data set names in '{0}'.\n".format(file_name) +
                             "Maybe you need to add the directory within the h5 file.\n" +
                             "E.g.: '{0}/Extrapolated_N2.dir'.".format(file_name))

        # Sort the data set names by increasing ell, then increasing m
        YLMdata = sorted(YLMdata, key=lambda data_set: [int(pattern_Ylm.search(data_set).group('L')),
                                                        int(pattern_Ylm.search(data_set).group('M'))])
        LM = np.array(sorted([[int(m.group('L')), int(m.group('M'))]
                              for data_set in YLMdata for m in [pattern_Ylm.search(data_set)] if m]))
        ell_min, ell_max = min(LM[:, 0]), max(LM[:, 0])
        if not np.array_equal(LM, sf.LM_range(ell_min, ell_max)):
            raise ValueError("Input [ell,m] modes are not complete.  Found modes:\n{0}\n".format(LM))
        n_modes = len(LM)

        # Get the time data (assuming all are equal)
        T = f[YLMdata[0]][:, 0]
        monotonic = index_is_monotonic(T)
        w.t = T[monotonic]
        n_times = len(w.t)

        # Loop through, setting data in each mode
        w.data = np.empty((n_times, n_modes), dtype=complex)
        for m, DataSet in enumerate(YLMdata):
            if f[DataSet].shape[0] != n_times:
                raise ValueError("The number of time steps in this dataset should be {0}; ".format(n_times) +
                                 "it is {0} in '{1}'.".format(f[DataSet].shape[0], DataSet))
            w.data[:, m] = f[DataSet][:, 1:3].view(dtype=np.complex)[monotonic, 0]

        # Now that the data is set, we can set these
        w.ells = ell_min, ell_max

        # Check up on the validity of the waveform
        if not w.ensure_validity(alter=True, assertions=False):
            raise ValueError("The data resulting from this input is invalid")

    except KeyError:
        print("\nThis H5 file appears to have not stored all the required information.\n\n")
        raise  # Re-raise the exception after adding our information

    finally:  # Use `finally` to make sure this happens:
        f_h5.close()

    if kwargs:
        import pprint
        warnings.warn("\nUnused kwargs passed to this function:\n{0}".format(pprint.pformat(kwargs, width=1)))

    return w


def write_to_h5(w, file_name, file_write_mode='w', attributes={}, use_NRAR_format=True):
    """
    Output the Waveform to an HDF5 file. Default behavior uses the NRAR format. 

    Note that the file_name is prepended with some descriptive information involving the data type and the frame type,
    such as 'rhOverM_Corotating_' or 'rMpsi4_Aligned_'.

    """

    import os.path
    import h5py
    import warnings

    group = None
    if '.h5' in file_name and not file_name.endswith('.h5'):
        file_name, group = file_name.split('.h5')
        file_name += '.h5'
    # Add descriptive prefix to file_name
    base_name = w.descriptor_string + '_' + os.path.basename(file_name)
    if not os.path.dirname(file_name):
        file_name = base_name
    else:
        file_name = os.path.join(os.path.dirname(file_name), base_name)
    # Open the file for output
    try:
        f = h5py.File(file_name, file_write_mode)
    except IOError:  # If that did not work...
        print("write_to_h5 was unable to open the file '{0}'.\n\n".format(file_name))
        raise  # re-raise the exception after the informative message above
    try:
        # If we are writing to a group within the file, create it
        if group:
            g = f.create_group(group)
        else:
            g = f
        # Now write all the data to various groups in the file
        g.attrs['OutputFormatVersion'] = 'scri.SpEC'
        g.create_dataset("History.txt", data='\n'.join(w.history) + '\n\nwrite_to_h5({0}, {1})\n'.format(w, file_name))
        g.attrs['FrameType'] = w.frameType
        g.attrs['DataType'] = translate_data_types_waveforms_to_GWFrames(w.dataType)
        g.attrs['RIsScaledOut'] = int(w.r_is_scaled_out)
        g.attrs['MIsScaledOut'] = int(w.m_is_scaled_out)
        if (len(w.version_hist)>0):
            g.create_dataset("VersionHist.ver", (len(w.version_hist),2), data = w.version_hist, dtype=h5py.special_dtype(vlen=bytes))
        for attr in attributes:
            try:
                g.attrs[attr] = attributes[attr]
            except:
                warning = "scri.SpEC.write_to_h5 unable to output attribute {0}={1}".format(attr, attributes[attr])
                warnings.warn(warning)
        if use_NRAR_format:
            for i_m in range(w.n_modes):
                ell, m = w.LM[i_m]
                Data_m = g.create_dataset("Y_l{0}_m{1}.dat".format(ell, m),
                                          data=[[t, d.real, d.imag] for t, d in zip(w.t, w.data[:, i_m])],
                                          compression="gzip", shuffle=True)
                Data_m.attrs['ell'] = ell
                Data_m.attrs['m'] = m
        else:
            g.create_dataset("Time", data=w.t.tolist(), compression="gzip", shuffle=True)
            if(len(w.frame)>0) :
                g.create_dataset("Frame", data=[[r.w, r.x, r.y, r.z] for r in w.frame])
            else:
                g.create_dataset("Frame", shape=())
            Data = g.create_group("Data")
            for i_m in range(w.n_modes):
                ell,m = w.LM[i_m]
                Data_m = Data.create_dataset("l{0}_m{1:+}".format(int(ell), int(m)), data=w.data[:,i_m],
                                             compression="gzip", shuffle=True)
                Data_m.attrs['ell'] = ell
                Data_m.attrs['m'] = m
    finally:  # Use `finally` to make sure this happens:
        f.close()
