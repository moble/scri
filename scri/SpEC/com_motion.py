#!/usr/bin/env python

# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>
"""Script and/or library to calculate optimal translation and boost from Horizons.h5

See docstrings below for more detail.

To use this function from another script, simply do something like the following:

    from scri.SpEC.com_motion import com_motion, estimate_avg_com_motion
    x_0, v_0, t_0 = estimate_avg_com_motion('path/to/Horizons.h5')

Other options may be passed to the `estimate_avg_com_motion` function; see its docstring for details.

"""


def com_motion(path_to_horizons_h5, path_to_matter_h5=None, m_A=None, m_B=None):
    """Calculate the center-of-mass motion from Horizons.h5 and/or Matter.h5

    Note that no input checking is done here.  If the data sets are missing, incomplete, or corrupted, you should
    expect to get exceptions or silently incorrect data.

    Parameters
    ----------
    path_to_horizons_h5 : string, optional [default is 'Horizons.h5']
        Absolute or relative path to 'Horizons.h5'
    path_to_matter_h5: None or string [default is None]
        Absolute or relative path to 'Matter.h5' for neutron star trajectories.
        If None, we assume that only black holes are involved, so that both A
        and B trajectories will be in Horizons.h5.
    m_A: None or float [default is None]
        If None, the mass will be read from Horizons.h5 (for BHs) or Matter.h5
        (for NSs).
    m_B: None or float [default is None]
        If None, the mass will be read from Horizons.h5 (for BHs) or Matter.h5
        (for NSs).

    Returns
    -------
    t : float array
        These are simply the coordinate times
    com : 3xN float array
        Position of CoM at each instant of time

    """
    import numpy as np
    import h5py

    if path_to_matter_h5 is None:
        with h5py.File(path_to_horizons_h5, "r") as horizons:
            t = horizons["AhA.dir/ChristodoulouMass.dat"][:, 0]
            m_A = horizons["AhA.dir/ChristodoulouMass.dat"][:, 1]
            x_A = horizons["AhA.dir/CoordCenterInertial.dat"][:, 1:]
            m_B = horizons["AhB.dir/ChristodoulouMass.dat"][:, 1]
            x_B = horizons["AhB.dir/CoordCenterInertial.dat"][:, 1:]
            m = m_A + m_B
            CoM = ((m_A[:, np.newaxis] * x_A) + (m_B[:, np.newaxis] * x_B)) / m[:, np.newaxis]
    else:
        # NOTE: Currently only supports BHNS, not NSNS
        with h5py.File(path_to_horizons_h5, "r") as horizons:
            t = horizons["AhA.dir/ChristodoulouMass.dat"][:, 0]
            if m_A is None:
                m_A = horizons["AhA.dir/ChristodoulouMass.dat"][:, 1]
            else:
                m_A = np.atleast_1d(m_A)
            x_A = horizons["AhA.dir/CoordCenterInertial.dat"][:, 1:]
        with h5py.File(path_to_matter_h5, "r") as matter:
            if m_B is None:
                m_B = matter["RestMass.dat"][:, 1]
            else:
                m_B = np.atleast_1d(m_B)
            x_B = matter["InertialCenterOfMass.dat"][:, 1:]
        m = m_A + m_B
        CoM = ((m_A[:, np.newaxis] * x_A) + (m_B[:, np.newaxis] * x_B)) / m[:, np.newaxis]

        # The waveform will be in units of the total mass, so we need to convert for compatibility
        t /= m
        CoM /= m[:, np.newaxis]

    return t, CoM


def estimate_avg_com_motion(
    path_to_horizons_h5="Horizons.h5",
    skip_beginning_fraction=0.01,
    skip_ending_fraction=0.10,
    plot=False,
    fit_acceleration=False,
    path_to_matter_h5=None,
    m_A=None,
    m_B=None,
):
    """Calculate optimal translation and boost from Horizons.h5

    This returns the optimal initial position and velocity such that the CoM is
    best approximated as having these initial values.  If the coordinate system
    is transformed by these values, the new CoM motion will be as close to the
    origin as possible (in the sense of squared distance from the origin
    integrated over time).

    The translation to be applied to the data should be calculated given the
    values returned by this function as

        np.array([x_i+v_i*t_j+0.5*a_i*t_j**2 for t_j in t])


    Parameters
    ----------
    path_to_horizons_h5: string, optional [default is 'Horizons.h5']
        Absolute or relative path to 'Horizons.h5'
    skip_beginning_fraction : float, optional
        Exclude this portion from the beginning of the data.  Note that this is
        a fraction, rather than a percentage.  The default value is 0.01,
        meaning the first 1% of the data will be ignored.
    skip_ending_fraction : float, optional
        Exclude this portion from the end of the data.  Note that this is a
        fraction, rather than a percentage.  The default value is 0.10, meaning
        the last 10% of the data will be ignored.
    plot : bool, optional
        If True, save plot showing CoM tracks before and after offset and
        translation, to file `CoM_before_and_after_translation.pdf` in the same
        directory as Horizons.h5.  Default: False.
    fit_acceleration: bool, optional
        If True, allow for an acceleration in the fit, and return as third
        parameter.  Default: False.
    path_to_matter_h5: None or string [default is None]
        Absolute or relative path to 'Matter.h5' for neutron star trajectories.
        If None, we assume that only black holes are involved, so that both A
        and B trajectories will be in Horizons.h5.
    m_A: None or float [default is None]
        If None, the mass will be read from Horizons.h5 (for BHs) or Matter.h5
        (for NSs).
    m_B: None or float [default is None]
        If None, the mass will be read from Horizons.h5 (for BHs) or Matter.h5
        (for NSs).

    Returns
    -------
    x_i : length-3 array of floats
        Best-fit initial position of the center of mass
    v_i : length-3 array of floats
        Best-fit initial velocity of the center of mass
    a_i : length-3 array of floats
        Best-fit initial acceleration of the center of mass [only if
        `fit_acceleration=True` is in input arguments]
    t_i : float
        Initial time used.  This is determined by the `skip_beginning_fraction`
        input parameter.
    t_f : float
        Final time used.  This is determined by the `skip_ending_fraction`
        input parameter.

    """
    import os.path
    import numpy as np
    from scipy.integrate import simps

    t, com = com_motion(path_to_horizons_h5, path_to_matter_h5, m_A, m_B)

    # We will be skipping the beginning and end of the data;
    # this gives us the initial and final indices
    t_i, t_f = t[0] + (t[-1] - t[0]) * skip_beginning_fraction, t[-1] - (t[-1] - t[0]) * skip_ending_fraction
    i_i, i_f = np.argmin(np.abs(t - t_i)), np.argmin(np.abs(t - t_f))

    # Find the optimum analytically
    CoM_0 = simps(com[i_i : i_f + 1], t[i_i : i_f + 1], axis=0)
    CoM_1 = simps((t[:, np.newaxis] * com)[i_i : i_f + 1], t[i_i : i_f + 1], axis=0)
    if fit_acceleration:
        CoM_2 = simps((t[:, np.newaxis] ** 2 * com)[i_i : i_f + 1], t[i_i : i_f + 1], axis=0)
        x_i = (
            3
            * (
                CoM_0
                * (3 * t_f ** 4 + 12 * t_f ** 3 * t_i + 30 * t_f ** 2 * t_i ** 2 + 12 * t_f * t_i ** 3 + 3 * t_i ** 4)
                - 12 * CoM_1 * (t_f + t_i) * (t_f ** 2 + 3 * t_f * t_i + t_i ** 2)
                + CoM_2 * (10 * t_f ** 2 + 40 * t_f * t_i + 10 * t_i ** 2)
            )
            / (t_f - t_i) ** 5
        )
        v_i = (
            12
            * (
                -3 * CoM_0 * (t_f + t_i) * (t_f ** 2 + 3 * t_f * t_i + t_i ** 2)
                + CoM_1 * (16 * t_f ** 2 + 28 * t_f * t_i + 16 * t_i ** 2)
                + CoM_2 * (-15 * t_f - 15 * t_i)
            )
            / (t_f - t_i) ** 5
        )
        a_i = (
            60
            * (CoM_0 * (t_f ** 2 + 4 * t_f * t_i + t_i ** 2) + CoM_1 * (-6 * t_f - 6 * t_i) + 6 * CoM_2)
            / (t_f - t_i) ** 5
        )
    else:
        x_i = 2 * (CoM_0 * (2 * t_f ** 3 - 2 * t_i ** 3) + CoM_1 * (-3 * t_f ** 2 + 3 * t_i ** 2)) / (t_f - t_i) ** 4
        v_i = 6 * (CoM_0 * (-t_f - t_i) + 2 * CoM_1) / (t_f - t_i) ** 3
        a_i = 0.0

    # If desired, save the plots
    if plot:
        import matplotlib as mpl

        mpl.use("Agg")  # Must come after importing mpl, but before importing plt
        import matplotlib.pyplot as plt

        directory = os.path.dirname(os.path.abspath(path_to_horizons_h5))
        try:
            from subprocess import check_output

            SXS_BBH = check_output(
                "awk '/alternative-names/{{print $3}}' " + os.path.join(directory, "metadata.txt"), shell=True
            )
            if SXS_BBH:
                SXS_BBH = "\n" + SXS_BBH.strip()
        except:
            SXS_BBH = ""
        delta_x = np.array([x_i + v_i * t_j + 0.5 * a_i * t_j ** 2 for t_j in t])
        comprm = com - delta_x
        max_displacement = np.linalg.norm(delta_x, axis=1).max()
        max_d_color = min(1.0, 10 * max_displacement)
        fig = plt.figure(figsize=(10, 7))
        plt.plot(t, com, alpha=0.25, lw=1.5)
        plt.gca().set_prop_cycle(None)
        lineObjects = plt.plot(t, comprm, lw=2)
        plt.xlabel(r"Coordinate time")
        plt.ylabel(r"CoM coordinate values")
        plt.legend(iter(lineObjects), ("x", "y", "z"), loc="upper left")
        fig.text(
            0.5,
            0.91,
            directory
            + SXS_BBH
            + "\n$x_i$ = [{}]".format(", ".join([str(tmp) for tmp in x_i]))
            + "\n$v_i$ = [{}]".format(", ".join([str(tmp) for tmp in v_i])),
            fontsize=8,
            ha="center",
        )
        fig.text(
            0.004,
            0.004,
            str(max_displacement),
            fontsize=24,
            ha="left",
            va="bottom",
            bbox=dict(facecolor=mpl.cm.jet(max_d_color), alpha=max_d_color, boxstyle="square,pad=0.2"),
        )
        fig.savefig(os.path.join(os.path.dirname(path_to_horizons_h5), "CoM_before_and_after_transformation.pdf"))
        plt.close()

    print("Optimal x_i: [{}, {}, {}]".format(*x_i))
    print("Optimal v_i: [{}, {}, {}]".format(*v_i))
    if fit_acceleration:
        print("Optimal a_i: [{}, {}, {}]".format(*a_i))
    print(f"t_i, t_f: {t_i}, {t_f}")

    if fit_acceleration:
        return x_i, v_i, a_i, t_i, t_f
    else:
        return x_i, v_i, t_i, t_f


def remove_avg_com_motion(
    path_to_waveform_h5="rhOverM_Asymptotic_GeometricUnits.h5/Extrapolated_N2.dir",
    path_to_horizons_h5=None,
    skip_beginning_fraction=0.01,
    skip_ending_fraction=0.10,
    plot=False,
    file_write_mode="w",
    path_to_matter_h5=None,
    m_A=None,
    m_B=None,
    file_format="NRAR",
):
    """Rewrite waveform data in center-of-mass frame

    This simply uses `estimate_avg_com_motion`, and then transforms to that
    frame as appropriate.  Most of the options are simply passed to that
    function.  Note, however, that the path to the Horizons.h5 file defaults to
    the directory of the waveform H5 file.

    Additional parameters
    ---------------------
    path_to_waveform_h5 : str, optional
        Absolute or relative path to SpEC waveform file, including the
        directory within the H5 file, if appropriate.  Default value is
        'rhOverM_Asymptotic_GeometricUnits.h5/Extrapolated_N2.dir'.
    path_to_horizons_h5: None or string [default is None]
        Absolute or relative path to 'Horizons.h5'.  If None, this will be
        inferred from the waveform path.
    path_to_matter_h5: None or string [default is None]
        Absolute or relative path to 'Matter.h5' for neutron star trajectories.
        If None, this will be inferred from the waveformpath.  If the file does
        not exist, we assume that only black holes are involved, so that both A
        and B trajectories will be in Horizons.h5.
    m_A: None or float [default is None]
        If None and there is no Matter.h5 file, the mass will be read from
        Horizons.h5; otherwise, the mass will be read from the metadata
        `reference_mass1`.
    m_B: None or float [default is None]
        If None and there is no Matter.h5 file, the mass will be read from
        Horizons.h5; otherwise, the mass will be read from the metadata
        `reference_mass2`.
    file_format: 'NRAR' or 'RPXMB' [default is 'NRAR']
        The file format of the waveform data H5 file. The 'NRAR' format is the
        default file format found in the SXS Catalog. The 'RPXMB' format is data
        compressed with the rotating_paired_xor_multishuffle_bzip2 scheme.

    Returns
    -------
    w_m : WaveformModes object
        This is the transformed object in the new frame

    """
    import os.path
    import pathlib
    import re
    import numpy as np
    from .file_io import read_from_h5, write_to_h5, rotating_paired_xor_multishuffle_bzip2
    from .. import h, hdot, psi4, psi3, psi2, psi1, psi0

    directory = os.path.dirname(os.path.abspath(path_to_waveform_h5.split(".h5", 1)[0] + ".h5"))
    subdir = os.path.basename(path_to_waveform_h5.split(".h5", 1)[1])

    # Read the waveform data in
    if file_format.lower() == "nrar":
        w_m = read_from_h5(path_to_waveform_h5)
    elif file_format.lower() == "rpxmb" or file_format.lower() == "rpxm":
        w_m = rotating_paired_xor_multishuffle_bzip2.load(path_to_waveform_h5)[0].to_inertial_frame()
    else:
        raise ValueError(f"File format {file_format} not recognized. Must be either 'NRAR' or 'RPXMB'.")

    if path_to_horizons_h5 is None:
        path_to_horizons_h5 = os.path.join(directory, "Horizons.h5")

    if path_to_matter_h5 is None:
        tmp = pathlib.Path(directory) / "Matter.h5"
        if tmp.exists():
            path_to_matter_h5 = str(tmp)
            if m_A is None:
                m_A = w_m.metadata.reference_mass1
            if m_B is None:
                m_B = w_m.metadata.reference_mass2

    # Get the CoM motion from Horizons.h5
    x_0, v_0, t_0, t_f = estimate_avg_com_motion(
        path_to_horizons_h5=path_to_horizons_h5,
        skip_beginning_fraction=skip_beginning_fraction,
        skip_ending_fraction=skip_ending_fraction,
        plot=plot,
        path_to_matter_h5=path_to_matter_h5,
        m_A=m_A,
        m_B=m_B,
    )

    # Set up the plot and plot the original data
    if plot:
        import matplotlib as mpl

        mpl.use("Agg")  # Must come after importing mpl, but before importing plt
        import matplotlib.pyplot as plt
        import scri.plotting

        try:
            if isinstance(w_m.metadata.alternative_names, list):
                SXS_BBH = "\n" + ", ".join(w_m.metadata.alternative_names)
            else:
                SXS_BBH = "\n" + w_m.metadata.alternative_names
        except:
            SXS_BBH = ""
        t_merger = w_m.max_norm_time() - 300.0
        t_ringdown = w_m.max_norm_time() + 100.0
        t_final = w_m.t[-1]
        delta_x = np.array([x_0 + v_0 * t_j for t_j in w_m.t])
        max_displacement = np.linalg.norm(delta_x, axis=1).max()
        max_d_color = min(1.0, 9 * max_displacement)
        LM_indices1 = [[2, 2], [2, 1], [3, 3], [3, 1], [4, 3]]
        LM_indices1 = [[ell, m] for ell, m in LM_indices1 if [ell, m] in w_m.LM.tolist()]
        indices1 = [(ell * (ell + 1) - 2 ** 2 + m) for ell, m in LM_indices1]
        LM_indices2 = [[2, 2], [3, 2], [4, 4], [4, 2]]
        LM_indices2 = [[ell, m] for ell, m in LM_indices2 if [ell, m] in w_m.LM.tolist()]
        indices2 = [(ell * (ell + 1) - 2 ** 2 + m) for ell, m in LM_indices2]
        fig1 = plt.figure(1, figsize=(10, 7))
        plt.gca().set_xscale("merger_zoom", t_merger=t_merger, t_ringdown=t_ringdown, t_final=t_final)
        lines1 = plt.semilogy(w_m.t, abs(w_m.data[:, indices1]), alpha=0.35, lw=1.5)
        fig2 = plt.figure(2, figsize=(10, 7))
        plt.gca().set_xscale("merger_zoom", t_merger=t_merger, t_ringdown=t_ringdown, t_final=t_final)
        lines2 = plt.semilogy(w_m.t, abs(w_m.data[:, indices2]), alpha=0.35, lw=1.5)

    # Import auxilliary waveforms if we are transforming Psi3, Psi2, Psi1, or Psi0. To apply a CoM
    # correction to these data types, information is required from all higher ranked Weyl scalars,
    # e.g. Psi2 requires information from Psi3 and Psi4.
    aux_waveforms = {}
    if w_m.dataType not in [h, hdot, psi4]:
        file_prefix = ["rMPsi4", "r2Psi3", "r3Psi2OverM", "r4Psi1OverM2"]
        regex_template = r"""(r2News|rhOverM|rMPsi4|r2Psi3|r3Psi2OverM|r4Psi1OverM2|r5Psi0OverM3)"""
        for i in range(5 - w_m.dataType):
            match = re.search(regex_template, path_to_waveform_h5)
            # Assume the auxiliary waveforms are in the same directory as the waveform to be transformed
            aux_path = path_to_waveform_h5.replace(match.group(1), file_prefix[i])
            try:
                if file_format.lower() == "nrar":
                    aux_waveforms[f"psi{4-i}_modes"] = read_from_h5(aux_path)
                elif file_format.lower() == "rpxmb" or file_format.lower() == "rpxm":
                    aux_waveforms[f"psi{4-i}_modes"] = rotating_paired_xor_multishuffle_bzip2.load(aux_path)[0]
                    aux_waveforms[f"psi{4-i}_modes"].to_inertial_frame()
            except (FileNotFoundError, OSError):
                raise FileNotFoundError(
                    f"A transformation of {w_m.data_type_string} requires waveform data from Psi{4-i}, "
                    "but the file {aux_path} could not be found."
                )

    # Transform the mode data
    w_m = w_m.transform(space_translation=x_0, boost_velocity=v_0, **aux_waveforms)

    # Write the data to the new file
    if file_format.lower() == "nrar":
        path_to_new_waveform_h5 = re.sub(
            w_m.descriptor_string + "_",
            "",  # Remove 'rhOverM_', 'rMPsi4_', or whatever
            path_to_waveform_h5.replace(".h5", "_CoM.h5", 1),  # Add '_CoM' once
            flags=re.I,
        )  # Ignore case of 'psi4'/'Psi4', etc.
        write_to_h5(
            w_m,
            path_to_new_waveform_h5,
            file_write_mode=file_write_mode,
            attributes={"space_translation": x_0, "boost_velocity": v_0},
        )
    w_m.boost_velocity = v_0
    w_m.space_translation = x_0
    if file_format.lower() == "rpxmb" or file_format.lower() == "rpxm":
        path_to_new_waveform_h5 = path_to_waveform_h5.replace(".h5", "_CoM.h5", 1)
        rotating_paired_xor_multishuffle_bzip2.save(w_m, path_to_new_waveform_h5)

    # Finish by plotting the new data and save to PDF
    if plot:
        plt.figure(1)
        for line, index, (ell, m) in zip(lines1, indices1, LM_indices1):
            plt.semilogy(w_m.t, abs(w_m.data[:, index]), color=plt.getp(line, "color"), lw=1.5, label=f"({ell}, {m})")
        plt.figure(2)
        for line, index, (ell, m) in zip(lines2, indices2, LM_indices2):
            plt.semilogy(w_m.t, abs(w_m.data[:, index]), color=plt.getp(line, "color"), lw=1.5, label=f"({ell}, {m})")
        for fig, num in [(fig1, 1), (fig2, 2)]:
            plt.figure(num)
            plt.axvline(t_merger, color="black", lw=2, alpha=0.125)
            plt.axvline(t_ringdown, color="black", lw=2, alpha=0.125)
            plt.xlabel(r"Coordinate time")
            plt.ylabel(r"Mode amplitudes")
            plt.xlim(0.0, w_m.t[-1])
            plt.ylim(1e-6, 0.6)
            plt.grid(axis="y")
            fig.text(
                0.5,
                0.91,
                os.path.abspath(path_to_waveform_h5)
                + SXS_BBH
                + "\n$x_0$ = [{}]".format(", ".join([str(tmp) for tmp in x_0]))
                + "\n$v_0$ = [{}]".format(", ".join([str(tmp) for tmp in v_0])),
                fontsize=8,
                ha="center",
            )
            fig.text(
                0.004,
                0.004,
                str(max_displacement),
                fontsize=24,
                ha="left",
                va="bottom",
                bbox=dict(facecolor=mpl.cm.jet(max_d_color), alpha=max_d_color, boxstyle="square,pad=0.2"),
            )
            plt.legend(loc="upper left", framealpha=0.9)
            fig.savefig(os.path.join(directory, "Modes_{}_{}_{}.pdf".format(w_m.descriptor_string, subdir[:-4], num)))
            plt.close()

    return w_m


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate optimal translation and boost from Horizons.h5",
        epilog=(
            "This simply attempts to minimize the squared distance of the CoM from the origin by applying some "
            "constant offset and linear-in-time translation.  Because it uses optimization, it may be sensitive "
            "to the initial guess and may not converge."
        ),
    )
    parser.add_argument(
        "path_to_horizons_h5", default="Horizons.h5", nargs="?", action="store", help="path to the Horizons.h5 file"
    )
    parser.add_argument(
        "--skip_beginning_fraction",
        default=0.01,
        nargs=1,
        type=float,
        action="store",
        help="optional initial portion of the data to ignore",
    )
    parser.add_argument(
        "--skip_ending_fraction",
        default=0.10,
        nargs=1,
        type=float,
        action="store",
        help="optional final portion of the data to ignore",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="make plots as `CoM_before_and_after_transformation.pdf` in same directory as Horizons.h5",
    )
    args = parser.parse_args()

    estimate_avg_com_motion(
        path_to_horizons_h5=args.path_to_horizons_h5,
        skip_beginning_fraction=args.skip_beginning_fraction,
        skip_ending_fraction=args.skip_ending_fraction,
        plot=args.plot,
    )
