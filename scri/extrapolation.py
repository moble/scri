import numpy as np
from numpy.polynomial.polynomial import polyfit

mode_regex = r"""Y_l(?P<L>[0-9]+)_m(?P<M>[-+0-9]+)\.dat"""


def pick_Ch_mass(filename="Horizons.h5"):
    """Deduce the best Christodoulou mass by finding the statistical "mode" (after binning)."""
    from h5py import File
    from os.path import isdir
    from numpy import histogram

    if isdir(filename):
        filename = filename + "Horizons.h5"
    try:
        f = File(filename, "r")
    except OSError:
        print(f"pick_Ch_mass could not open the file '{filename}'")
        raise
    ChMass = f["AhA.dir/ChristodoulouMass.dat"][:, 1] + f["AhB.dir/ChristodoulouMass.dat"][:, 1]
    f.close()
    hist, bins = histogram(ChMass, bins=len(ChMass))
    return bins[hist.argmax()]


def monotonic_indices(T, MinTimeStep=1.0e-3):
    """Given an array of times, return the indices that make the array strictly monotonic."""
    from numpy import delete

    Ind = range(len(T))
    Size = len(Ind)
    i = 1
    while i < Size:
        if T[Ind[i]] <= T[Ind[i - 1]] + MinTimeStep:
            j = 0
            while T[Ind[j]] + MinTimeStep < T[Ind[i]]:
                j += 1
            # erase data from j (inclusive) to i (exclusive)
            Ind = delete(Ind, range(j, i))
            Size = len(Ind)
            i = j - 1
        i += 1
    return Ind


def intersection(t1, t2, min_step=None, min_time=None, max_time=None):
    """Return the intersection of two time sequences.

    The time step at each point is the minimum of the time steps in
    t1 and t2 at that instant, or min_step, whichever is greater.
    The output starts at the earliest moment common to t1 and t2, or
    min_time, whichever is greater.

    Parameters
    ----------
    t1: 1-d float array
    t2: 1-d float array
    min_step: float
    min_time: float
    max_time: float

    Returns
    -------
    1-d float array

    """
    import numpy as np

    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    if t1.size == 0:
        raise ValueError("t1 is empty.  Assuming this is not desired.")
    if t2.size == 0:
        raise ValueError("t2 is empty.  Assuming this is not desired.")
    t = np.empty(t1.size + t2.size)
    min1 = t1[0]
    min2 = t2[0]
    if min_time is None:
        mint = max(min1, min2)
    else:
        mint = max(max(min1, min2), min_time)
    max1 = t1[-1]
    max2 = t2[-1]
    if max_time is None:
        maxt = min(max1, max2)
    else:
        maxt = min(min(max1, max2), max_time)
    if mint > max1 or mint > max2:
        message = "Empty intersection in t1=[{0}, ..., {1}], t2=[{2}, ..., {3}] " + "with min_time={4}"
        raise ValueError(message.format(min1, max1, min2, max2, min_time))
    if maxt < min1 or maxt < min2:
        message = "Empty intersection in t1=[{0}, ..., {1}], t2=[{2}, ..., {3}] " + "with max_time={4}"
        raise ValueError(message.format(min1, max1, min2, max2, max_time))
    if min_step is None:
        min_step = min(np.min(np.diff(t1)), np.min(np.diff(t2)))
    t[0] = mint
    I = 0
    I1 = 0
    I2 = 0
    size1 = t1.size
    size2 = t2.size
    while t[I] < maxt:
        # adjust I1 to ensure that t[I] is in the interval ( t1[I1-1], t1[I1] ]
        if t[I] < min1 or t[I] > max1:
            # if t[I] is less than the smallest t1, or greater than the largest t1, I1=0
            I1 = 0
        else:
            I1 = max(I1, 1)
            while t[I] > t1[I1] and I1 < size1:
                I1 += 1
        # adjust I2 to ensure that t[I] is in the interval ( t2[I2-1], t2[I2] ]
        if t[I] < min2 or t[I] > max2:
            # if t[I] is less than the smallest t2, or greater than the largest t2, I2=0
            I2 = 0
        else:
            I2 = max(I2, 1)
            while t[I] > t2[I2] and I2 < size2:
                I2 += 1
        t[I + 1] = t[I] + max(min(t1[I1] - t1[I1 - 1], t2[I2] - t2[I2 - 1]), min_step)
        I += 1
        if t[I] > maxt:
            break
    return t[:I]  # only take the relevant part of the reserved vector


def validate_single_waveform(h5file, filename, WaveformName, ExpectedNModes, ExpectedNTimes, LModes):
    # from sys import stderr
    from re import compile as re_compile

    CompiledModeRegex = re_compile(mode_regex)
    Valid = True
    # Check ArealRadius
    if not h5file[WaveformName + "/ArealRadius.dat"].shape == (ExpectedNTimes, 2):
        Valid = False
        print(
            "{}:{}/ArealRadius.dat\n\tGot shape {}; expected ({}, 2)".format(
                filename,
                WaveformName,
                h5file[WaveformName + "/ArealRadius.dat"].shape,
                ExpectedNTimes,
            )
        )
    # Check AverageLapse
    if not h5file[WaveformName + "/AverageLapse.dat"].shape == (ExpectedNTimes, 2):
        Valid = False
        print(
            "{}:{}/AverageLapse.dat\n\tGot shape {}; expected ({}, 2)".format(
                filename,
                WaveformName,
                h5file[WaveformName + "/AverageLapse.dat"].shape,
                ExpectedNTimes,
            )
        )
    # Check Y_l*_m*.dat
    NModes = len(
        [
            True
            for dataset in list(h5file[WaveformName])
            for m in [CompiledModeRegex.search(dataset)]
            if m and int(m.group("L")) in LModes
        ]
    )
    if not NModes == ExpectedNModes:
        Valid = False
        print(
            "{}:{}/{}\n\tGot {} modes; expected {}".format(filename, WaveformName, mode_regex, NModes, ExpectedNModes)
        )
    for dataset in list(h5file[WaveformName]):
        if CompiledModeRegex.search(dataset):
            if not h5file[WaveformName + "/" + dataset].shape == (ExpectedNTimes, 3):
                Valid = False
                (
                    "{}:{}/{}\n\tGot shape {}; expected ({}, 3)".format(
                        filename,
                        WaveformName,
                        dataset,
                        h5file[WaveformName + "/" + dataset].shape,
                        ExpectedNTimes,
                    )
                )
    return Valid


def validate_group_of_waveforms(h5file, filename, WaveformNames):
    from re import compile as re_compile
    import scri

    DataType = datatype_from_filename(filename)
    # Set the correct LModes based on the spin-weight
    if DataType == scri.psi3 or DataType == scri.psi1:
        LModes = range(1, 200)
    elif DataType == scri.psi2:
        LModes = range(0, 200)
    else:
        LModes = range(2, 200)

    ExpectedNTimes = h5file[WaveformNames[0] + "/ArealRadius.dat"].shape[0]
    ExpectedNModes = len(
        [
            True
            for dataset in list(h5file[WaveformNames[0]])
            for m in [re_compile(mode_regex).search(dataset)]
            if m and int(m.group("L")) in LModes
        ]
    )
    Valid = True
    FailedWaveforms = []
    for WaveformName in WaveformNames:
        if not validate_single_waveform(h5file, filename, WaveformName, ExpectedNModes, ExpectedNTimes, LModes):
            Valid = False
            FailedWaveforms.append(WaveformName)
    if not Valid:
        # from sys import stderr
        print("In '{}', the following waveforms are not valid:\n\t{}".format(filename, "\n\t".join(FailedWaveforms)))
    return Valid

def datatype_from_filename(filename):
    from os.path import basename
    import scri

    DataType = basename(filename).partition("_")[0]
    if "hdot" in DataType.lower():
        DataType = scri.hdot
    elif "h" in DataType.lower():
        DataType = scri.h
    elif "psi4" in DataType.lower():
        DataType = scri.psi4
    elif "psi3" in DataType.lower():
        DataType = scri.psi3
    elif "psi2" in DataType.lower():
        DataType = scri.psi2
    elif "psi1" in DataType.lower():
        DataType = scri.psi1
    elif "psi0" in DataType.lower():
        DataType = scri.psi0
    else:
        DataType = scri.UnknownDataType
        message = (
            "The file '{0}' does not contain a recognizable " +
            "description "+
            "of the data type ('h', 'psi4', 'psi3', 'psi2', "+
            "'psi1', 'psi0')."
        )
        raise ValueError(message.format(filename))
    return DataType

def read_finite_radius_waveform_nrar(filename, WaveformName):

    from h5py import File
    from numpy import array
    import scri
    import os

    # Open the file twice.
    # The first time is for the auxiliary quantities.
    with File(filename,"r") as f:
        W = f[WaveformName]
        
        # Read the time, account for repeated indices
        T = W["AverageLapse.dat"][:, 0]
        Indices = monotonic_indices(T)
        T = T[Indices]

        # Read the other auxiliary quantities
        Radii = array(W["ArealRadius.dat"])[Indices, 1]
        AverageLapse = array(W["AverageLapse.dat"])[Indices, 1]
        CoordRadius = W["CoordRadius.dat"][0, 1]
        InitialAdmEnergy = W["InitialAdmEnergy.dat"][0, 1]

    # The second time we read the file is for the waveform
    waveform = scri.SpEC.file_io.read_from_h5(
        os.path.join(filename,WaveformName),
        frameType=scri.Inertial,
        dataType=datatype_from_filename(filename),
        r_is_scaled_out=True,
        m_is_scaled_out=False, # For now. We will change this later.
    )
    
    return waveform,T,Indices,Radii,AverageLapse,CoordRadius,InitialAdmEnergy

def read_finite_radius_waveform_rpxmb(filename, groupname, WaveformName):
    """This is just a worker function defined for read_finite_radius_data, 
       below, reading a single waveform from an h5 file of many waveforms. 
       You probably don't need to call this directly.
    """

    from h5py import File
    import scri
    from numpy import array

    # Open the file twice.
    # The first time is for the auxiliary quantities.
    with File(filename,"r") as f:
        W = f[groupname][WaveformName]

        # Read the time, account for repeated indices
        T = array(W["Time.dat"])
        Indices = monotonic_indices(T)
        T = T[Indices]

        # Read the other auxiliary quantities
        Radii = array(W["ArealRadius.dat"])[Indices]
        AverageLapse = array(W["AverageLapse.dat"])[Indices]
        CoordRadius = W["CoordRadius.dat"][1]
        InitialAdmEnergy = W["InitialAdmEnergy.dat"][1]

    # The second time we read the file is for the waveform
    # Note that groupname begins with a '/' so
    # os.path.join(filename,groupname,WaveformName) does not work.
    waveform=scri.rpxmb.load(filename+groupname+"/"+
                             WaveformName)[0].to_inertial_frame()

    return waveform,T,Indices,Radii,AverageLapse,CoordRadius,InitialAdmEnergy

def read_finite_radius_waveform(filename, groupname, WaveformName, ChMass):
    from scipy.integrate import cumtrapz as integrate
    from numpy import log, sqrt
    import scri

    if groupname is None:
       waveform,T,Indices,Radii,AverageLapse,CoordRadius,InitialAdmEnergy = \
           read_finite_radius_waveform_nrar(filename,WaveformName)
    else:
       waveform,T,Indices,Radii,AverageLapse,CoordRadius,InitialAdmEnergy = \
           read_finite_radius_waveform_rpxmb(filename,groupname,WaveformName)

    # Rescale and offset the time array so that the time array is
    # approximately the tortoise coordinate.
    T[1:] = integrate(AverageLapse / sqrt(((-2.0 * InitialAdmEnergy) / Radii)\
                                          + 1.0), T) + T[0]
    T -= Radii + (2.0 * InitialAdmEnergy) \
        * log((Radii / (2.0 * InitialAdmEnergy)) - 1.0)
   
    # Now determine the scaling with mass.
    if waveform.dataType == scri.h:
        UnitScaleFactor = 1.0 / ChMass
        RadiusRatioExp = 1.0
    elif waveform.dataType == scri.hdot:
        UnitScaleFactor = 1.0
        RadiusRatioExp = 1.0
    elif waveform.dataType == scri.psi4:
        UnitScaleFactor = ChMass
        RadiusRatioExp = 1.0
    elif waveform.dataType == scri.psi3:
        UnitScaleFactor = 1.0
        RadiusRatioExp = 2.0
    elif waveform.dataType == scri.psi2:
        UnitScaleFactor = 1.0 / ChMass
        RadiusRatioExp = 3.0
    elif waveform.dataType == scri.psi1:
        UnitScaleFactor = 1.0 / ChMass ** 2
        RadiusRatioExp = 4.0
    elif waveform.dataType == scri.psi0:
        UnitScaleFactor = 1.0 / ChMass ** 3
        RadiusRatioExp = 5.0
    else:
        raise ValueError(f'DataType "{waveform.dataType}" is unknown.')

    # Rescale the times and the data
    RadiusRatio = (Radii / CoordRadius) ** RadiusRatioExp
    waveform.t = T/ChMass
    for m,_ in enumerate(waveform.LM):
        waveform.data[:,m] = \
            waveform.data[Indices,m] * RadiusRatio * UnitScaleFactor
    waveform.m_is_scaled_out = True

    # Add the history information
    history =  (
        """# extrapolation.read_finite_radius_waveform""" +
        """({0}, {1}, {2}, {3})"""
    )
    history = history.format(
        filename,
        groupname,
        WaveformName,
        ChMass,
    )
    waveform.history=[history]

    return waveform, Radii/ChMass
        
def read_finite_radius_data(ChMass=0.0,
                            filename="rh_FiniteRadii_CodeUnits.h5",
                            CoordRadii=[]):
    """Read data at various radii, and offset by tortoise coordinate."""

    if ChMass == 0.0:
        raise ValueError("ChMass=0.0 is not a valid input value.")

    from sys import stdout, stderr
    from os.path import basename
    from h5py import File
    from re import compile as re_compile
    import scri

    YLMRegex = re_compile(mode_regex)
    
    # If 'filename' is of the form "h5_file_name.h5/groupname" then we have an
    # RPXMB file, and groupname identifies the quantity we are extrapolating.
    # Otherwise, we have a NRAR finite-radius file and the filename identifies
    # the quantity we are extrapolating.
    groupname = None
    if ".h5" in filename and not filename.endswith(".h5"):
        filename, groupname = filename.split(".h5")
        filename = filename + ".h5"
        if groupname == "/":
            groupname = None

    try:
        f = File(filename, "r")
    except OSError:
        print(f"read_finite_radius_data could not open the file '{filename}'")
        raise
    try:
        # Get list of waveforms we'll be using
        if groupname is None:
            WaveformNames = list(f)
            # VersionHist.ver is not one of the WaveformNames.
            if "VersionHist.ver" in WaveformNames:
                WaveformNames.remove("VersionHist.ver")
        else:
            WaveformNames = list(f[groupname])
        if not CoordRadii:
            # If the list of Radii is empty, figure out what they are
            CoordRadii = [
                m.group("r") for Name in WaveformNames for m in [re_compile(r"""R(?P<r>.*?)\.dir""").search(Name)] if m
            ]
        else:
            # Pare down the WaveformNames list appropriately
            if type(CoordRadii[0]) == int:
                WaveformNames = [WaveformNames[i] for i in CoordRadii]
                CoordRadii = [
                    m.group("r") for Name in CoordRadii
                    for m in
                    [ re_compile(r"""R(?P<r>.*?)\.dir""").search(Name)] if m ]
            else:
                WaveformNames = [
                    Name for Name in WaveformNames for Radius in
                    CoordRadii for m in [re_compile(Radius).search(Name)] if m]
        NWaveforms = len(WaveformNames)
        
        # Check input data for NRAR format
        if groupname is None:
            if not validate_group_of_waveforms(f, filename, WaveformNames):
                raise ValueError(f"Bad input waveforms in {filename}.")
            stdout.write(f"{filename} passed the data-integrity tests.\n")
            stdout.flush()
        Ws = [scri.WaveformModes() for i in range(NWaveforms)]
        Radii = [None] * NWaveforms

    finally:
        f.close()

    PrintedLine = ""
    for n in range(NWaveforms):
        if n == NWaveforms - 1:
            WaveformNameString = WaveformNames[n] + "\n"
        else:
            WaveformNameString = WaveformNames[n] + ", "
        if len(PrintedLine + WaveformNameString) > 100:
            stdout.write("\n" + WaveformNameString)
            stdout.flush()
            PrintedLine = WaveformNameString
        else:
            stdout.write(WaveformNameString)
            stdout.flush()
            PrintedLine += WaveformNameString
        Ws[n], Radii[n] = read_finite_radius_waveform(filename,groupname,
                                                      WaveformNames[n],
                                                      ChMass)
    return Ws, Radii, CoordRadii

def set_common_time(Ws, Radii, MinTimeStep=0.005, EarliestTime=-3e300, LatestTime=3e300):
    """Interpolate Waveforms and radius data to a common set of times."""
    from scipy import interpolate

    NWaveforms = len(Radii)
    TLimits = [EarliestTime, LatestTime]
    # Get the new time data before any interpolations
    T = intersection(TLimits, Ws[0].t, MinTimeStep, EarliestTime, LatestTime)
    for i_W in range(1, NWaveforms):
        T = intersection(T, Ws[i_W].t)
    # Interpolate Radii and then Ws (in that order!)
    for i_W in range(NWaveforms):
        Radii[i_W] = interpolate.InterpolatedUnivariateSpline(Ws[i_W].t, Radii[i_W])(T)
        Ws[i_W] = Ws[i_W].interpolate(T)
    return


def extrapolate(**kwargs):
    """Perform extrapolations from finite-radius data.

    See arXiv:2010.15200 for specific details.

    Parameters
    ----------
      InputDirectory : str, (Default: './')
        Where to find the input data.  Can be relative or absolute.

      OutputDirectory : str, (Default: './')
        This directory will be made if it does not exist.

      DataFile : str, (Default: 'rh_FiniteRadii_CodeUnits.h5')
        Input file holding the data from all the radii.

      ChMass : float, (Default: 0.0)
        Christodoulou mass in the same units as the rest of the
        data.  All the data will be rescaled into units such that
        this is one.  If this is zero, the Christodoulou mass will
        be extracted automatically from the horizons file below.

      HorizonsFile : str, (Default: 'Horizons.h5')
        File name to read for horizon data (if ChMass is 0.0).

      CoordRadii : list of str or list of int, (Default: [])
        List of strings containing the radii to use, or of (integer)
        indices of the list of waveform names.  If this is a list of
        indices, the order is just the order output by the command
        `list(h5py.File(DataFile))` which *should* be the same as
        `h5ls`.  If the list is empty, all radii that can be found
        are used.

      ExtrapolationOrders : list of int, (Default: [-1, 2, 3, 4, 5, 6])
        Negative numbers correspond to extracted data, counting down
        from the outermost extraction radius (which is -1).

      UseOmega : bool, (Default: False)
        Whether or not to extrapolate as a function of lambda/r =
        1/(r*m*omega), where omega is the instantaneous angular
        frequency of rotation.  If this is True, the extrapolation
        will usually not converge for high N; if this is False, SVD
        will generally cause the convergence to appear to fall to
        roundoff, though the accuracy presumably is not so great.

      OutputFrame : {scri.Inertial, scri.Corotating}
        Transform to this frame before comparison and output.

      ExtrapolatedFiles : str, (Default: 'Extrapolated_N{N}.h5')
      DifferenceFiles   : str, (Default: 'ExtrapConvergence_N{N}-N{Nm1}.h5')
        These are python-formatted output file names, where the
        extrapolation order N is substituted for '{N}', and the
        previous extrapolation order is substituted for '{Nm1}'.
        The data-type inferred from the DataFile name is prepended.
        If DifferenceFiles is empty, the corresponding file is not
        output.

      UseStupidNRARFormat : bool, (Default: False)
        If True (and `ExtrapolatedFiles` does not end in '.dat'),
        then the h5 output format will be that stupid, old
        NRAR/NINJA format that doesn't convey enough information,
        is slow, and uses 33% more space than it needs to.  But you
        know, if you're into that kind of thing, whatever.  Who am
        I to judge?

      PlotFormat : str, (Default: '')
        The format of output plots.  This can be the empty string,
        in which case no plotting is done.  Or, these can be any of
        the formats supported by your installation of matplotlib.

      MinTimeStep : float, (Default: 0.005)
        The smallest allowed time step in the output data.

      EarliestTime : float, (Default: -3.0e300)
        The earliest time in the output data.  For values less than
        0, some of the data corresponds to times when only junk
        radiation is present.

      LatestTime : float, (Default: 3.0e300)
        The latest time in the output data.

      AlignmentTime : float, (Default: None)
        The time at which to align the Waveform with the dominant
        eigenvector of <LL>.  If the input value is `None` or is
        outside of the input data, it will be reset to the midpoint
        of the waveform: (W_outer.T(0)+W_outer.T(-1))/2

      NoiseFloor : float, (Default: None)
        ONLY USED FOR Psi1 AND Psi0 WAVEFORMS.
        The effective noise floor of the SpEC simulation.
        If Psi1 or Psi0 (NOT scaled by radius) falls beneath
        this value for certain extraction radii, then those
        radii are not included in the extrapolation. The value
        1.0e-9 seems to work well for a few BBH systems.

    """

    # Basic imports
    from os import makedirs, remove
    from os.path import exists, basename, dirname
    from sys import stdout, stderr
    from textwrap import dedent
    from numpy import sqrt, abs, fmod, pi, transpose, array
    from scipy.interpolate import splev, splrep
    import scri
    from scri import Inertial, Corotating, WaveformModes

    # Process keyword arguments
    InputDirectory = kwargs.pop("InputDirectory", "./")
    OutputDirectory = kwargs.pop("OutputDirectory", "./")
    DataFile = kwargs.pop("DataFile", "rh_FiniteRadii_CodeUnits.h5")
    ChMass = kwargs.pop("ChMass", 0.0)
    HorizonsFile = kwargs.pop("HorizonsFile", "Horizons.h5")
    CoordRadii = kwargs.pop("CoordRadii", [])
    ExtrapolationOrders = kwargs.pop("ExtrapolationOrders", [-1, 2, 3, 4, 5, 6])
    UseOmega = kwargs.pop("UseOmega", False)
    OutputFrame = kwargs.pop("OutputFrame", Inertial)
    ExtrapolatedFiles = kwargs.pop("ExtrapolatedFiles", "Extrapolated_N{N}.h5")
    DifferenceFiles = kwargs.pop("DifferenceFiles", "ExtrapConvergence_N{N}-N{Nm1}.h5")
    UseStupidNRARFormat = kwargs.pop("UseStupidNRARFormat", False)
    PlotFormat = kwargs.pop("PlotFormat", "")
    MinTimeStep = kwargs.pop("MinTimeStep", 0.005)
    EarliestTime = kwargs.pop("EarliestTime", -3.0e300)
    LatestTime = kwargs.pop("LatestTime", 3.0e300)
    AlignmentTime = kwargs.pop("AlignmentTime", None)
    NoiseFloor = kwargs.pop("NoiseFloor", None)
    return_finite_radius_waveforms = kwargs.pop("return_finite_radius_waveforms", False)
    if len(kwargs) > 0:
        raise ValueError(f"Unknown arguments to `extrapolate`: kwargs={kwargs}")

    # Polish up the input arguments
    if not InputDirectory.endswith("/"):
        InputDirectory += "/"
    if not OutputDirectory.endswith("/"):
        OutputDirectory += "/"
    if not exists(HorizonsFile):
        HorizonsFile = InputDirectory + HorizonsFile
    if not exists(DataFile):
        DataFile = InputDirectory + DataFile
    if ChMass == 0.0:
        print("WARNING: ChMass is being automatically determined from the data, " + "rather than metadata.txt.")
        ChMass = pick_Ch_mass(HorizonsFile)

    # AlignmentTime is reset properly once the data are read in, if necessary.
    # The reasonableness of ExtrapolationOrder is checked below.

    # Don't bother loading plotting modules unless we're plotting
    if PlotFormat:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt

        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
            "color", ["#000000", "#cc79a7", "#d55e00", "#0072b2", "#f0e442", "#56b4e9", "#e69f00", "#2b9f78",],
        )
        figabs = plt.figure(0)
        figarg = plt.figure(1)
        fignorm = plt.figure(2)

    # Read in the Waveforms
    print(f"Reading Waveforms from {DataFile}...")
    stdout.flush()
    Ws, Radii, CoordRadii = read_finite_radius_data(
        ChMass=ChMass, filename=DataFile, CoordRadii=CoordRadii)

    Radii_shape = (len(Radii), len(Radii[0]))

    # Make sure there are enough radii to do the requested extrapolations
    if (len(Ws) <= max(ExtrapolationOrders)) and (max(ExtrapolationOrders) > -1):
        raise ValueError(
            "Not enough data sets ({}) for max extrapolation order (N={}).".format(len(Ws), max(ExtrapolationOrders))
        )
    if -len(Ws) > min(ExtrapolationOrders):
        raise ValueError(
            "Not enough data sets ({}) for min extrapolation order (N={}).".format(len(Ws), min(ExtrapolationOrders))
        )

    # Figure out which is the outermost data
    SortedRadiiIndices = sorted(range(len(CoordRadii)), key=lambda k: float(CoordRadii[k]))
    i_outer = SortedRadiiIndices[-1]

    # Interpolate to common times
    print("Interpolating to common times...")
    stdout.flush()
    set_common_time(Ws, Radii, MinTimeStep, EarliestTime, LatestTime)
    W_outer = Ws[i_outer]

    # If the AlignmentTime is not set properly, set it to the default
    if (not AlignmentTime) or AlignmentTime < W_outer.t[0] or AlignmentTime >= W_outer.t[-1]:
        AlignmentTime = (W_outer.t[0] + W_outer.t[-1]) / 2.0

    # Print the input arguments neatly for the history
    InputArguments = """\
        # Extrapolation input arguments:
        D = {{}}
        D['InputDirectory'] = {InputDirectory}
        D['OutputDirectory'] = {OutputDirectory}
        D['DataFile'] = {DataFile}
        D['ChMass'] = {ChMass}
        D['HorizonsFile'] = {HorizonsFile}
        D['CoordRadii'] = {CoordRadii}
        D['ExtrapolationOrders'] = {ExtrapolationOrders}
        D['UseOmega'] = {UseOmega}
        D['OutputFrame'] = {OutputFrame}
        D['ExtrapolatedFiles'] = {ExtrapolatedFiles}
        D['DifferenceFiles'] = {DifferenceFiles}
        D['UseStupidNRARFormat'] = {UseStupidNRARFormat}
        D['PlotFormat'] = {PlotFormat}
        D['MinTimeStep'] = {MinTimeStep}
        D['EarliestTime'] = {EarliestTime}
        D['LatestTime'] = {LatestTime}
        D['AlignmentTime'] = {AlignmentTime}
        D['NoiseFloor'] = {NoiseFloor}
        # End Extrapolation input arguments
        """.format(
        InputDirectory=InputDirectory,
        OutputDirectory=OutputDirectory,
        DataFile=DataFile,
        ChMass=ChMass,
        HorizonsFile=HorizonsFile,
        CoordRadii=CoordRadii,
        ExtrapolationOrders=ExtrapolationOrders,
        UseOmega=UseOmega,
        OutputFrame=OutputFrame,
        ExtrapolatedFiles=ExtrapolatedFiles,
        DifferenceFiles=DifferenceFiles,
        UseStupidNRARFormat=UseStupidNRARFormat,
        PlotFormat=PlotFormat,
        MinTimeStep=MinTimeStep,
        EarliestTime=EarliestTime,
        LatestTime=LatestTime,
        AlignmentTime=AlignmentTime,
        NoiseFloor=NoiseFloor,
    )
    InputArguments = dedent(InputArguments)

    # If required, figure out the orbital frequencies
    if UseOmega:
        Omegas = [sqrt(sum([c ** 2 for c in o])) for o in W_outer.AngularVelocityVectorRelativeToInertial([2])]
    else:
        Omegas = []

    # Transform W_outer into its smoothed corotating frame, and align modes with
    # frame at given instant
    stdout.write("Rotating into common (outer) frame...\n")
    stdout.flush()
    if W_outer.frameType != Inertial:
        raise ValueError("Extrapolation assumes that the input data are in the inertial frame")
    print("Using alignment region (0.1, 0.8)")
    W_outer.to_corotating_frame(z_alignment_region=(0.1, 0.8))
    # W_outer.to_corotating_frame()
    # W_outer.align_decomposition_frame_to_modes(AlignmentTime)

    # Transform everyone else into the same frame
    for i in SortedRadiiIndices[:-1]:
        Ws[i].rotate_decomposition_basis(W_outer.frame)
        Ws[i].frameType = Corotating

    # Remove old h5 file if necessary
    if not ExtrapolatedFiles.endswith(".dat") and UseStupidNRARFormat:
        h5Index = ExtrapolatedFiles.find(".h5/")
        if h5Index > 0:
            if exists(ExtrapolatedFiles[: h5Index + 3]):
                remove(ExtrapolatedFiles[: h5Index + 3])

    # Do the actual extrapolations
    print("Running extrapolations.")
    stdout.flush()
    # Ws = Waveforms(_vectorW(Ws))
    # Ws.CommonTimeIsSet()
    # print([i for i in range(1)]); stdout.flush()
    # ExtrapolatedWaveformsObject = Ws.extrapolate(Radii, ExtrapolationOrders, Omegas)
    # print(type(ExtrapolatedWaveformsObject))
    # print([10])
    # for i in range(10):
    #     print("Yep"); stdout.flush()
    # print([i for i in range(1)]); stdout.flush()
    # ExtrapolatedWaveforms = [ExtrapolatedWaveformsObject.GetWaveform(i)
    #                         for i in range(ExtrapolatedWaveformsObject.size())]
    ExtrapolatedWaveforms = _Extrapolate(Ws, Radii, ExtrapolationOrders, Omegas, NoiseFloor)

    NExtrapolations = len(ExtrapolationOrders)
    for i, ExtrapolationOrder in enumerate(ExtrapolationOrders):
        # If necessary, rotate
        if OutputFrame == Inertial or OutputFrame == Corotating:
            stdout.write(f"N={ExtrapolationOrder}: Rotating into inertial frame... ")
            stdout.flush()
            ExtrapolatedWaveforms[i].to_inertial_frame()
            print("☺")
            stdout.flush()
        if OutputFrame == Corotating:
            stdout.write(f"N={ExtrapolationOrder}: Rotating into corotating frame... ")
            stdout.flush()
            ExtrapolatedWaveforms[i].to_corotating_frame()
            print("☺")
            stdout.flush()

        # Append the relevant information to the history
        ExtrapolatedWaveforms[i]._append_history(str(InputArguments))

        # Output the data
        ExtrapolatedFile = OutputDirectory + ExtrapolatedFiles.format(N=ExtrapolationOrder)
        stdout.write(f"N={ExtrapolationOrder}: Writing {ExtrapolatedFile}... ")
        stdout.flush()
        if not exists(OutputDirectory):
            makedirs(OutputDirectory)
        if ExtrapolatedFile.endswith(".dat"):
            ExtrapolatedWaveforms[i].Output(
                dirname(ExtrapolatedFile)
                + "/"
                + ExtrapolatedWaveforms[i].descriptor_string
                + "_"
                + ExtrapolatedWaveforms[i].frame_type_string
                + "_"
                + basename(ExtrapolatedFile)
            )

        else:
            from scri.SpEC import write_to_h5

            if i == 0:
                file_write_mode = "w"
            else:
                file_write_mode = "a"
            write_to_h5(
                ExtrapolatedWaveforms[i],
                ExtrapolatedFile,
                file_write_mode=file_write_mode,
                use_NRAR_format=UseStupidNRARFormat,
            )
        print("☺")
        stdout.flush()

    MaxNormTime = ExtrapolatedWaveforms[0].max_norm_time()
    FileNamePrefixString = (
        ExtrapolatedWaveforms[0].descriptor_string + "_" + ExtrapolatedWaveforms[0].frame_type_string + "_"
    )
    if PlotFormat:
        figabs.gca().set_xlabel(r"$(t-r_\ast)/M$")
        figarg.gca().set_xlabel(r"$(t-r_\ast)/M$")
        fignorm.gca().set_xlabel(r"$(t-r_\ast)/M$")
        figabs.gca().set_ylabel(
            r"$\Delta\, \mathrm{abs} \left( " + ExtrapolatedWaveforms[0].data_type_latex + r" \right) $"
        )
        figarg.gca().set_ylabel(
            r"$\Delta\, \mathrm{uarg} \left( " + ExtrapolatedWaveforms[0].data_type_latex + r" \right) $"
        )
        fignorm.gca().set_ylabel(
            r"$\left\| \Delta\, " + ExtrapolatedWaveforms[0].data_type_latex + r" \right\|_{L_2} $"
        )

    for i, ExtrapolationOrder in reversed(list(enumerate(ExtrapolationOrders))):
        if i > 0:  # Compare to the last one
            if DifferenceFiles or PlotFormat:
                Diff = scri.WaveformModes(ExtrapolatedWaveforms[i].compare(ExtrapolatedWaveforms[i - 1]))
            if DifferenceFiles:
                DifferenceFile = OutputDirectory + DifferenceFiles.format(
                    N=ExtrapolationOrder, Nm1=ExtrapolationOrders[i - 1]
                )
                stdout.write(f"N={ExtrapolationOrder}: Writing {DifferenceFile}... ")
                stdout.flush()
                if DifferenceFile.endswith(".dat"):
                    Diff.Output(
                        dirname(DifferenceFile)
                        + "/"
                        + Diff.descriptor_string
                        + "_"
                        + Diff.frame_type_string
                        + "_"
                        + basename(DifferenceFile)
                    )
                else:
                    from scri.SpEC import write_to_h5

                    write_to_h5(Diff, DifferenceFile, use_NRAR_format=UseStupidNRARFormat)
                print("☺")
                stdout.flush()
            if PlotFormat:
                # stdout.write("Plotting... "); stdout.flush()
                Interpolated = scri.WaveformModes(ExtrapolatedWaveforms[i].interpolate(Diff.t))
                Normalization = Interpolated.norm(True)
                rep_A = splrep(
                    ExtrapolatedWaveforms[i].t,
                    ExtrapolatedWaveforms[i].abs[:, ExtrapolatedWaveforms[i].index(2, 2)],
                    s=0,
                )
                rep_B = splrep(
                    ExtrapolatedWaveforms[i - 1].t,
                    ExtrapolatedWaveforms[i - 1].abs[:, ExtrapolatedWaveforms[i - 1].index(2, 2)],
                    s=0,
                )
                AbsA = splev(Diff.t, rep_A, der=0)
                AbsB = splev(Diff.t, rep_B, der=0)
                AbsDiff = abs(AbsA - AbsB) / AbsA
                rep_arg_A = splrep(
                    ExtrapolatedWaveforms[i].t,
                    ExtrapolatedWaveforms[i].arg_unwrapped[:, ExtrapolatedWaveforms[i].index(2, 2)],
                    s=0,
                )
                rep_arg_B = splrep(
                    ExtrapolatedWaveforms[i].t,
                    ExtrapolatedWaveforms[i - 1].arg_unwrapped[:, ExtrapolatedWaveforms[i - 1].index(2, 2)],
                    s=0,
                )
                ArgDiff = splev(Diff.t, rep_arg_A, der=0) - splev(Diff.t, rep_arg_B, der=0)

                if abs(ArgDiff[len(ArgDiff) // 3]) > 1.9 * pi:
                    ArgDiff -= 2 * pi * round(ArgDiff[len(ArgDiff) // 3] / (2 * pi))
                plt.figure(0)
                plt.semilogy(
                    Diff.t,
                    AbsDiff,
                    label=r"$(N={}) - (N={})$".format(ExtrapolationOrder, ExtrapolationOrders[i - 1]),
                )
                plt.figure(1)
                plt.semilogy(
                    Diff.t,
                    abs(ArgDiff),
                    label=r"$(N={}) - (N={})$".format(ExtrapolationOrder, ExtrapolationOrders[i - 1]),
                )
                plt.figure(2)
                plt.semilogy(
                    Diff.t,
                    Diff.norm(True) / Normalization,
                    label=r"$(N={}) - (N={})$".format(ExtrapolationOrder, ExtrapolationOrders[i - 1]),
                )
                # print("☺"); stdout.flush()

    # Finish up the plots and save
    if PlotFormat:
        stdout.write("Saving plots... ")
        stdout.flush()
        plt.figure(0)
        plt.legend(
            borderpad=0.2,
            labelspacing=0.1,
            handlelength=1.5,
            handletextpad=0.1,
            loc="lower left",
            prop={"size": "small"},
        )
        plt.gca().set_ylim(1e-8, 10)
        plt.gca().axvline(x=MaxNormTime, ls="--")
        try:
            from matplotlib.pyplot import tight_layout

            tight_layout(pad=0.5)
        except:
            pass
        figabs.savefig("{}/{}ExtrapConvergence_Abs.{}".format(OutputDirectory, FileNamePrefixString, PlotFormat))
        if PlotFormat != "png":
            figabs.savefig("{}/{}ExtrapConvergence_Abs.{}".format(OutputDirectory, FileNamePrefixString, "png"))
        plt.gca().set_xlim(MaxNormTime - 500.0, MaxNormTime + 200.0)
        figabs.savefig("{}/{}ExtrapConvergence_Abs_Merger.{}".format(OutputDirectory, FileNamePrefixString, PlotFormat))
        if PlotFormat != "png":
            figabs.savefig("{}/{}ExtrapConvergence_Abs_Merger.{}".format(OutputDirectory, FileNamePrefixString, "png"))
        plt.close(figabs)
        plt.figure(1)
        plt.legend(
            borderpad=0.2,
            labelspacing=0.1,
            handlelength=1.5,
            handletextpad=0.1,
            loc="lower left",
            prop={"size": "small"},
        )
        plt.gca().set_xlabel("")
        plt.gca().set_ylim(1e-8, 10)
        plt.gca().axvline(x=MaxNormTime, ls="--")
        try:
            tight_layout(pad=0.5)
        except:
            pass
        figarg.savefig("{}/{}ExtrapConvergence_Arg.{}".format(OutputDirectory, FileNamePrefixString, PlotFormat))
        if PlotFormat != "png":
            figarg.savefig("{}/{}ExtrapConvergence_Arg.{}".format(OutputDirectory, FileNamePrefixString, "png"))
        plt.gca().set_xlim(MaxNormTime - 500.0, MaxNormTime + 200.0)
        figarg.savefig("{}/{}ExtrapConvergence_Arg_Merger.{}".format(OutputDirectory, FileNamePrefixString, PlotFormat))
        if PlotFormat != "png":
            figarg.savefig("{}/{}ExtrapConvergence_Arg_Merger.{}".format(OutputDirectory, FileNamePrefixString, "png"))
        plt.close(figarg)
        plt.figure(2)
        plt.legend(
            borderpad=0.2,
            labelspacing=0.1,
            handlelength=1.5,
            handletextpad=0.1,
            loc="lower left",
            prop={"size": "small"},
        )
        plt.gca().set_ylim(1e-6, 10)
        plt.gca().axvline(x=MaxNormTime, ls="--")
        try:
            tight_layout(pad=0.5)
        except:
            pass
        fignorm.savefig("{}/{}ExtrapConvergence_Norm.{}".format(OutputDirectory, FileNamePrefixString, PlotFormat))
        if PlotFormat != "png":
            fignorm.savefig("{}/{}ExtrapConvergence_Norm.{}".format(OutputDirectory, FileNamePrefixString, "png"))
        plt.gca().set_xlim(MaxNormTime - 500.0, MaxNormTime + 200.0)
        fignorm.savefig(
            "{}/{}ExtrapConvergence_Norm_Merger.{}".format(OutputDirectory, FileNamePrefixString, PlotFormat)
        )
        if PlotFormat != "png":
            fignorm.savefig(
                "{}/{}ExtrapConvergence_Norm_Merger.{}".format(OutputDirectory, FileNamePrefixString, "png")
            )
        plt.close(fignorm)
        print("☺")
        stdout.flush()

    if return_finite_radius_waveforms:
        return ExtrapolatedWaveforms, Ws
    return ExtrapolatedWaveforms


#####################################
### Batch extrapolation utilities ###
#####################################


# Local utility function
def _safe_format(s, **keys):
    """Like str.format, but doesn't mind missing arguments.

    This function is used to replace strings like '{SomeKey}' in
    the template with the arguments given as keys.  For example,

      _safe_format('{SomeKey} {SomeOtherKey}', SomeKey='Hello', SomeMissingKey='Bla')

    returns 'Hello {SomeOtherKey}', without errors, ignoring the
    `SomeMissingKey` argument, and not bothering with
    '{SomeOtherKey}', so that that can be replaced later.

    """

    class Default(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    from string import Formatter

    return Formatter().vformat(s, (), Default(keys))


def UnstartedExtrapolations(TopLevelOutputDir, SubdirectoriesAndDataFiles):
    """Find unstarted extrapolation directories."""
    from os.path import exists

    Unstarted = []
    for Subdirectory, DataFile in SubdirectoriesAndDataFiles:
        StartedFile = "{}/{}/.started_{}".format(TopLevelOutputDir, Subdirectory, DataFile)
        if not exists(StartedFile):
            Unstarted.append([Subdirectory, DataFile])
    return Unstarted


def NewerDataThanExtrapolation(TopLevelInputDir, TopLevelOutputDir, SubdirectoriesAndDataFiles):
    """Find newer data than extrapolation."""
    from os.path import exists, getmtime

    Newer = []
    for Subdirectory, DataFile in SubdirectoriesAndDataFiles:
        FinishedFile = "{}/{}/.finished_{}".format(TopLevelOutputDir, Subdirectory, DataFile)
        if exists(FinishedFile):
            TimeFinished = getmtime(FinishedFile)
            Timemetadata = getmtime(f"{TopLevelInputDir}/{Subdirectory}/metadata.txt")
            TimeData = getmtime(f"{TopLevelInputDir}/{Subdirectory}/{DataFile}")
            if TimeData > TimeFinished or Timemetadata > TimeFinished:
                Newer.append([Subdirectory, DataFile])
    return Newer


def StartedButUnfinishedExtrapolations(TopLevelOutputDir, SubdirectoriesAndDataFiles):
    """Find directories with extrapolations that started but didn't finish."""
    from os.path import exists

    Unfinished = []
    for Subdirectory, DataFile in SubdirectoriesAndDataFiles:
        StartedFile = "{}/{}/.started_{}".format(TopLevelOutputDir, Subdirectory, DataFile)
        ErrorFile = "{}/{}/.error_{}".format(TopLevelOutputDir, Subdirectory, DataFile)
        FinishedFile = "{}/{}/.finished_{}".format(TopLevelOutputDir, Subdirectory, DataFile)
        if exists(StartedFile) and not exists(ErrorFile) and not exists(FinishedFile):
            Unfinished.append([Subdirectory, DataFile])
    return Unfinished


def ErroredExtrapolations(TopLevelOutputDir, SubdirectoriesAndDataFiles):
    """Find directories with errors."""
    from os.path import exists

    Errored = []
    for Subdirectory, DataFile in SubdirectoriesAndDataFiles:
        ErrorFile = "{}/{}/.error_{}".format(TopLevelOutputDir, Subdirectory, DataFile)
        if exists(ErrorFile):
            Errored.append([Subdirectory, DataFile])
    return Errored


def FindPossibleExtrapolationsToRun(TopLevelInputDir):
    """Find all possible extrapolations."""
    from os import walk
    from re import compile as re_compile

    SubdirectoriesAndDataFiles = []
    LevPattern = re_compile(r"/Lev[0-9]*$")

    # Walk the input directory
    for step in walk(TopLevelInputDir, followlinks=True):
        if LevPattern.search(step[0]):
            if "metadata.txt" in step[2]:
                if "rh_FiniteRadii_CodeUnits.h5" in step[2]:
                    SubdirectoriesAndDataFiles.append(
                        [step[0].replace(TopLevelInputDir + "/", ""), "rh_FiniteRadii_CodeUnits.h5",]
                    )
                if "rPsi4_FiniteRadii_CodeUnits.h5" in step[2]:
                    SubdirectoriesAndDataFiles.append(
                        [step[0].replace(TopLevelInputDir + "/", ""), "rPsi4_FiniteRadii_CodeUnits.h5",]
                    )
                if "r2Psi3_FiniteRadii_CodeUnits.h5" in step[2]:
                    SubdirectoriesAndDataFiles.append(
                        [step[0].replace(TopLevelInputDir + "/", ""), "r2Psi3_FiniteRadii_CodeUnits.h5",]
                    )
                if "r3Psi2_FiniteRadii_CodeUnits.h5" in step[2]:
                    SubdirectoriesAndDataFiles.append(
                        [step[0].replace(TopLevelInputDir + "/", ""), "r3Psi2_FiniteRadii_CodeUnits.h5",]
                    )
                if "r4Psi1_FiniteRadii_CodeUnits.h5" in step[2]:
                    SubdirectoriesAndDataFiles.append(
                        [step[0].replace(TopLevelInputDir + "/", ""), "r4Psi1_FiniteRadii_CodeUnits.h5",]
                    )
                if "r5Psi0_FiniteRadii_CodeUnits.h5" in step[2]:
                    SubdirectoriesAndDataFiles.append(
                        [step[0].replace(TopLevelInputDir + "/", ""), "r5Psi0_FiniteRadii_CodeUnits.h5",]
                    )
    return SubdirectoriesAndDataFiles


def RunExtrapolation(TopLevelInputDir, TopLevelOutputDir, Subdirectory, DataFile, Template):
    from os import makedirs, chdir, getcwd, utime, remove
    from os.path import exists
    from subprocess import call

    InputDir = f"{TopLevelInputDir}/{Subdirectory}"
    OutputDir = f"{TopLevelOutputDir}/{Subdirectory}"
    if not exists(OutputDir):
        makedirs(OutputDir)

    # If OutputDir/.started_r...h5 doesn't exist, touch it; remove errors and
    # finished reports
    with open(f"{OutputDir}/.started_{DataFile}", "a") as f:
        pass
    utime(f"{OutputDir}/.started_{DataFile}", None)
    if exists(f"{OutputDir}/.error_{DataFile}"):
        remove(f"{OutputDir}/.error_{DataFile}")
    if exists(f"{OutputDir}/.finished_{DataFile}"):
        remove(f"{OutputDir}/.finished_{DataFile}")

    # Copy the template file to OutputDir
    with open("{}/Extrapolate_{}.py".format(OutputDir, DataFile[:-3]), "w") as TemplateFile:
        TemplateFile.write(_safe_format(Template, DataFile=DataFile, Subdirectory=Subdirectory))

    # Try to run the extrapolation
    OriginalDir = getcwd()
    try:
        try:
            chdir(OutputDir)
        except:
            print(f"Couldn't change directory to '{OutputDir}'.")
            raise
        print("\n\nRunning {1}/Extrapolate_{0}.py\n\n".format(DataFile[:-3], getcwd()))
        ReturnValue = call(
            "set -o pipefail; python Extrapolate_{0}.py 2>&1 | " + "tee Extrapolate_{}.log".format(DataFile[:-3]),
            shell=True,
        )
        if ReturnValue:
            print(
                "\n\nRunExtrapolation got an error ({4}) on "
                + "['{}', '{}', '{}', '{}'].\n\n".format(
                    TopLevelInputDir,
                    TopLevelOutputDir,
                    Subdirectory,
                    DataFile,
                    ReturnValue,
                )
            )
            with open(f"{OutputDir}/.error_{DataFile}", "w"):
                pass
            chdir(OriginalDir)
            return ReturnValue
        with open(f"{OutputDir}/.finished_{DataFile}", "w"):
            pass
        print("\n\nFinished Extrapolate_{}.py in {}\n\n".format(DataFile[:-3], getcwd()))
    except:
        with open(f"{OutputDir}/.error_{DataFile}", "w"):
            pass
        print(
            "\n\nRunExtrapolation got an error on "
            + "['{}', '{}', '{}', '{}'].\n\n".format(TopLevelInputDir, TopLevelOutputDir, Subdirectory, DataFile)
        )
    finally:
        chdir(OriginalDir)

    return 0


def _Extrapolate(FiniteRadiusWaveforms, Radii, ExtrapolationOrders, Omegas=None, NoiseFloor=None):
    import numpy
    import scri
    from tqdm import trange

    # Get the various dimensions, etc.
    MaxN = max(ExtrapolationOrders)
    MinN = min(ExtrapolationOrders)
    UseOmegas = (Omegas is not None) and (len(Omegas) != 0)
    NTimes = FiniteRadiusWaveforms[0].n_times
    NModes = FiniteRadiusWaveforms[0].n_modes
    NFiniteRadii = len(FiniteRadiusWaveforms)
    NExtrapolations = len(ExtrapolationOrders)
    #SVDTol = 1.0e-12  # Same as Numerical Recipes default in fitsvd.h
    DataType = FiniteRadiusWaveforms[NFiniteRadii - 1].dataType
    ExcludeInsignificantRadii = DataType in [scri.psi1, scri.psi0] and bool(NoiseFloor)
    if ExcludeInsignificantRadii:
        from spherical_functions import LM_index

        ell_min = FiniteRadiusWaveforms[NFiniteRadii - 1].ell_min
        print("Performing extrapolation excluding insignifcant outer radii.")

    # Make sure everyone is playing with a full deck
    if abs(MinN) > NFiniteRadii:
        print(
            """ERROR: Asking for finite-radius waveform {0}, but only got """
            + """{1} finite-radius Waveform objects; need at least as {2} """
            + """finite-radius waveforms.

        """.format(
                MinN, NFiniteRadii, abs(MinN)
            )
        )
        raise ValueError("scri_IndexOutOfBounds")
    if MaxN > 0 and MaxN >= NFiniteRadii:
        print(
            "ERROR: Asking for extrapolation up to order {}, but only got {} "
            "finite-radius Waveform objects; need at least {} waveforms.".format(MaxN, NFiniteRadii, MaxN + 1)
        )
        raise ValueError("scri_IndexOutOfBounds")
    if len(Radii) != NFiniteRadii:
        print(
            """ERROR: Mismatch in data to be extrapolated; there are different """
            + """numbers of waveforms and radius vectors.
        len(FiniteRadiusWaveforms)={}; len(Radii)={}
        """.format(
                NFiniteRadii, len(Radii)
            )
        )
        raise ValueError("scri_VectorSizeMismatch")
    if UseOmegas and len(Omegas) != NTimes:
        print(
            "ERROR: NTimes mismatch in data to be extrapolated.\n"
            + f"       FiniteRadiusWaveforms[0].NTimes()={NTimes}\n"
            + "       Omegas.size()={}\n".format(len(Omegas))
            + "\n"
        )
        raise ValueError("scri_VectorSizeMismatch")
    for i_W in range(1, NFiniteRadii):
        if FiniteRadiusWaveforms[i_W].n_times != NTimes:
            print(
                "ERROR: NTimes mismatch in data to be extrapolated.\n"
                + f"       FiniteRadiusWaveforms[0].NTimes()={NTimes}\n"
                + "       FiniteRadiusWaveforms[{i_W}].NTimes()={0}\n".format(FiniteRadiusWaveforms[i_W].n_times)
                + "\n"
            )
            raise ValueError("scri_VectorSizeMismatch")
        if FiniteRadiusWaveforms[i_W].n_modes != NModes:
            print(
                "ERROR: NModes mismatch in data to be extrapolated.\n"
                + f"       FiniteRadiusWaveforms[0].NModes()={NModes}\n"
                + "       FiniteRadiusWaveforms[{i_W}].NModes()={0}\n".format(FiniteRadiusWaveforms[i_W].n_modes)
                + "\n"
            )
            raise ValueError("scri_VectorSizeMismatch")
        if len(Radii[i_W]) != NTimes:
            print(
                "ERROR: NTimes mismatch in data to be extrapolated.\n"
                + f"       FiniteRadiusWaveforms[0].NTimes()={NTimes}\n"
                + "       Radii[{}].size()={}\n".format(i_W, len(Radii[i_W]))
                + "\n"
            )
            raise ValueError("scri_VectorSizeMismatch")

    # Set up the containers that will be used to store the data during
    # extrapolation.  These are needed so that we don't have to make too many
    # calls to Waveform object methods, which have to go through the _Waveform
    # wrapper, and are therefore slow.
    data = numpy.array([W.data for W in FiniteRadiusWaveforms])
    data = data.view(dtype=float).reshape((data.shape[0], data.shape[1], data.shape[2], 2))
    extrapolated_data = numpy.empty((NExtrapolations, NTimes, NModes), dtype=complex)

    # Set up the output data, recording everything but the mode data
    ExtrapolatedWaveforms = [None] * NExtrapolations
    for i_N in range(NExtrapolations):
        N = ExtrapolationOrders[i_N]
        if N < 0:
            ExtrapolatedWaveforms[i_N] = scri.WaveformModes(FiniteRadiusWaveforms[NFiniteRadii + N])
            ExtrapolatedWaveforms[i_N].history += [f"### Extrapolating with N={N}\n"]
        else:
            # Do everything but set the data
            ExtrapolatedWaveforms[i_N] = scri.WaveformModes(
                t=FiniteRadiusWaveforms[NFiniteRadii - 1].t,
                frame=FiniteRadiusWaveforms[NFiniteRadii - 1].frame,
                data=np.zeros(
                    (NTimes, NModes),
                    dtype=FiniteRadiusWaveforms[NFiniteRadii - 1].data.dtype,
                ),
                history=FiniteRadiusWaveforms[NFiniteRadii - 1].history + [f"### Extrapolating with N={N}\n"],
                version_hist=FiniteRadiusWaveforms[NFiniteRadii - 1].version_hist,
                frameType=FiniteRadiusWaveforms[NFiniteRadii - 1].frameType,
                dataType=FiniteRadiusWaveforms[NFiniteRadii - 1].dataType,
                r_is_scaled_out=FiniteRadiusWaveforms[NFiniteRadii - 1].r_is_scaled_out,
                m_is_scaled_out=FiniteRadiusWaveforms[NFiniteRadii - 1].m_is_scaled_out,
                ell_min=FiniteRadiusWaveforms[NFiniteRadii - 1].ell_min,
                ell_max=FiniteRadiusWaveforms[NFiniteRadii - 1].ell_max,
            )

    if MaxN < 0:
        return ExtrapolatedWaveforms

    MaxCoefficients = MaxN + 1

    # Loop over time
    for i_t in trange(NTimes):

        # Set up the radius data (if we are NOT using Omega)
        if not UseOmegas:
            OneOverRadii = [1.0 / Radii[i_W][i_t] for i_W in range(NFiniteRadii)]

            data_t = data[:, i_t, :, 0] + 1j * data[:, i_t, :, 1]

            if ExcludeInsignificantRadii:
                # For the sake of avoiding a poorly conditioned polynomial fit, we need to set an absolute minimum for the
                # possible number of radii that may be used. Occasionally where there is junk radiation the amplitude of
                # the waveform may be spuriously small at a few timesteps, setting this minimum avoids excluding an
                # unreasonable number of radii and then having polyfit return an error.
                MinimumNRadii = max(ExtrapolationOrders) + 4

                # Since we are in the corotating frame, we can be sure that the (2,2) mode is dominant. For the sake of
                # consistency we will use the same radial weights for each mode at a given retarded time.
                WaveformAmplitude = np.linalg.norm(Re[MinimumNRadii] + 1j * Im[MinimumNRadii])

                # Radius at which the amplitude of the waveform is equal to error_tol.
                LargestSignificantRadius = (WaveformAmplitude / NoiseFloor) ** (1 / (6 - DataType))

                # Use as many radii as possible that are smaller than LargestSignificantRadius.
                RadiiOnTimeSlice = np.array(Radii)[:, i_t]
                MinimumNRadii = max(sum(RadiiOnTimeSlice <= LargestSignificantRadius), MinimumNRadii)

                # Remove the outer radii
                OneOverRadii = OneOverRadii[:MinimumNRadii]
                data_t = data_t[:MinimumNRadii, :]

            # Loop over extrapolation orders
            for i_N in range(NExtrapolations):
                N = ExtrapolationOrders[i_N]

                # If non-extrapolating, skip to the next one (the copying was
                # done when ExtrapolatedWaveforms[i_N] was constructed)
                if N < 0:
                    continue

                # Do the extrapolations
                extrapolated_data[i_N, i_t, :] = polyfit(OneOverRadii, data_t, N)[0, :]

        else:  # UseOmegas

            # Loop over modes
            for i_m in range(NModes):

                # Set up the radius data (if we ARE using Omega)
                M = FiniteRadiusWaveforms[0].LM(i_m)[1]
                if UseOmegas:
                    OneOverRadii = [
                        1.0 / (Radii[i_W][i_t] * M * Omegas[i_t]) if M != 0 else 1.0 / (Radii[i_W][i_t])
                        for i_W in range(NFiniteRadii)
                    ]

                # Set up the mode data
                Re = data[:, i_m, i_t, 0]
                Im = data[:, i_m, i_t, 1]

                # Loop over extrapolation orders
                for i_N in range(NExtrapolations):
                    N = ExtrapolationOrders[i_N]

                    # If non-extrapolating, skip to the next one (the copying was
                    # done when ExtrapolatedWaveforms[i_N] was constructed)
                    if N < 0:
                        continue

                    # Do the extrapolations
                    re = numpy.polyfit(OneOverRadii, Re, N)[-1]
                    im = numpy.polyfit(OneOverRadii, Im, N)[-1]
                    extrapolated_data[i_N, i_t, i_m] = re + 1j * im

    for i_N in range(NExtrapolations):
        N = ExtrapolationOrders[i_N]
        if N >= 0:
            ExtrapolatedWaveforms[i_N].data[...] = extrapolated_data[i_N]

    print("")

    return ExtrapolatedWaveforms
