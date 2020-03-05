# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import

import pytest
import os.path
import numpy as np
import h5py
import scri
import scri.SpEC
import shutil


# NOTE: test_file_io() depends on output from test_NRAR_extrapolation(), so the latter
# must come first in this file.

@pytest.fixture(scope="session")  # The following will exist for an entire run of pytest
def tempdir(tmp_path_factory):
    """Test fixture to create a temp directory and write the fake finite-radius file"""
    tmpdir = tmp_path_factory.mktemp("test_extrapolation")
    filename = tmpdir / "rh_FiniteRadii_CodeUnits.h5"
    scri.sample_waveforms.create_fake_finite_radius_strain_h5file(
        output_file_path=filename,
        ell_max=3,
        t_1=2000.0,
    )
    return tmpdir


def test_extrapolation(tempdir):
    """Test extrapolation of waveforms with output in the scri format"""
    input_dir = str(tempdir)
    output_dir = str(tempdir / "test_extrapolation")
    scri.extrapolation.extrapolate(InputDirectory=input_dir, OutputDirectory=output_dir, ChMass=1.0)


@pytest.mark.skipif(
    not os.path.exists(input_dir + filename),
    reason="requires {}".format(input_dir + filename),
)
def test_NRAR_extrapolation():
    """
    Test extrapolation of waveforms with output in the NRAR format.
    """
    output_dir = output_dir_base + "test_NRAR_extrapolation_output/"
    # Extrapolation fails if the output filenames exist so we delete
    # the output files from a previous test if they exist.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Computing DifferenceFiles and making plots are tested in test_extrapolation() so
    # to save time, we don't do it here.
    scri.extrapolation.extrapolate(
        InputDirectory=input_dir,
        OutputDirectory=output_dir,
        DifferenceFiles="",
        PlotFormat="",
        UseStupidNRARFormat=True,
        ChMass=1.0,
    )


def test_file_io(tempdir):
    """Test file I/O with a round-trip and compare H5 files"""
    input_file = tempdir / "test_NRAR_extrapolation" / "rhOverM_Extrapolated_N2.h5"
    output_dir = tempdir / "test_file_io"
    output_file = str(output_dir / "Asymptotic_GeometricUnits.h5")
    output_file_result = str(output_dir / "rhOverM_Asymptotic_GeometricUnits.h5")

    if not input_file.exists():
        scri.extrapolation.extrapolate(
            InputDirectory=str(tempdir),
            OutputDirectory=str(tempdir / "test_NRAR_extrapolation"),
            DifferenceFiles="",
            PlotFormat="",
            UseStupidNRARFormat=True,
            ChMass=1.0,
        )
    output_dir.mkdir()

    w = scri.SpEC.read_from_h5(input_file)
    scri.SpEC.write_to_h5(w, output_file)

    f1 = h5py.File(test_file,'r')
    f2 = h5py.File("../SpEC/samples/rhOverM_Asymptotic_GeometricUnits_test.h5",'r')

    # Check top-level attributes
    list1 = sorted(f1.attrs.items())
    list2 = sorted(f2.attrs.items())
    # attribute 3 is waveform output version
    list1.remove(list1[3])
    list2.remove(list2[3])
    assert list1 == list2

    # Check that original history is contained (with extra comment characters) in second history
    assert ("# " + "\n# ".join(f1["History.txt"][()].split("\n"))) in f2[
        "History.txt"
    ][()]

    # Now check each mode from the original
    for mode in list(f1):
        if mode.startswith("Y"):
            assert list(f1[mode].attrs.items()) == list(f2[mode].attrs.items())
            assert np.array_equal(f1[mode][:], f2[mode][:])
