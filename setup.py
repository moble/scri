#!/usr/bin/env python

# Copyright (c) 2019, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

from os import getenv

# Construct the version number from the date and time this python version was created.
from os import environ
from sys import platform

on_windows = "win" in platform.lower() and not "darwin" in platform.lower()
if "package_version" in environ:
    version = environ["package_version"]
    print(f"Setup.py using environment version='{version}'")
else:
    print("The variable 'package_version' was not present in the environment")
    try:
        # For cases where this is being installed from git.  This gives the true version number.
        from subprocess import check_output

        if on_windows:
            version = check_output("""git log -1 --format=%cd --date=format:'%Y.%m.%d.%H.%M.%S'""", shell=False)
            version = version.decode("ascii").strip().replace(".0", ".").replace("'", "")
        else:
            try:
                from subprocess import DEVNULL as devnull

                version = check_output(
                    """git log -1 --format=%cd --date=format:'%Y.%-m.%-d.%-H.%-M.%-S'""", shell=True, stderr=devnull
                )
            except AttributeError:
                from os import devnull

                version = check_output(
                    """git log -1 --format=%cd --date=format:'%Y.%-m.%-d.%-H.%-M.%-S'""", shell=True, stderr=devnull
                )
            version = version.decode("ascii").rstrip()
        print(f"Setup.py using git log version='{version}'")
    except Exception:
        # For cases where this isn't being installed from git.  This gives the wrong version number,
        # but at least it provides some information.
        # import traceback
        # print(traceback.format_exc())
        try:
            from time import strftime, gmtime

            try:
                version = strftime("%Y.%-m.%-d.%-H.%-M.%-S", gmtime())
            except ValueError:  # because Windows
                version = strftime("%Y.%m.%d.%H.%M.%S", gmtime()).replace(".0", ".")
            print(f"Setup.py using strftime version='{version}'")
        except:
            version = "0.0.0"
            print(f"Setup.py failed to determine the version; using '{version}'")
with open("scri/_version.py", "w") as f:
    f.write(f'__version__ = "{version}"')

long_description = """\
This package collects a number of functions for constructing and manipulating gravitational
waveforms, including rotating, determining the angular velocity, finding the co-precessing and
co-rotating frames, and applying boosts, translations, and supertranslations.
"""

if __name__ == "__main__":
    from setuptools import setup

    setup(
        name="scri",
        version=version,
        description="Manipulating time-dependent functions of spin-weighted spherical harmonics",
        long_description=long_description,
        url="https://github.com/moble/scri",
        author="Michael Boyle",
        author_email="mob22@cornell.edu",
        packages=[
            "scri",
            "scri.asymptotic_bondi_data",
            "scri.LVC",
            "scri.pn",
            "scri.SpEC",
            "scri.SpEC.file_io",
        ],
        zip_safe=False,
        install_requires=[
            "numpy>=1.13",
            "scipy>=0.18.0",
            "h5py",
            "numba>=0.49.1",
            "numpy-quaternion>=2019.7.15.12.50.36",
            "spinsfast",
            "spherical-functions>=2020.8.18.15.37.20",
            "sxs",
        ],
        python_requires=">=3.6",
    )
