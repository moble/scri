"""

Read output with

    import pandas as pd
    pd.read_csv("BMSTransformations.csv")

"""

import sys
from os import walk
from os.path import join, dirname
from fnmatch import filter
import numpy as np
import h5py
from scri.SpEC import estimate_avg_com_motion as eacm


def run_in(filename, i_this, i_tot, f):
    print(f"Working on {filename} -- {i_this} of {i_tot}")
    sys.stdout.flush()
    try:
        with h5py.File(filename, "r") as horizons:
            tf = horizons["AhA.dir/ChristodoulouMass.dat"][-1, 0]
        vals = np.hstack(eacm(filename))
        dir, lev = dirname(filename).split("Lev")
        csv = f"{dir}, {float(lev)}, {', '.join([str(f) for f in np.hstack(vals)])}, {tf}\n"
        f.write(csv)
    except:
        print(f"Failed in {filename} -- {i_this} of {i_tot}")
        sys.stdout.flush()
    print(f"Finished {filename} -- {i_this} of {i_tot}")
    sys.stdout.flush()


if __name__ == "__main__":
    print("Finding files to operate on")
    files = [
        join(root, "Horizons.h5")
        for top_dir in ["Catalog", "Incoming"]
        for root, dirnames, filenames in walk(top_dir)
        for filename in filter(filenames, "Horizons.h5")
    ]
    print(f"Finished finding {len(files)} files to operate on")
    with open("BMSTransformations.csv", "w") as f:
        f.write("dirname,Lev,x0,y0,z0,vx0,vy0,vz0,t0,tf\n")
        for i_this, filename in enumerate(files, 1):
            run_in(filename, i_this, len(files), f)
