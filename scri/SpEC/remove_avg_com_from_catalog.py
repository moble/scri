import sys
from os import walk
from os.path import join, exists
from fnmatch import filter
from multiprocessing import Pool
from scri.SpEC import remove_avg_com_motion as racm


def run_in(args):
    (filename, i_this, i_tot) = args
    if exists(filename.replace(".h5", "_CoM.h5")):
        return
    print(f"Working on {filename} -- {i_this} of {i_tot}")
    sys.stdout.flush()
    try:
        racm(f"{filename}/Extrapolated_N2.dir", plot=True, file_write_mode="w")
        try:
            racm(f"{filename}/Extrapolated_N3.dir", plot=True, file_write_mode="a")
        except:
            print(f"Failed in {filename}/Extrapolated_N3.dir -- {i_this} of {i_tot}")
            sys.stdout.flush()
        try:
            racm(f"{filename}/Extrapolated_N4.dir", plot=True, file_write_mode="a")
        except:
            print(f"Failed in {filename}/Extrapolated_N4.dir -- {i_this} of {i_tot}")
            sys.stdout.flush()
        try:
            racm(f"{filename}/OutermostExtraction.dir", plot=True, file_write_mode="a")
        except:
            print(f"Failed in {filename}/OutermostExtraction.dir -- {i_this} of {i_tot}")
            sys.stdout.flush()
    except:
        print(f"Failed in {filename} -- {i_this} of {i_tot}")
        sys.stdout.flush()
    print(f"Finished {filename} -- {i_this} of {i_tot}")
    sys.stdout.flush()


if __name__ == "__main__":
    print("Finding files to operate on")
    files = [
        join(root, "rhOverM_Asymptotic_GeometricUnits.h5")
        for top_dir in ["Catalog", "Incoming"]
        for root, dirnames, filenames in walk(top_dir)
        for filename in filter(filenames, "rhOverM_Asymptotic_GeometricUnits.h5")
    ]
    print("Finished finding files to operate on")
    pool = Pool(processes=12)
    pool.map(run_in, zip(files, range(1, len(files) + 1), [len(files)] * len(files)))
