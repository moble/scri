# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

### NOTE: The functions in this file are intended purely for inclusion in the AsymptoticBondData class.
### In particular, they assume that the first argument, `self` is an instance of AsymptoticBondData.  
### They should probably not be used outside of that class.

import os
import scri
import numpy as np
import quaternion
import spherical_functions as sf

from spherical_functions import LM_index as lm

from scipy.optimize import minimize

import ast
import json

from quaternion.calculus import indefinite_integral as integrate

### Useful Functions

def modes_time_series_to_waveform_modes(mts, dataType=scri.h):
    """Convert a ModesTimeSeries obejct to a WaveformModes object"""
    
    h = scri.WaveformModes(t=mts.t,\
                           data=np.array(mts)[:,lm(mts.ell_min, -mts.ell_min, mts.ell_min):lm(mts.ell_max+1, -(mts.ell_max+1), mts.ell_min)],\
                           ell_min=mts.ell_min,\
                           ell_max=mts.ell_max,\
                           frameType=scri.Inertial,\
                           dataType=dataType
                          )
    return h

def time_after_half_orbits(h, n_half_orbits, start_time = None):
    """Compute the time that is a certain number of half orbits past a starting time"""
    
    if start_time == None:
        start_time = h.t[0]
    a = np.angle(scri.to_coprecessing_frame(h.copy()).data[:,lm(2,2,h.ell_min)])
    maxs = np.array(np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], True], dtype=bool)
    maxs_after = maxs[np.argmin(abs(h.t - start_time)):]
    dt = h.t[np.where(maxs_after[1:] == 1)[0][n_half_orbits - 1]] - h.t[np.where(maxs_after[1:] == 1)[0][0]]
    return h.t[np.argmin(abs(h.t - (start_time + dt)))]

### Functions for CoM

def transformation_from_com_charge(G, t, t1, t2):
    """Obtain the space translation and boost velocity from the CoM charge G"""
    
    idx1 = np.argmin(abs(t - t1))
    idx2 = np.argmin(abs(t - t2))
    space_translation = []
    boost_velocity = []
    for i in range(3):
        polynomial_fit = np.polyfit(t[idx1:idx2], G[idx1:idx2, i], deg=1)
        space_translation.append(polynomial_fit[1])
        boost_velocity.append(polynomial_fit[0])

    transformation = {
        "space_translation": np.array(space_translation),
        "boost_velocity": np.array(boost_velocity)
    }
    
    return transformation

def transformation_to_map_to_com_frame(abd, t1=None, t2=None, n_iterations=5, padding_time=100, interpolate=False, return_convergence=False):
    """Obtain the optimal space translation and boost velocity to map to the CoM frame"""
    
    if t1 == None:
        t1 = time_after_half_orbits(modes_time_series_to_waveform_modes(2.0*abd.sigma.bar,scri.h), 6, 200)
    if t2 == None:
        t2 = time_after_half_orbits(modes_time_series_to_waveform_modes(2.0*abd.sigma.bar,scri.h), 8, t1)

    # interpolate to make things faster
    if interpolate:
        abd = abd.interpolate(abd.t[np.argmin(abs(abd.t - (t1 - padding_time))):np.argmin(abs(abd.t - (t2 + padding_time)))])
    
    G = abd.bondi_comoving_CoM_charge()/abd.bondi_four_momentum()[:, 0, None]
    
    com_transformation = transformation_from_com_charge(G, abd.t, t1, t2)
    
    com_transformations = [com_transformation]

    for itr in range(n_iterations - 1):
        abd_prime = abd.transform(space_translation = com_transformation["space_translation"],\
                                  boost_velocity = com_transformation["boost_velocity"])

        G_prime = abd_prime.bondi_comoving_CoM_charge()/abd_prime.bondi_four_momentum()[:, 0, None]

        com_transformation_prime = transformation_from_com_charge(G_prime, abd_prime.t, t1, t2)
        
        for transformation in com_transformation_prime:
            com_transformation[transformation] += com_transformation_prime[transformation]

        com_transformations.append(com_transformation)

    if return_convergence:
        def norm(v):
            return np.sqrt(sum(n * n for n in v))
        
        convergence = {
            "space_translation": np.array([norm(com_transformations[0]["space_translation"])] + \
                                          [0.0]*(len(com_transformations)-1)),
            "boost_velocity": np.array([norm(com_transformations[0]["boost_velocity"])] +\
                                       [0.0]*(len(com_transformations)-1))
        }
        for i in range(1,len(com_transformations)):
            for transformation in convergence:
                convergence[transformation][i] =\
                    norm(com_transformations[i][transformation]) -\
                    norm(com_transformations[i-1][transformation])
        return com_transformation, convergence
    else:
        return com_transformation

### Functions for BMS

def combine_transformations_to_supertranslation(transformations):
    """Convert time and space translations to ell=0 and ell=1 supertranslations"""
    
    combined_transformations = {}
    
    supertranslation_modes = []
    for transformation in transformations:
        if transformation == "time_translation":
            supertranslation_modes.append(0)
        if transformation == "space_translation":
            supertranslation_modes.append(1)
        if "supertranslation" in transformation:
            ell = int(transformation.split('ell_')[1])
            supertranslation_modes.append(ell)
            
    if len(supertranslation_modes) > 0:
        supertranslation = np.zeros(int((max(supertranslation_modes) + 1)**2.0)).tolist()
        for transformation in transformations:
            if transformation == "time_translation":
                supertranslation[0] = sf.constant_as_ell_0_mode(transformations[transformation][0])
            if transformation == "space_translation":
                supertranslation[1:4] = -sf.vector_as_ell_1_modes(np.array(transformations[transformation]))
            if transformation == "frame_rotation":
                combined_transformations[transformation] = transformations[transformation]
            if transformation == "boost_velocity":
                combined_transformations[transformation] = transformations[transformation]
            if "supertranslation" in transformation:
                ell = int(transformation.split('ell_')[1])
                supertranslation[int(ell**2):int(ell**2 + (2*ell+1))] = transformations[transformation]
            
        combined_transformations["supertranslation"] = supertranslation

    return combined_transformations

def transform_abd(abd, transformations):
    """Transform an ABD object using a transformations dictionary,
    assuming that the dictionary has already been converted to be
    supertranslations, a frame rotation, and a boost velocity"""
    
    if "supertranslation" in transformations:
        if "frame_rotation" in transformations:
            if "boost_velocity" in transformations:
                abd_prime = abd.transform(supertranslation=transformations["supertranslation"],\
                                          frame_rotation=transformations["frame_rotation"],\
                                          boost_velocity=transformations["boost_velocity"])
            else:
                abd_prime = abd.transform(supertranslation=transformations["supertranslation"],\
                                          frame_rotation=transformations["frame_rotation"])
        elif "boost_velocity" in transformations:
            abd_prime = abd.transform(supertranslation=transformations["supertranslation"],\
                                      boost_velocity=transformations["boost_velocity"])
        else:
            abd_prime = abd.transform(supertranslation=transformations["supertranslation"])
    elif "frame_rotation" in transformations:
        if "boost_velocity" in transformations:
            abd_prime = abd.transform(frame_rotation=transformations["frame_rotation"],\
                                      boost_velocity=transformations["boost_velocity"])
        else:
            abd_prime = abd.transform(frame_rotation=transformations["frame_rotation"])
    elif "boost_velocity" in transformations:
        abd_prime = abd.transform(boost_velocity=transformations["boost_velocity"])
    else:
        abd_prime = abd.copy()
    return abd_prime

def compute_initial_guess_for_transformation(previous_transformations, transformation, abd, iteration):
    """Transform an ABD object using a transformations dictionary,
    assuming that the dictionary has already been converted to be
    supertranslations, a frame rotation, and a boost velocity"""
    
    initial_guess = previous_transformations.copy()

    if iteration == 0:
        if transformation == "time_translation":
            initial_guess[transformation] = [0.0]
        if transformation == "frame_rotation":
            initial_guess[transformation] = [1.0, 0.0, 0.0, 0.0]
        if "supertranslation" in transformation:
            ell = int(transformation.split('ell_')[1])
            initial_guess[transformation] = [0.0]*int(2.0*ell + 1)
        
    return initial_guess

### Functions to Convert between initial guess array and transformations

def as_complexes(modes, ell_min, ell_max):
    """Convert a supertranslation to an array of real and imaginary components.
    The array is aranged as follows: the real component of the (ell,m) mode, with m increasing from -ell to 0 for each ell,
    the imaginary component of (ell,m) mode, with m increasing from -ell to -1"""
    
    complexes = []
    for L in range(ell_min, ell_max+1):
        for M in range(-L,0+1):
            complexes.append(modes[lm(L,M,ell_min)].real)
    for L in range(ell_min, ell_max+1):
        for M in range(-L,0):
            complexes.append(modes[lm(L,M,ell_min)].imag)
    return complexes

def as_modes(complexes, ell_min, ell_max):
    """Convert an array of real and imaginary components to a supertranslation.
    The array is aranged as follows: the real component of the (ell,m) mode, with m increasing from -ell to 0 for each ell,
    the imaginary component of (ell,m) mode, with m increasing from -ell to -1"""
    
    def fix_idx(L,ell_min):
        return int((L-ell_min)*(L+ell_min-1)/2)
        
    modes = np.zeros((ell_max + 1)**2 - (ell_min)**2, dtype=complex)
    for L in range(ell_min, ell_max+1):
        for M in range(-L,0+1):
            if M == 0:
                modes[lm(L,M,ell_min)] = complexes[lm(L,M,ell_min)-fix_idx(L,ell_min)]
            else:
                modes[lm(L,M,ell_min)] = complexes[lm(L,M,ell_min)-fix_idx(L,ell_min)] +\
                    1.0j*complexes[(lm(ell_max,ell_max,ell_min)-fix_idx(ell_max+1,ell_min)+1)+\
                           lm(L,M,ell_min)-fix_idx(L,ell_min)-L+ell_min]
        for M in range(1,L+1):
            modes[lm(L,M,ell_min)] = (-1)**M * np.conj(modes[lm(L,-M,ell_min)])
    return modes

def convert_initial_guess_transformations_to_initial_guess_array(initial_guess_transformations, bounds):
    """Convert a initial guess transformations to an initial guess array"""

    x0 = []
    x0_bounds = []
    x0_constraints = []
    
    frame_rotation_idx = None
    for transformation in initial_guess_transformations:
        if transformation == "time_translation":
            x0 += initial_guess_transformations[transformation]
        if transformation == "frame_rotation":
            # we only need the final three components (restrict to unit quaternions)
            x0 += initial_guess_transformations[transformation][1:4]
            frame_rotation_idx = len(x0) - 3
            x0_constraints.append({'type': 'ineq', 'fun': lambda x: 1.0 - sum(n * n for n in x[frame_rotation_idx:frame_rotation_idx + 3])})
        if "supertranslation" in transformation:
            ell = int(transformation.split('ell_')[1])
            x0 += as_complexes(initial_guess_transformations[transformation], ell, ell)

        if transformation != "space_translation" and transformation != "boost_velocity":
            for bound in bounds[transformation]:
                x0_bounds.append(bound)

    return x0, x0_bounds, tuple(x0_constraints), frame_rotation_idx

def convert_initial_guess_array_to_transformations(x0, transformation_keys, com_transformation, to_constant_and_vector=True):
    """Convert an initial guess array to transformations"""

    bms_transformation = {}

    supertranslation_modes = []
    for transformation in transformation_keys:
        if transformation == "time_translation":
            supertranslation_modes.append(0)
        if transformation == "space_translation":
            supertranslation_modes.append(1)
        if "supertranslation" in transformation:
            ell = int(transformation.split('ell_')[1])
            supertranslation_modes.append(ell)

    idx_iterator = 0
    supertranslation = np.zeros(int((max(supertranslation_modes) + 1)**2.0)).tolist()
    for transformation in transformation_keys:
        if transformation == "time_translation":
            if to_constant_and_vector:
                bms_transformation[transformation] = x0[idx_iterator:idx_iterator + 1]
            else:
                supertranslation[0] = sf.constant_as_ell_0_mode(x0[idx_iterator:idx_iterator + 1])
            idx_iterator += 1

        # space translation is held fixed by CoM correction
        if transformation == "space_translation":
            if to_constant_and_vector:
                bms_transformation[transformation] = com_transformation[transformation]
            else:
                supertranslation[1:4] = -sf.vector_as_ell_1_modes(np.array(com_transformation[transformation]))
                
        if transformation == "frame_rotation":
            bms_transformation[transformation] = [np.sqrt(1.0 - sum(n * n for n in x0[idx_iterator:idx_iterator + 3]))] + [n for n in x0[idx_iterator:idx_iterator + 3]]
            idx_iterator += 3

        # boost velocity is held fixed by CoM correction
        if transformation == "boost_velocity":
            bms_transformation[transformation] = com_transformation[transformation]
            
        if "supertranslation" in transformation:
            ell = int(transformation.split('ell_')[1])
            if to_constant_and_vector:
                bms_transformation[transformation] = as_modes(x0[idx_iterator:idx_iterator + int(2.0*ell + 1)], ell, ell)
            else:
                supertranslation[int(ell**2):int(ell**2 + (2*ell+1))] = as_modes(x0[idx_iterator:idx_iterator + int(2.0*ell + 1)], ell, ell)
            idx_iterator += int(2.0*ell + 1)
            
    if not to_constant_and_vector:
        bms_transformation["supertranslation"] = supertranslation
    
    return bms_transformation

### Functions for Minimizer

def L2_norm(h, h_target, t1_idx, t2_idx):
    """Get the L2 norm of the difference of two strains"""
    
    h_interpolated = h.interpolate(
        h.t[t1_idx:t2_idx])
    h_target_interpolated = h_target.interpolate(h_interpolated.t)

    ell_max = min(h_interpolated.ell_max, h_target_interpolated.ell_max)

    diff = h_interpolated.copy()
    diff.data = diff.data[:,lm(2,-2,diff.ell_min):lm(ell_max + 1, -(ell_max + 1), diff.ell_min)] -\
        h_target_interpolated.data[:,lm(2,-2,h_target_interpolated.ell_min):lm(ell_max + 1, -(ell_max + 1), h_target_interpolated.ell_min)]
    diff.ell_min = 2
    diff.ell_max = ell_max

    return integrate(diff.norm(), diff.t)[-1]
    
def minimize_L2_norm(x0, abd, h_target, t1_idx, t2_idx, transformation_keys=None, com_transformation=None, frame_rotation_idx=None):
    """Minimize the L2 norm of the difference of two strains;
    x0 is aranged as follows: the real component of the (ell,m) mode, with m increasing from -ell to 0 for each ell,
    the imaginary component of (ell,m) mode, with m increasing from -ell to -1"""
    
    if x0 != []:
        if frame_rotation_idx != None:
            if 1.0 - sum(n * n for n in x0[frame_rotation_idx:frame_rotation_idx + 3]) < 0:
                return 1e6
            
        bms_transformation = convert_initial_guess_array_to_transformations(x0, transformation_keys, com_transformation, to_constant_and_vector=False)

        abd_prime = transform_abd(abd, bms_transformation)
    else:
        abd_prime = abd.copy()

    h_prime = modes_time_series_to_waveform_modes(2.0*abd_prime.sigma.bar, dataType=scri.h)
    return L2_norm(h_prime, h_target, t1_idx, t2_idx)

### Functions to read transformations

def minimize_supermomentum_norm(x0, abd, t1_idx, t2_idx, transformation_keys=None, com_transformation=None, frame_rotation_idx=None):
    """Minimize the L2 norm of the difference of two strains;
    x0 is aranged as follows: the real component of the (ell,m) mode, with m increasing from -ell to 0 for each ell,
    the imaginary component of (ell,m) mode, with m increasing from -ell to -1"""
    
    if x0 != []:
        if frame_rotation_idx != None:
            if 1.0 - sum(n * n for n in x0[frame_rotation_idx:frame_rotation_idx + 3]) < 0:
                return 1e6
            
        bms_transformation = convert_initial_guess_array_to_transformations(x0, transformation_keys, com_transformation, to_constant_and_vector=False)

        abd_prime = transform_abd(abd, bms_transformation)
    else:
        abd_prime = abd.copy()

    PsiM = modes_time_series_to_waveform_modes(abd_prime.psi2 + abd_prime.sigma*abd_prime.sigma.bar.dot + abd_prime.sigma.bar.eth_GHP.eth_GHP, scri.psi2)
    PsiM.data = PsiM.data[:,lm(2,-2,PsiM.ell_min):]
    PsiM.ell_min = 2
    
    return integrate(PsiM.norm()[t1_idx:t2_idx], PsiM.t[t1_idx:t2_idx])[-1]

### Functions to read transformations

def read_in_previous_bms_transformation(transformations, json_file, return_everything=False, read_transformations_thus_far=False):
    transformations_thus_far = []
    previous_bms_transformation = {}
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
            if "transformations_thus_far" in data and read_transformations_thus_far:
                transformations_thus_far = data["transformations_thus_far"]
                transformations_thus_far.remove("space_translation")
                transformations_thus_far.remove("boost_velocity")
            bms_transformation_from_file = data["transformations"]
        if not return_everything:
            for transformation in transformations:
                if transformation in bms_transformation_from_file:
                    previous_bms_transformation[transformation] = ast.literal_eval(bms_transformation_from_file[transformation])
        else:
            for transformation in bms_transformation_from_file:
                previous_bms_transformation[transformation] = ast.literal_eval(bms_transformation_from_file[transformation])
                
    if not read_transformations_thus_far:
        return previous_bms_transformation
    else:
        return previous_bms_transformation, transformations_thus_far

### Functions to write transformations

def write_bms_transformation(bms_transformation, times, errors, json_file, iteration, complete=False):
    """Write a transformations dictionary to a json file"""
    
    default_order = ["time_translation","space_translation","frame_rotation","boost_velocity"]
    
    reordered_bms_transformation = {
        "times": {
        },
        "transformations_thus_far": {
        },
        "transformations": {
        },
        "errors": {
        },
    }

    # times
    for time in times:
        reordered_bms_transformation["times"][time] = str(np.array(times[time]).tolist())

    # transformations
    for transformation in default_order:
        if transformation in bms_transformation:
            reordered_bms_transformation["transformations"][transformation] = str(np.array(bms_transformation[transformation]).tolist())

    supertranslations = []
    for transformation in bms_transformation:
        if "supertranslation" in transformation:
            supertranslations.append(transformation)
    supertranslations = np.sort(supertranslations)
    for transformation in supertranslations:
        reordered_bms_transformation["transformations"][transformation] = str(np.array(bms_transformation[transformation]).tolist())

    # errors
    for error in errors:
        reordered_bms_transformation["errors"][error] = str(np.array(errors[error]).tolist())

    if iteration == 0:
        previous_bms_transformation = read_in_previous_bms_transformation(list(bms_transformation.keys()), json_file)
    else:
        previous_bms_transformation = read_in_previous_bms_transformation(list(bms_transformation.keys()), json_file, return_everything=True)
    if not previous_bms_transformation == {}:
        for transformation in previous_bms_transformation:
            if not transformation in reordered_bms_transformation["transformations"]:
                reordered_bms_transformation["transformations"][transformation] = str(np.array(previous_bms_transformation[transformation]).tolist())
                
    reordered_bms_transformation["transformations_thus_far"] = list(bms_transformation.keys())
            
    if complete:
        reordered_bms_transformation.pop("transformations_thus_far", None)
            
    with open(json_file, 'w') as f:
        json.dump(reordered_bms_transformation, f, indent=2, separators=(",", ": "), ensure_ascii=True)

### Main Looping Function

def transformation_to_map_to_pn_bms_frame(self, h_target, json_file, bms_transformations=None, bounds=None, t1=None, t2=None, padding_time=100, ftol=1e-12, n_iterations=4, debug=True, CoM=True):
    """Map an AsymptoticBondiData object to the BMS frame of another object

    Parameters
    ==========
    abd: AsymptoticBondiData
        The object storing the modes of the data, which will be transformed in
        this function. This is the only required argument to this function.
    h_target: WaveformModes
        The target strain waveform.
    json_file: string
        The json_file to output the transformation to.
    bms_transformations: dictionary, optional
        Defaults to
        bms_transformation = {
            "supertranslation_ell_2": None,
            "time_translation": None,
            "frame_rotation": None,
            "supertranslation_ell_3": None,
            "supertranslation_ell_4": None
        }
    bounds: dictionary, optional
        Defaults to the following bounds for the input bms transformations:
        bounds = {
            "time_translation": [(-100.0, 100.0)],
            "space_translation": [(-1.0, 1.0)]*3,
            "frame_rotation": [(-1.0, 1.0)]*3,
            "boost_velocity": [(-1e-4, 1e-4)]*3,
            "supertranslation_ell_2": [(-1.0, 1.0)]*5,
            "supertranslation_ell_3": [(-1.0, 1.0)]*7,
            "supertranslation_ell_4": [(-1.0, 1.0)]*9
        }
    t1: float, optional
        Defaults to three orbits past t=200M.
    t2: float, optional
        Defaults to four orbits past t1.
    padding time: float, optional
        Defaults to 100
    ftol: tolerance to use in the minimizer.
        Defaults to 1e-8
    n_iterations: int, optional
        Defaults to 2

    Returns
    -------
    bms_transformation: dictionary
        Object representing the optimized bms transformation.

    """

    if bms_transformations == None:
        bms_transformations = {
            "supertranslation_ell_2": None,
            "time_translation": None,
            "space_translation": [0.0]*3,
            "frame_rotation": None,
            "boost_velocity": [0.0]*3,
            "supertranslation_ell_3": None,
            "supertranslation_ell_4": None
        }

    if t1 == None:
        t1 = time_after_half_orbits(modes_time_series_to_waveform_modes(2.0*self.sigma.bar,scri.h), 6, 200)
    if t2 == None:
        t2 = time_after_half_orbits(modes_time_series_to_waveform_modes(2.0*self.sigma.bar,scri.h), 8, t1)

    if bounds == None:
        bounds = bms_transformations.copy()
        default_bounds = {
            "time_translation": [(-100.0, 100.0)],
            "space_translation": [(-1.0, 1.0)]*3,
            "frame_rotation": [(-1.0, 1.0)]*3,
            "boost_velocity": [(-1e-4, 1e-4)]*3,
            "supertranslation_ell_2": [(-1.0, 1.0)]*5,
            "supertranslation_ell_3": [(-1.0, 1.0)]*7,
            "supertranslation_ell_4": [(-1.0, 1.0)]*9
        }
        for transformation in bounds:
            bounds[transformation] = default_bounds[transformation]

    # interpolate to make things faster
    abd = self.interpolate(self.t[np.argmin(abs(self.t - (t1 - padding_time))):np.argmin(abs(self.t - (t2 + padding_time)))])

    t1_idx = np.argmin(abs(abd.t - t1))
    t2_idx = np.argmin(abs(abd.t - t2))

    error1 = minimize_L2_norm([], abd, h_target, t1_idx, t2_idx)
    
    print("\n", "Initial Error:", error1, "\n")

    for i in range(n_iterations):
        # initialize abd_prime and com transformation
        if i == 0:
            if CoM:
                com_transformation = transformation_to_map_to_com_frame(abd, t1=abd.t[t1_idx], t2=abd.t[t2_idx])
            else:
                com_transformation = {"space_translation": np.array([0.0]*3),
                                      "boost_velocity": np.array([0.0]*3)
                                  }

        print("**************\n", "Iteration:", i, "\n**************\n")

        first_bms_transformation = read_in_previous_bms_transformation(list(bms_transformations.keys()), json_file, read_transformations_thus_far=True)
        if first_bms_transformation[0] == {}:
            first_idx = 0
        else:
            first_idx = len(first_bms_transformation[1])
        for idx in range(first_idx, len(list(bms_transformations.keys())) - 2):
            transformations = bms_transformations.copy()
            free_transformation_keys = [transformation for transformation in list(bms_transformations.keys()) if bms_transformations[transformation] == None][:(idx + 1)]
            for transformation in transformations.copy():
                if not (transformation in free_transformation_keys or transformation == "space_translation" or transformation == "boost_velocity"):
                    transformations.pop(transformation)

            if debug:
                print("Transformations:", list(transformations.keys()), "\n")

            # read in the previous bms transformation for the current transformation keys (includes CoM)
            previous_bms_transformation = read_in_previous_bms_transformation(list(transformations.keys()), json_file)
            previous_bms_transformation["space_translation"] = com_transformation["space_translation"]
            previous_bms_transformation["boost_velocity"] = com_transformation["boost_velocity"]

            if debug:
                print("Previous Map:", previous_bms_transformation, "\n")

            # copy previous_bms_transformation and add the new transformation (0 for first iteration, previous transformation (with CoM) for iteration > 0)
            initial_guess_bms_transformation = compute_initial_guess_for_transformation(previous_bms_transformation, free_transformation_keys[-1], abd, iteration=i)

            if debug:
                print("Initial Guess:", initial_guess_bms_transformation, "\n")
            
            # obtain minimizer initial guesses, bounds, and constraints (only looks at non-CoM transformations)
            x0, x0_bounds, x0_constraints, frame_rotation_idx = convert_initial_guess_transformations_to_initial_guess_array(initial_guess_bms_transformation, bounds)

            if debug:
                print("Initial Guess Minimizer:", x0)
                print("Bounds:", x0_bounds)
                print("Constraints:", x0_constraints)

            # remove the other functions to speed up minimization (?)
            abd_prime = abd.copy()
            abd_prime.psi0 = 0.0*abd_prime.psi0;
            abd_prime.psi1 = 0.0*abd_prime.psi1;
            abd_prime.psi2 = 0.0*abd_prime.psi2;
            abd_prime.psi3 = 0.0*abd_prime.psi3;
            abd_prime.psi4 = 0.0*abd_prime.psi4;

            if not debug:
                res = minimize(minimize_L2_norm, x0=x0,
                               args=(abd_prime, h_target, t1_idx, t2_idx, list(transformations.keys()), com_transformation, frame_rotation_idx),
                               method='SLSQP', bounds=x0_bounds, constraints=x0_constraints,\
                               options={'ftol': ftol, 'disp': False})
            else:
                res = minimize(minimize_L2_norm, x0=x0,
                               args=(abd_prime, h_target, t1_idx, t2_idx, list(transformations.keys()), com_transformation, frame_rotation_idx),
                               method='SLSQP', bounds=x0_bounds, constraints=x0_constraints,\
                               options={'ftol': ftol, 'disp': True})
                # 'maxiter': 2})
                
            error2 = minimize_L2_norm(res.x, abd_prime, h_target, t1_idx, t2_idx, list(transformations.keys()), com_transformation, frame_rotation_idx)
                        
            print("Final Error:", error2, "\n")

            # this loops over all transformations and grabs com_transformation's transformations when necessary
            resulting_bms_transformation = convert_initial_guess_array_to_transformations(res.x, list(transformations.keys()), com_transformation, to_constant_and_vector=True)

            times = {
                "t1": t1,
                "t2": t2
            }
            errors = {
                "error1": error1,
                "error2": error2
            }

            if not idx == (len(list(bms_transformations.keys())) - 2) - 1:
                write_bms_transformation(resulting_bms_transformation, times, errors, json_file, iteration=i)
            else:
                write_bms_transformation(resulting_bms_transformation, times, errors, json_file, iteration=i, complete=True)

        print("Final Transformation:", resulting_bms_transformation, "\n")
                
        if CoM:
            # obtain the com transformation on the new abd object (only used in the following iteration)
            abd_bms_frame = transform_abd(abd, combine_transformations_to_supertranslation(resulting_bms_transformation))
            
            com_transformation = transformation_to_map_to_com_frame(abd_bms_frame, t1=abd_bms_frame.t[t1_idx], t2=abd_bms_frame.t[t2_idx])
            
            print("Found CoM Transformation:", com_transformation, "\n")
            
            com_transformation["space_translation"] += resulting_bms_transformation["space_translation"]
            com_transformation["boost_velocity"] += resulting_bms_transformation["boost_velocity"]
            
            print("Future CoM Transformation:", com_transformation, "\n")
        else:
            com_transformation = {"space_translation": np.array([0.0]*3),
                                  "boost_velocity": np.array([0.0]*3)
                              }
            
    return resulting_bms_transformation

def transformation_to_map_to_superrest_frame(self, json_file, bms_transformations=None, bounds=None, t1=None, t2=None, padding_time=100, ftol=1e-12, n_iterations=4, debug=True, CoM=True):
    """Map an AsymptoticBondiData object to the BMS frame of another object

    Parameters
    ==========

    Returns
    -------

    """

    if bms_transformations == None:
        bms_transformations = {
            "supertranslation_ell_2": None,
            "space_translation": [0.0]*3,
            "boost_velocity": [0.0]*3,
            "supertranslation_ell_3": None,
            "supertranslation_ell_4": None
        }

    if t1 == None:
        h_for_peak_time = modes_time_series_to_waveform_modes(2.0*self.sigma.bar, scri.h)
        h_for_peak_time.data = h_for_peak_time.data[:,lm(2,-2,h_for_peak_time.ell_min):]
        h_for_peak_time.ell_min = 2
        peak_time = h_for_peak_time.t[np.argmax(h_for_peak_time.norm())]
        t1 = h_for_peak_time.t[np.argmin(abs(h_for_peak_time.t - (peak_time + 150)))]
    if t2 == None:
        t2 = t1 + 200

    if bounds == None:
        bounds = bms_transformations.copy()
        default_bounds = {
            "space_translation": [(-10.0, 10.0)]*3,
            "boost_velocity": [(-1e-1, 1e-1)]*3,
            "supertranslation_ell_2": [(-1.0, 1.0)]*5,
            "supertranslation_ell_3": [(-1.0, 1.0)]*7,
            "supertranslation_ell_4": [(-1.0, 1.0)]*9
        }
        for transformation in bounds:
            bounds[transformation] = default_bounds[transformation]

    # interpolate to make things faster
    abd = self.interpolate(self.t[np.argmin(abs(self.t - (t1 - padding_time))):np.argmin(abs(self.t - (t2 + padding_time)))])
    
    t1_idx = np.argmin(abs(abd.t - t1))
    t2_idx = np.argmin(abs(abd.t - t2))

    error1 = minimize_supermomentum_norm([], abd, t1_idx, t2_idx)
    
    print("\n", "Initial Error:", error1, "\n")

    for i in range(n_iterations):
        # initialize abd_prime and com transformation
        if i == 0:
            if CoM:
                com_transformation = transformation_to_map_to_com_frame(abd, t1=abd.t[t1_idx], t2=abd.t[t2_idx])
            else:
                com_transformation = {"space_translation": np.array([0.0]*3),
                                      "boost_velocity": np.array([0.0]*3)
                                  }

        print("**************\n", "Iteration:", i, "\n**************\n")

        first_bms_transformation = read_in_previous_bms_transformation(list(bms_transformations.keys()), json_file, read_transformations_thus_far=True)
        if first_bms_transformation[0] == {}:
            first_idx = 0
        else:
            first_idx = len(first_bms_transformation[1])
        for idx in range(first_idx, len(list(bms_transformations.keys())) - 2):
            transformations = bms_transformations.copy()
            free_transformation_keys = [transformation for transformation in list(bms_transformations.keys()) if bms_transformations[transformation] == None][:(idx + 1)]
            for transformation in transformations.copy():
                if not (transformation in free_transformation_keys or transformation == "space_translation" or transformation == "boost_velocity"):
                    transformations.pop(transformation)

            if debug:
                print("Transformations:", list(transformations.keys()), "\n")

            # read in the previous bms transformation for the current transformation keys (includes CoM)
            previous_bms_transformation = read_in_previous_bms_transformation(list(transformations.keys()), json_file)
            previous_bms_transformation["space_translation"] = com_transformation["space_translation"]
            previous_bms_transformation["boost_velocity"] = com_transformation["boost_velocity"]

            if debug:
                print("Previous Map:", previous_bms_transformation, "\n")

            # copy previous_bms_transformation and add the new transformation (0 for first iteration, previous transformation (with CoM) for iteration > 0)
            initial_guess_bms_transformation = compute_initial_guess_for_transformation(previous_bms_transformation, free_transformation_keys[-1], abd, iteration=i)

            if debug:
                print("Initial Guess:", initial_guess_bms_transformation, "\n")
            
            # obtain minimizer initial guesses, bounds, and constraints (only looks at non-CoM transformations)
            x0, x0_bounds, x0_constraints, frame_rotation_idx = convert_initial_guess_transformations_to_initial_guess_array(initial_guess_bms_transformation, bounds)

            if debug:
                print("Initial Guess Minimizer:", x0)
                print("Bounds:", x0_bounds)
                print("Constraints:", x0_constraints)

            # remove the other functions to speed up minimization (?)
            abd_prime = abd.copy()

            if not debug:
                res = minimize(minimize_supermomentum_norm, x0=x0,
                               args=(abd_prime, t1_idx, t2_idx, list(transformations.keys()), com_transformation, frame_rotation_idx),
                               method='SLSQP', bounds=x0_bounds, constraints=x0_constraints,\
                               options={'ftol': ftol, 'disp': False})
            else:
                res = minimize(minimize_supermomentum_norm, x0=x0,
                               args=(abd_prime, t1_idx, t2_idx, list(transformations.keys()), com_transformation, frame_rotation_idx),
                               method='SLSQP', bounds=x0_bounds, constraints=x0_constraints,\
                               options={'ftol': ftol, 'disp': True})
                # 'maxiter': 2})
                
            error2 = minimize_supermomentum_norm(res.x, abd_prime, t1_idx, t2_idx, list(transformations.keys()), com_transformation, frame_rotation_idx)
                        
            print("Final Error:", error2, "\n")

            # this loops over all transformations and grabs com_transformation's transformations when necessary
            resulting_bms_transformation = convert_initial_guess_array_to_transformations(res.x, list(transformations.keys()), com_transformation, to_constant_and_vector=True)

            times = {
                "t1": t1,
                "t2": t2
            }
            errors = {
                "error1": error1,
                "error2": error2
            }

            if not idx == (len(list(bms_transformations.keys())) - 2) - 1:
                write_bms_transformation(resulting_bms_transformation, times, errors, json_file, iteration=i)
            else:
                write_bms_transformation(resulting_bms_transformation, times, errors, json_file, iteration=i, complete=True)

        print("Final Transformation:", resulting_bms_transformation, "\n")
                
        if CoM:
            # obtain the com transformation on the new abd object (only used in the following iteration)
            abd_bms_frame = transform_abd(abd, combine_transformations_to_supertranslation(resulting_bms_transformation))
            
            com_transformation = transformation_to_map_to_com_frame(abd_bms_frame, t1=abd_bms_frame.t[t1_idx], t2=abd_bms_frame.t[t2_idx])
            
            print("Found CoM Transformation:", com_transformation, "\n")
            
            com_transformation["space_translation"] += resulting_bms_transformation["space_translation"]
            com_transformation["boost_velocity"] += resulting_bms_transformation["boost_velocity"]
            
            print("Future CoM Transformation:", com_transformation, "\n")
        else:
            com_transformation = {"space_translation": np.array([0.0]*3),
                                  "boost_velocity": np.array([0.0]*3)
                              }
            
    return resulting_bms_transformation
    
