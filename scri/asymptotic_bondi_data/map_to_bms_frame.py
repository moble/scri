## Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

import os
import ast
import json
import scri
import numpy as np
import spherical_functions as sf
from spherical_functions import LM_index as LM

from quaternion.calculus import indefinite_integral as integrate

from scipy.optimize import minimize

def modes_time_series_to_waveform_modes(mts, dataType=scri.h):
    """Convert a ModesTimeSeries obejct to a WaveformModes object."""
    
    h = scri.WaveformModes(t=mts.t,\
                           data=np.array(mts)[:,LM(abs(mts.s), -abs(mts.s), mts.ell_min):LM(mts.ell_max+1, -(mts.ell_max+1), mts.ell_min)],\
                           ell_min=abs(mts.s),\
                           ell_max=mts.ell_max,\
                           frameType=scri.Inertial,\
                           dataType=dataType
                          )
    h.r_is_scaled_out = True
    h.m_is_scaled_out = True
    
    return h

def transformation_from_CoM_charge(G, t, t1, t2):
    """Obtain the space translation and boost velocity from the bondi_CoM_charge G."""
    
    idx1 = np.argmin(abs(t - t1))
    idx2 = np.argmin(abs(t - t2))

    polynomial_fit = np.polyfit(t[idx1:idx2], G[idx1:idx2], deg=1)

    CoM_transformation = {
        "space_translation": polynomial_fit[1],
        "boost_velocity": polynomial_fit[0]
    }
    
    return CoM_transformation


def transformation_to_map_to_CoM_frame(self, t1, t2, tol=1e-10, n_itr_max=10, padding_time=5):
    """Obtain the space translation and boost velocity to map to the CoM frame.

    This fits degree one polynomials, a * t + b, to the three components of the
    Bondi center-of-mass charge to obtain the space translation (b) and the
    boost velocity (a) needed to map the system to the CoM frame. To obtain the
    most accurate transformations, this function iterates over the aforementioned
    charge-fitting procedure until the transformations converge below the tolerance.

    Parameters
    ==========
    t1: float
        Where to start the charge-fitting procedure.

    t2: float
        Where to end the charge-fitting procedure.

    tol: float, optional
        The required tolerance for the relative error of the transformations.
        Default is 1e-10. 

    n_itr_max: int, optional
        Maximum number of iterations to perform.
        Default is 10

    padding_time: float, optional
        The time by which to extend t1 and t2 when performing transformations.
        Default is 5.

    Returns
    -------
    transformations: dict
        Dict with keys 'space_translation' and 'boost_velocity' whose values
        are the transformations needed to map to the CoM frame.
    """

    if not (t1 and t2):
        raise ValueError("The inputs t1 and t2 are both required.")
    elif t1 > t2:
        raise ValueError(f"t1 = {t1} must be less than t2 = {t2}.")
    else:
        if t1 < self.t[0] + padding_time:
            print(f"t1 = {t1} is less than self.t[0] + padding_time = {self.t[0] + padding_time}, using self.t[0] instead.")
        if t2 > self.t[-1] - padding_time:
            print(f"t2 = {t2} is more than self.t[-1] - padding_time = {self.t[-1] - padding_time}, using self.t[-1] instead.")
    
    # interpolate to make computations slightly faster
    abd = self.interpolate(self.t[np.argmin(abs(self.t - (t1 - padding_time))):\
                                  np.argmin(abs(self.t - (t2 + padding_time)))])

    G = abd.bondi_CoM_charge()/abd.bondi_four_momentum()[:, 0, None]
    
    CoM_transformation = transformation_from_CoM_charge(G, abd.t, t1, t2)
    
    CoM_transformations = [CoM_transformation]

    n_itr = 1
    rel_err = np.ones(6)
    while (rel_err > tol).sum() != 0 and n_itr < n_itr_max:
        abd_prime = abd.transform(space_translation = CoM_transformations[-1]["space_translation"],\
                                  boost_velocity = CoM_transformations[-1]["boost_velocity"])

        G_prime = abd_prime.bondi_CoM_charge()/abd_prime.bondi_four_momentum()[:, 0, None]
        
        CoM_transformation_prime = transformation_from_CoM_charge(G_prime, abd_prime.t, t1, t2)
        
        new_CoM_transformation = {}
        for transformation in CoM_transformation_prime:
            new_CoM_transformation[transformation] = CoM_transformations[-1][transformation] + CoM_transformation_prime[transformation]
            
        CoM_transformations.append(new_CoM_transformation)
        
        rel_err = np.array([(CoM_transformations[-1][transformation][i] - CoM_transformations[-2][transformation][i])\
                            / CoM_transformations[-1][transformation][i] for transformation in CoM_transformation for i in range(3)], dtype=float)
        
        n_itr += 1

    if not n_itr < n_itr_max:
        print(f"Maximum number of iterations reached; the max error was {max(rel_err)}.")
    else:
        print(f"Tolerance achieved in {n_itr} iterations!")

    return CoM_transformation


def as_complexes(modes, ell_min, ell_max):
    """Convert a supertranslation to an array of real and imaginary components.
    The array is aranged as follows: the real component of the (ell,m) mode, with m increasing from -ell to 0 for each ell,
    the imaginary component of (ell,m) mode, with m increasing from -ell to -1."""
    
    complexes = []
    for L in range(ell_min, ell_max+1):
        for M in range(-L, 0 + 1):
            complexes.append(modes[LM(L, M, ell_min)].real)
    for L in range(ell_min, ell_max+1):
        for M in range(-L, 0):
            complexes.append(modes[LM(L, M, ell_min)].imag)
            
    return complexes


def as_modes(complexes, ell_min, ell_max):
    """Convert an array of real and imaginary components to a supertranslation.
    The array is aranged as follows: the real component of the (ell,m) mode, with m increasing from -ell to 0 for each ell,
    the imaginary component of (ell,m) mode, with m increasing from -ell to -1."""
    
    def fix_idx(L, ell_min):
        return int((L - ell_min) * (L + ell_min - 1) / 2)
        
    modes = np.zeros((ell_max + 1)**2 - ell_min**2, dtype=complex)
    for L in range(ell_min, ell_max + 1):
        for M in range(-L, 0 + 1):
            if M == 0:
                modes[LM(L, M, ell_min)] = complexes[LM(L, M, ell_min) - fix_idx(L, ell_min)]
            else:
                modes[LM(L, M, ell_min)] = complexes[LM(L, M, ell_min) - fix_idx(L, ell_min)] +\
                    1.0j*complexes[(LM(ell_max, ell_max, ell_min) - fix_idx(ell_max + 1, ell_min) + 1) +\
                                   LM(L, M, ell_min) - fix_idx(L, ell_min) - L + ell_min]
        for M in range(1,L+1):
            modes[LM(L,M,ell_min)] = (-1)**M * np.conj(modes[LM(L, -M, ell_min)])
            
    return modes
    

def convert_parameters_to_transformation(x0, transformation_keys, CoM_transformation=None, combine_translations=True):
    """Converts the parameters x0, which are arranged in order of the transformation_keys,
    to a dict containing the keys 'supertranslation', 'frame_rotation', and 'boost_velocity'.
    For transformations involving complex parameters, the x0 interval corresponding to that
    transformation is organized as the real component of the (ell,m) mode, with m increasing
    from -ell to 0 for each ell, and then the imaginary component, with m increasing from
     -ell to -1. The reason we only include -m parameters is because the +m parameters
    are fixed by requiring that the transformations are real."""
    
    bms_transformation = {
        'supertranslation': 0,
        'frame_rotation': [1, 0, 0, 0],
        'boost_velocity': [0, 0, 0],
    }             
    if not combine_translations:
        bms_transformation = {}

    if CoM_transformation != None:
        transformation_keys.append('space_translation')
        transformation_keys.append('boost_velocity')
        
    supertranslation_modes = []
    for transformation in transformation_keys:
        if transformation == "time_translation":
            supertranslation_modes.append(0)
        if transformation == "space_translation":
            supertranslation_modes.append(1)
        if "supertranslation" in transformation:
            ell = int(transformation.split('ell_')[1])
            supertranslation_modes.append(ell)
            
    idx = 0
    supertranslation = np.zeros(int((max(supertranslation_modes) + 1)**2.0)).tolist()
    for transformation in transformation_keys:
        if transformation == "time_translation":
            if not combine_translations:
                bms_transformation[transformation] = x0[idx:idx + 1]
            else:
                supertranslation[0] = sf.constant_as_ell_0_mode(x0[idx])
            idx += 1

        if transformation == "space_translation":
            if CoM_transformation == None:
                if not combine_translations:
                    bms_transformation[transformation] = x0[idx:idx + 3]
                else:
                    supertranslation[1:4] = as_modes(x0[idx:idx + 3], 1, 1)
                    
                idx += 3
            else:
                if not combine_translations:
                    bms_transformation[transformation] = CoM_transformation[transformation]
                else:
                    supertranslation[1:4] = as_modes(CoM_transformation[transformation], 1, 1)

        if transformation == "frame_rotation":
            bms_transformation[transformation] = [np.sqrt(1.0 - sum(n * n for n in x0[idx:idx + 3]))] + [n for n in x0[idx:idx + 3]]
            idx += 3

        if transformation == "boost_velocity":
            if CoM_transformation == None:
                bms_transformation[transformation] = x0[idx:idx + 3]
                idx += 3
            else:
                bms_transformation[transformation] = CoM_transformation[transformation]

        if "supertranslation" in transformation:
            ell = int(transformation.split('ell_')[1])
            if not combine_translations:
                bms_transformation[transformation] = as_modes(x0[idx:idx + int(2.0 * ell + 1)], ell, ell)
            else:
                supertranslation[int(ell**2):int(ell**2 + (2 * ell + 1))] = as_modes(x0[idx:idx + int(2.0*ell + 1)], ell, ell)
            idx += int(2.0 * ell + 1)

    if combine_translations:
        bms_transformation["supertranslation"] = supertranslation

    return bms_transformation


def initial_guess_transformations_to_x0(initial_guess_transformations, bounds):
    """Converts the bms transformations in the transformation dict to an array of parameters.
    For transformations involving complex parameters, the x0 interval corresponding to that
    transformation is organized as the real component of the (ell,m) mode, with m increasing
    from -ell to 0 for each ell, and then the imaginary component, with m increasing from
     -ell to -1. The reason we only include -m parameters is because the +m parameters
    are fixed by requiring that the transformations are real."""
    
    x0 = []
    x0_bounds = []
    x0_constraints = []

    frame_rotation_idx = None
    for transformation in initial_guess_transformations:
        if initial_guess_transformations[transformation] == None:
            break

        if transformation == "time_translation":
            x0 += initial_guess_transformations[transformation]
        elif transformation == "space_translation":
            x0 += as_complexes(initial_guess_transformations[transformation], 1, 1)
        elif transformation == "frame_rotation":
            # we only need the final three components (restrict to unit quaternion)
            x0 += initial_guess_transformations[transformation][1:4]
            frame_rotation_idx = len(x0) - 3
            x0_constraints.append({'type': 'ineq', 'fun': lambda x: 1.0 - sum(n * n for n in x[frame_rotation_idx:frame_rotation_idx + 3])})
        elif transformation == "boost_velocity":
            # boost velocity cannot a norm of one
            x0 += initial_guess_transformations[transformation]
            boost_velocity_idx = len(x0) - 3
            x0_constraints.append({'type': 'ineq', 'fun': lambda x: 1.0 - sum(n * n for n in x[boost_velocity_idx:boost_velocity_idx + 3])})
        elif "supertranslation" in transformation:
            ell = int(transformation.split('ell_')[1])
            x0 += as_complexes(initial_guess_transformations[transformation], ell, ell)
            
        for bound in bounds[transformation]:
            x0_bounds.append(bound)

    return x0, x0_bounds, tuple(x0_constraints), frame_rotation_idx, boost_velocity_idx
        

def initial_guess_from_bms_transformation(bms_transformation, bounds, transformation_names):
    """Initialize a dict of bms transformations. If previous information is available,
    the initial guess will use that information, otherwise it will just be zero."""
    
    initial_guess_transformations = {}
    for transformation in transformation_names:
        if bms_transformation[transformation] == None:
            if transformation == "time_translation":
                initial_guess_transformations[transformation] = [0.0]
            elif transformation == "space_translation":
                initial_guess_transformations[transformation] = [0.0]*3
            elif transformation == "frame_rotation":
                initial_guess_transformations[transformation] = [1.0] + [0.0]*3
            elif transformation == "boost_velocity":
                initial_guess_transformations[transformation] = [0.0]*3
            elif "supertranslation" in transformation:
                ell = int(transformation.split('ell_')[1])
                initial_guess_transformations[transformation] = [0.0]*int(2.0*ell + 1)
        else:
            initial_guess_transformations[transformation] = bms_transformation[transformation]
            
    return initial_guess_transformations_to_x0(initial_guess_transformations, bounds)


def func_to_minimize(x0, abd, h_target, t1_idx, t2_idx, transformation_keys, CoM_transformation=None, frame_rotation_idx=None, boost_velocity_idx=None):
    """Function to be minimized by SciPy's minimization function. This will either minimize
    the norm of the difference of two strain waveforms if h_target != None, otherwise it
    will minimize the norm of the Moreschi supermomentum."""
    
    minimize_Moreschi_supermomentum = True
    if h_target != None:
        minimize_Moreschi_supermomentum = False

    if np.array(x0).size != 0:
        if frame_rotation_idx != None:
            if 1.0 - sum(n * n for n in x0[frame_rotation_idx:frame_rotation_idx + 3]) < 0:
                return 1e6
        if boost_velocity_idx != None:
            if 1.0 - sum(n * n for n in x0[boost_velocity_idx:boost_velocity_idx + 3]) < 0:
                return 1e6
            
        # this converts the parameters used in the minimization to a dict containing
        # 'supertranslation', 'frame_rotation', and 'boost_velocity'
        bms_transformation = convert_parameters_to_transformation(x0, transformation_keys, CoM_transformation)
            
        abd_prime = abd.transform(supertranslation=bms_transformation['supertranslation'],
                                  frame_rotation=bms_transformation['frame_rotation'],
                                  boost_velocity=bms_transformation['boost_velocity'])
    else:
        abd_prime = abd.copy()

    if minimize_Moreschi_supermomentum:
        PsiM = modes_time_series_to_waveform_modes(abd_prime.supermomentum('Moreschi'), scri.psi2).interpolate(abd_prime.t[t1_idx:t2_idx])
        return integrate(PsiM.norm(), PsiM.t)[-1 ]/ (abd_prime.t[t2_idx] - abd_prime.t[t1_idx])
    else:
        h = modes_time_series_to_waveform_modes(2.0*abd_prime.sigma.bar)[t1_idx:t2_idx]
        h = h[:,:(h_target.ell_max + 1)]
        h_diff = h.copy()
        h_diff.data -= h_target.interpolate(h.t).data
        return integrate(h_diff.norm(), h_diff.t)[-1] / (h_diff.t[-1] - h_diff.t[0])

    
def read_bms_transformation(bms_transformation, json_file, no_CoM=False):
    """Read a dictionary of transformations from a json file."""
    
    bms_transformation_from_file = {}
    if os.path.exists(json_file):
        with open(json_file, 'r') as f: 
            bms_transformation_from_file = json.load(f)['transformations']
        for transformation in bms_transformation_from_file:
            bms_transformation[transformation] = ast.literal_eval(bms_transformation_from_file[transformation])
    
    if no_CoM:
        bms_transformation.pop('space_translation', None)
        bms_transformation.pop('boost_velocity', None)
    
    return bms_transformation

    
def write_bms_transformation(bms_transformation, json_file, times, errors, in_order=False):
    """Write a dictionary of transformations to a json file."""
    
    transformation_and_minimize_information = {
        'times': {
        },
        'transformations': {
        },
        'errors': {
        },
    }

    transformation_and_minimize_information['times'] = times
    if not in_order:
        for transformation in bms_transformation:
            transformation_and_minimize_information['transformations'][transformation] = str(np.array(bms_transformation[transformation]).tolist())
    else:
        order = ["time_translation", "space_translation", "frame_rotation", "boost_velocity"]
        for transformation in order:
            if transformation in bms_transformation:
                transformation_and_minimize_information['transformations'][transformation] = str(np.array(bms_transformation[transformation]).tolist())
        for transformation in bms_transformation:
            if 'supertranslation' in transformation:
                transformation_and_minimize_information['transformations'][transformation] = str(np.array(bms_transformation[transformation]).tolist())
    
    with open(json_file, 'w') as f:
        json.dump(transformation_and_minimize_information, f, indent=2, separators=(",", ": "), ensure_ascii=True)
    

def transformation_to_map_to_BMS_frame(self, json_file, t1, t2,
                                       bms_transformation={
                                           "supertranslation_ell_2": None,
                                           "time_translation":       None,
                                           "space_translation":      None,
                                           "frame_rotation":         None,
                                           "boost_velocity":         None,
                                           "supertranslation_ell_3": None,
                                           "supertranslation_ell_4": None
                                       },
                                       bounds={
                                           "time_translation":       [(-100.0, 100.0)],
                                           "space_translation":      [(-1.0, 1.0)]*3,
                                           "frame_rotation":         [(-1.0, 1.0)]*3,
                                           "boost_velocity":         [(-1e-2, 1e-2)]*3,
                                           "supertranslation_ell_2": [(-1.0, 1.0)]*5,
                                           "supertranslation_ell_3": [(-1.0, 1.0)]*7,
                                           "supertranslation_ell_4": [(-1.0, 1.0)]*9
                                       },
                                       tol=1e-12, n_itr_max=1, padding_time=100, h_target=None, CoM=False):
    """Obtain the BMS transformation to map to a specified BMS frame. This may either
    be the BMS frame corresponding to minimizing the Moreschi supermomentum or
    the BMS frame of an input strain waveform, e.g., a PN waveform.

    This uses SciPy's Sequential Lease Squares Programing to perform a minimization
    over a range of BMS transformations. To help this minimizer converge, we iteratively
    loop over each BMS transformation, using the previous findings as initial guesses.

    Parameters
    ==========
    json_file: string
        Where to read/write BMS transformations from/to.

    t1: float
        Where to start the minimization procedure.

    t2: float
        Where to end the minimization procedure.

    bms_transformation: dict, optional
        Which transformations to include in the minmization. The order of the keys
        determines the order in which the transformations will be included in the minimizaiton.
        Default is
        bms_transformation = {
            "supertranslation_ell_2": None,
            "time_translation":       None,
            "space_translation":      None,
            "frame_rotation":         None,
            "boost_velocity":         None,
            "supertranslation_ell_3": None,
            "supertranslation_ell_4": None
        }
        Note that performing the supertranslation_ell_2 first is important because this
        tends to be the most influencial factor when mapping to a BMS frame.

    bounds: dict, optional
        Bounds on the transformations provided above.
        Default is
        bounds = {
            "time_translation":       [(-100.0, 100.0)],
            "space_translation":      [(-1.0, 1.0)]*3,
            "frame_rotation":         [(-1.0, 1.0)]*3,
            "boost_velocity":         [(-1e-4, 1e-4)]*3,
            "supertranslation_ell_2": [(-1.0, 1.0)]*5,
            "supertranslation_ell_3": [(-1.0, 1.0)]*7,
            "supertranslation_ell_4": [(-1.0, 1.0)]*9
        }

    tol: float, optional
        The required tolerance for the relative error of the transformations.
        Default is 1e-12.

    n_itr_max: int, optional
        Maximum number of iterations to perform. If CoM is True, then
        this should be, at a minimum, more than 1.
        Default is 1. 

    padding_time: float, optional
        The time by which to extend t1 and t2 when performing transformations.
        Default is 100

    h_target: WaveformModes, optional
        The strain whose BMS frame we will try to map to.
        Default is None, i.e., minimize the L2 norm of the Moreschi supermomentum.

    CoM: bool, optional
        Determine the space translation and boost velocity based on the
        Bondi center-of-mass charge.
        Default is False.

    Returns
    -------
    transformations: dict
        Dict whose keys are
           * time_translation
           * space_translation
           * boost_velocity
           * frame_rotation
           * supertranslation_ell_2
           * supertranslation_ell_3
           * supertranslation_ell_4
           ...
    """

    if not (t1 and t2):
        raise ValueError("The inputs t1 and t2 are both required.")
    elif t1 > t2:
        raise ValueError(f"t1 = {t1} must be less than t2 = {t2}.")
    else:
        if t1 < self.t[0] + padding_time:
            print(f"t1 = {t1} is less than self.t[0] + padding_time = {self.t[0] + padding_time}, using self.t[0] instead.")
        if t2 > self.t[-1] - padding_time:
            print(f"t2 = {t2} is more than self.t[-1] - padding_time = {self.t[-1] - padding_time}, using self.t[-1] instead.")
            
        peak_time = self.t[np.argmax(modes_time_series_to_waveform_modes(2.0*self.sigma.bar).norm())]
        if t1 < peak_time and t2 > peak_time:
            print("Warning: it does not make much sense to have t1 < peak time and t2 > peak time.")
            print(f"t1 = {t1}, t2 = {t2}, peak_time = {peak_time}.")
            
    if h_target == None and not CoM:
        print("Warning: it is recommended to use CoM = True if no h_target is provided.")

    # check to make sure bms_transformations and bounds match
    if list(bms_transformation.keys()).sort() != list(bounds.keys()).sort():
        raise ValueError("The keys of bms_transformations and bounds do not match.")

    if ('space_translation' in bms_transformation or 'boost_velocity' in bms_transformation) and CoM:
        raise ValueError("If CoM is true, then the bms_transformation cannot include space_translation or boost_velocity.")

    # fix h_target to avoid interpolate bug
    if h_target != None:
        h_target = h_target.copy()
    
    # interpolate to make things faster
    abd = self.interpolate(self.t[np.argmin(abs(self.t - (t1 - padding_time))):np.argmin(abs(self.t - (t2 + padding_time)))])

    t1_idx = np.argmin(abs(abd.t - t1))
    t2_idx = np.argmin(abs(abd.t - t2))
    
    error1 = func_to_minimize([], abd, h_target, t1_idx, t2_idx, [])
    
    # iterate over minimizer to find the best BMS transformation;
    # this may not even be necessary and really only makes sense if we're using CoM = True
    for itr in range(n_itr_max):
        if os.path.exists(json_file):
            prev_bms_transformation = read_bms_transformation(bms_transformation, json_file)
        else:
            prev_bms_transformation = {}
        for transformation in prev_bms_transformation:
            if transformation == None:
                bms_transformation.pop(transformation)

        # if a previous bms transformation exists, apply it, find the new CoM transformation,
        # and then compose it with the previous CoM transformation. Otherwise, just find
        # the CoM transformation based on the input ABD object.
        if CoM:
            if prev_bms_transformation != {}:
                CoM_transformation = {}
                for transformation in ['space_translation','boost_velocity']:
                    CoM_transformation[transformation] = prev_bms_transformation[transformation]
                    prev_bms_transformation.pop(transformation, None)
                bms_transformation_params, _, _, _ = initial_guess_from_bms_transformation(prev_bms_transformation, bounds, list(prev_bms_transformation.keys()))
                bms_transformation_to_apply = convert_parameters_to_transformation(bms_transformation_params, list(prev_bms_transformation.keys()), CoM_transformation)
                abd_prime = abd.transform(supertranslation=bms_transformation_to_apply['supertranslation'],
                                          frame_rotation=bms_transformation_to_apply['frame_rotation'],
                                          boost_velocity=bms_transformation_to_apply['boost_velocity'])
            else:
                abd_prime = abd.copy()

            new_CoM_transformation = abd_prime.transformation_to_map_to_CoM_frame(abd_prime.t[t1_idx], abd_prime.t[t2_idx])
            for transformation in CoM_transformation:
                CoM_transformation[transformation] += new_CoM_transformation[transformation]
        else:
            CoM_transformation = None
        
        # iterate over the BMS transformations that we want to apply
        for transformation in bms_transformation:
            if os.path.exists(json_file):
                bms_transformation = read_bms_transformation(bms_transformation, json_file, no_CoM=CoM)
            
            # this only happens if we are performing a restart
            if itr == 0 and bms_transformation[transformation] != None:
                continue
                
            transformation_names = list(bms_transformation.keys())[:list(bms_transformation.keys()).index(transformation) + 1]
                        
            x0, x0_bounds, x0_constraints, frame_rotation_idx, boost_velocity_idx = initial_guess_from_bms_transformation(bms_transformation, bounds, transformation_names)
            
            # run minimization
            res = minimize(func_to_minimize, x0=x0,
                           args=(abd, h_target, t1_idx, t2_idx, transformation_names, CoM_transformation, frame_rotation_idx, boost_velocity_idx),
                           method='SLSQP', bounds=x0_bounds, constraints=x0_constraints,
                           options={'ftol': tol, 'disp': True})
            
            x0 = res.x
            error2 = func_to_minimize(x0, abd, h_target, t1_idx, t2_idx, transformation_names, CoM_transformation, frame_rotation_idx, boost_velocity_idx)
            
            # convert x0 to transformations
            resulting_bms_transformation = convert_parameters_to_transformation(x0, transformation_names, combine_translations=False)
            
            if CoM:
                for transformation in CoM_transformation:
                    resulting_bms_transformation[transformation] = CoM_transformation[transformation]
            
            # write transformations
            if not (transformation == list(bms_transformation.keys())[-1] and itr == n_itr_max - 1):
                write_bms_transformation(resulting_bms_transformation, json_file, {'t1': t1, 't2': t2}, {'error1': error1, 'error2': error2})
            else:
                write_bms_transformation(resulting_bms_transformation, json_file, {'t1': t1, 't2': t2}, {'error1': error1, 'error2': error2}, in_order=True)
                
    return resulting_bms_transformation
