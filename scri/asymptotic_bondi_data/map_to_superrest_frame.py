import os
import copy
import warnings
import numpy as np

import sxs
import scri
import spinsfast
import spherical_functions as sf
from spherical_functions import LM_index as LM

import quaternion
from quaternion.calculus import derivative

from scipy.interpolate import CubicSpline

def MT_to_WM(h_mts, sxs_version=False, dataType=scri.h):
    """Function for converting a ModesTimeSeries object to a
    scri.WaveformModes or a sxs.WaveformModes object.

    """
    if not sxs_version:
        h = scri.WaveformModes(t=h_mts.t,\
                           data=np.array(h_mts)[:,sf.LM_index(abs(h_mts.s),-abs(h_mts.s),0):],\
                           ell_min=abs(h_mts.s),\
                           ell_max=h_mts.ell_max,\
                           frameType=scri.Inertial,\
                           dataType=dataType
                          )
        h.r_is_scaled_out = True
        h.m_is_scaled_out = True
        return h
    else:
        h = sxs.WaveformModes(input_array=np.array(h_mts)[:,sf.LM_index(abs(h_mts.s),-abs(h_mts.s),0):],\
                              time=h_mts.t,\
                              time_axis=0,\
                              modes_axis=1,\
                              ell_min=abs(h_mts.s),\
                              ell_max=h_mts.ell_max,\
                              spin_weight=h_mts.s)
        return h

def 𝔇(h, ell_max):
    """Differential operator 𝔇 acting on spin-weight s=0 function.

    This is equivalent to ð^{2}\bar{ð}^{2}.

    It is explicitly defined in Eqs (16) and (17) of https://doi.org/10.1063/1.532646.

    """
    h_with_operator = h.copy()
    for ell in range(0, ell_max + 1):
        if ell < 2:
            value = 0
        else:
            value = ((ell + 2) * (ell + 1) * (ell) * (ell - 1)) / 4.
        h_with_operator[LM(ell, -ell, 0) : LM(ell, ell, 0) + 1] *= value

    return h_with_operator

def 𝔇inverse(h, ell_max):
    """Inverse of differential operator 𝔇 acting on spin-weight s=0 function.
   
    """
    h_with_operator = h.copy()
    for ell in range(0, ell_max + 1):
        if ell < 2:
            value = 0
        else:
            value = 4. / ((ell + 2) * (ell + 1) * (ell) * (ell - 1))
        h_with_operator[LM(ell, -ell, 0) : LM(ell, ell, 0) + 1] *= value

    return h_with_operator

def compute_bondi_rest_mass_and_conformal_factor(PsiM, ell_max):
    """Compute the Bondi rest mass and the conformal factor K from
    the Moreschi supermomentum to help obtain the Moreschi supermomentum
    in a supertranslated frame. These are defined in Eqs (14) and (15) of
    https://doi.org/10.1063/1.532646.

    This handles two differenct cases: one where the Moreschi supermomentum
    is a function of time and one where it is not.

    """
    if len(PsiM.shape) > 1:
        P = np.empty(PsiM.shape, dtype=float)
        P[..., 0] = PsiM[:, 0].real
        P[..., 1] = (PsiM[:, 1] - PsiM[:, 3]).real / np.sqrt(6)
        P[..., 2] = (PsiM[:, 1] + PsiM[:, 3]).imag / np.sqrt(6)
        P[..., 3] = PsiM[:, 2].real / np.sqrt(3)
    else:
        P = np.zeros(4, dtype=float)
        P[0] = PsiM[0].real
        P[1] = (PsiM[1] - PsiM[3]).real / np.sqrt(6)
        P[2] = (PsiM[1] + PsiM[3]).imag / np.sqrt(6)
        P[3] = PsiM[2].real / np.sqrt(3)
    P /= -np.sqrt(4 * np.pi)
    
    r_vector_modes_on_S2 = []
    for L, M in [(0, 0), (1, -1), (1, 0), (1, +1)]:
        r_vector_modes = np.zeros((ell_max + 1) ** 2)
        r_vector_modes[LM(L, M, 0)] = 1
        r_vector_modes_on_S2.append(spinsfast.salm2map(r_vector_modes, 0, ell_max, 2 * ell_max + 1, 2 * ell_max + 1))

    r_vector = np.zeros(4, dtype=np.ndarray)
    r_vector[0] = r_vector_modes_on_S2[0].real
    r_vector[1] = (r_vector_modes_on_S2[1] - r_vector_modes_on_S2[3]).real / np.sqrt(6)
    r_vector[2] = -(r_vector_modes_on_S2[1] + r_vector_modes_on_S2[3]).imag / np.sqrt(6)
    r_vector[3] = r_vector_modes_on_S2[2].real / np.sqrt(3)
    r_vector *= np.sqrt(4 * np.pi)

    # for some reason np.sum() fails for certain indices...
    if len(PsiM.shape) > 1:
        M_S2 = np.sqrt(P[:, 0] ** 2 - (P[:, 1] ** 2 + P[:, 2] ** 2 + P[:, 3] ** 2))

        K_S2 = M_S2[:, None, None] / (np.tensordot(P[:, 0], r_vector[0], axes=0) -\
                                      (np.tensordot(P[:, 1], r_vector[1], axes=0) +\
                                       np.tensordot(P[:, 2], r_vector[2], axes=0) +\
                                       np.tensordot(P[:, 3], r_vector[3], axes=0)))
    else:
        M_S2 = np.sqrt(P[0] ** 2 - (P[1] ** 2 + P[2] ** 2 + P[3] ** 2))
        
        K_S2 = M_S2 / (P[0] * r_vector[0] - (P[1] * r_vector[1] + P[2] * r_vector[2] + P[3] * r_vector[3]))
    
    return M_S2, K_S2

def compute_Moreschi_supermomentum(abd, alpha, ell_max):
    """Compute the Moreschi supermomentum in a supertranslated frame.
    This transformation can be found in Eq (9) of 
    https://doi.org/10.1063/1.532646.

    """
    PsiM = abd.supermomentum('Moreschi')
    M_S2, K_S2 = compute_bondi_rest_mass_and_conformal_factor(np.array(PsiM), ell_max)
    
    PsiM_S2 = PsiM.grid()
    PsiM_S2_interp = sf.SWSH_grids.Grid(np.zeros((2 * ell_max + 1, 2 * ell_max + 1), dtype=complex), spin_weight=0)
    K_S2_interp = sf.SWSH_grids.Grid(np.zeros((2 * ell_max + 1, 2 * ell_max + 1), dtype=complex), spin_weight=0)

    # interpolate these functions onto the new retarded time
    for i in range(2 * ell_max + 1):
        for j in range(2 * ell_max + 1):
            alpha_i_j = alpha[i, j]
            PsiM_S2_interp[i, j] = CubicSpline(
                PsiM.t, PsiM_S2[:, i, j]
            )(alpha_i_j)
            
            K_S2_interp[i, j] = CubicSpline(
                PsiM.t, K_S2[:, i, j]
            )(alpha_i_j)

    # compute the supertranslation term
    alpha_modes = spinsfast.map2salm(alpha.view(np.ndarray), 0, ell_max)
    D_alpha_modes = 𝔇(alpha_modes, ell_max)
    D_alpha_modes_S2 = sf.SWSH_grids.Grid(spinsfast.salm2map(D_alpha_modes.view(np.ndarray), 0, ell_max, 2 * ell_max + 1, 2 * ell_max + 1), spin_weight=0)

    # transform the Moreschi supermomentum
    PsiM_S2_interp = (PsiM_S2_interp - D_alpha_modes_S2) / K_S2_interp ** 3

    PsiM_interp = spinsfast.map2salm(PsiM_S2_interp.view(np.ndarray), 0, ell_max)
    
    return PsiM_interp

def compute_alpha_perturbation(PsiM, M_S2, K_S2, ell_max):
    """From the Moreschi supermomentum transformation law,
    compute the supertranslation that maps to the superrest frame.

    This equation can be found in Eq (10) of
    https://doi.org/10.1063/1.532646.

    """
    PsiM_S2 = spinsfast.salm2map(PsiM.view(np.ndarray), 0, ell_max, 2 * ell_max + 1, 2 * ell_max + 1)
    PsiM_plus_M_K3_S2 = PsiM_S2 + M_S2 * K_S2 ** 3
    PsiM_plus_M_K3 = spinsfast.map2salm(PsiM_plus_M_K3_S2.view(np.ndarray), 0, ell_max)
    
    alpha = 𝔇inverse(PsiM_plus_M_K3, ell_max)
    
    return spinsfast.salm2map(alpha, 0, ell_max, 2 * ell_max + 1, 2 * ell_max + 1).real

def supertranslation_to_map_to_super_rest_frame(abd, N_itr_max=10, rel_err_tol=1e-12, ell_max=None):
    """Determine the supertranslation needed to map an abd object to the superrest frame
    through an iterative solve; e.g., compute the supertranslation needed to minimize
    the Moreschi supermomentum according to Eq (10) of https://doi.org/10.1063/1.532646,
    transform the Moreschi supermomentum, and repeat until the supertranslation converges.
    
    """
    
    alpha_S2 = np.zeros((2 * ell_max + 1, 2 * ell_max + 1), dtype=complex)
    
    itr = 0
    rel_err = np.inf
    rel_errs = []
    while itr < N_itr_max and not rel_err < rel_err_tol:
        prev_alpha_S2 = sf.SWSH_grids.Grid(alpha_S2.copy(), spin_weight=0)

        PsiM = compute_Moreschi_supermomentum(abd, alpha_S2, ell_max)
        
        M_S2, K_S2 = compute_bondi_rest_mass_and_conformal_factor(np.array(PsiM), ell_max)
        
        alpha_S2 += compute_alpha_perturbation(PsiM, M_S2, K_S2, ell_max)
        
        rel_err = (spinsfast.map2salm((abs(sf.SWSH_grids.Grid(alpha_S2.copy(), spin_weight=0) - prev_alpha_S2) /\
                                       abs(sf.SWSH_grids.Grid(alpha_S2.copy(), spin_weight=0))).view(np.ndarray), 0, ell_max)[LM(0, 0, 0)] / np.sqrt(4 * np.pi)).real
        rel_errs.append(rel_err)
        
        itr += 1
        
    if not itr < N_itr_max:
        print(f"supertranslation: maximum number of iterations reached; the min error was {min(np.array(rel_errs).flatten())}.")
    else:
        print(f"supertranslation: tolerance achieved in {itr} iterations!")
        
    supertranslation = spinsfast.map2salm(alpha_S2.view(np.ndarray), 0, ell_max)
    supertranslation[0:4] = 0
        
    return supertranslation, rel_errs

def transformation_from_CoM_charge(G, t):
    """Obtain the space translation and boost velocity from the center-of-mass charge.

    This is defined in Eq (18) of https://journals.aps.org/prd/abstract/10.1103/PhysRevD.104.024051.
    
    """

    polynomial_fit = np.polyfit(t, G, deg=1)

    CoM_transformation = {
        "space_translation": polynomial_fit[1],
        "boost_velocity": polynomial_fit[0]
    }
    
    return CoM_transformation

def com_transformation_to_map_to_super_rest_frame(abd, N_itr_max=10, rel_err_tol=1e-12, ell_max=None, space_translation=True, boost_velocity=True):
    """Determine the space translation and boost needed to map an abd object to the superrest frame
    through an iterative solve; e.g., compute the transformations needed to minimize
    the center-of-mass charge, transform the abd object, and repeat until the transformations converge.

    This function can also be used to find just the space translation or the boost velocity, rather than both.
    
    """
    
    if not space_translation and not boost_velocity:
        raise ValueError('space_translation and boost_velocity cannot both be False.')
    
    CoM_transformation = {
        "space_translation": np.zeros(3),
        "boost_velocity": np.zeros(3)
    }
        
    itr = 0
    rel_err = np.array(2*[np.inf])
    rel_errs = []
    while itr < N_itr_max and not (rel_err < rel_err_tol).sum() == int(space_translation) + int(boost_velocity):
        prev_CoM_transformation = copy.deepcopy(CoM_transformation)
        
        if itr == 0:
            abd_prime = abd.copy()
        else:
            abd_prime = abd.transform(space_translation = CoM_transformation["space_translation"],\
                                      boost_velocity = CoM_transformation["boost_velocity"])

        G_prime = abd_prime.bondi_CoM_charge()/abd_prime.bondi_four_momentum()[:, 0, None]
        
        new_CoM_transformation = transformation_from_CoM_charge(G_prime, abd_prime.t)
        for transformation in CoM_transformation:
            if not space_translation:
                if transformation == 'space_translation':
                    continue
            if not boost_velocity:
                if transformation == 'boost_velocity':
                    continue
            CoM_transformation[transformation] += new_CoM_transformation[transformation]
        
        for i, transformation in enumerate(CoM_transformation):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
                err = np.sqrt(abs(np.dot(CoM_transformation[transformation] - prev_CoM_transformation[transformation],\
                                         CoM_transformation[transformation] - prev_CoM_transformation[transformation]) /\
                                  np.dot(CoM_transformation[transformation], CoM_transformation[transformation])))
   
                if err == np.inf:
                    err = 1
                rel_err[i] = err
        
        rel_errs.append(copy.deepcopy(rel_err))
        
        itr += 1

    transformation_name = 'CoM'
    if not space_translation:
        transformation_name = 'boost'
        transformation = CoM_transformation['boost_velocity']
        rel_errs = np.array(rel_errs)[:, 1:]
    elif not boost_velocity:
        transformation_name = 'space_translation'
        transformation = CoM_transformation['space_translation']
        rel_errs = np.array(rel_errs)[:, :1]
    else:
        transformation = CoM_transformation
        rel_errs = np.array(rel_errs)
        
    if not itr < N_itr_max:
        print(f"{transformation_name}: maximum number of iterations reached; the min error was {min(rel_errs.flatten())}.")
    else:
        print(f"{transformation_name}: tolerance achieved in {itr} iterations!")
   
    return transformation, rel_errs

def rotation_from_spin_charge(chi, t):
    """Obtain the rotation from the angular momentum charge.

    This is defined in Eq (15) of https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.124029.
    
    """
    
    chi_f = chi[np.argmin(abs(t))]
    
    theta = np.arccos(chi_f[2]/np.linalg.norm(chi_f))
    if theta > np.pi / 2.:
        theta -= np.pi
    r_dir = np.cross([0, 0, 1], chi_f)
    r_dir = theta * r_dir / np.linalg.norm(r_dir)
    q = quaternion.from_rotation_vector(r_dir)
    
    return q / np.sqrt(q.norm())

def rotation_to_map_to_super_rest_frame(abd, N_itr_max=10, rel_err_tol=1e-12, ell_max=None):
    """Determine the rotation needed to map an abd object to the superrest frame
    through an iterative solve; e.g., compute the transformation needed to align
    the angular momentum charge with the z-axis, transform the abd object,
    and repeat until the transformation converges.

    Note that the angular momentum charge is aligned with either the
    positive or negative z-axis, depending on which it is initially closest to.
    
    """
    
    rotation = quaternion.quaternion(1,0,0,0)
    
    itr = 0
    rel_err = np.inf
    rel_errs = []
    while itr < N_itr_max and not rel_err < rel_err_tol:
        prev_rotation = copy.deepcopy(rotation)
        
        if itr == 0:
            abd_prime = abd.copy()
        else:
            abd_prime = abd.transform(frame_rotation=rotation.components)

        chi_prime = abd_prime.bondi_dimensionless_spin()
        
        rotation = rotation_from_spin_charge(chi_prime, abd_prime.t)*rotation
        
        rel_err = np.sqrt((rotation - prev_rotation).norm() / rotation.norm())
        rel_errs.append(rel_err)
        
        itr += 1
        
    if not itr < N_itr_max:
        print(f"rotation: maximum number of iterations reached; the min error was {min(np.array(rel_errs).flatten())}.")
    else:
        print(f"rotation: tolerance achieved in {itr} iterations!")
        
    return rotation.components, rel_errs

def time_translation(abd, t_0=None):
    """Time translate an abd object. This is necessary because creating a copy
    of an abd object and then changing it's time variable does not change
    the time variable of the waveform variables.

    """
    abd_prime = scri.asymptotic_bondi_data.AsymptoticBondiData(abd.t, abd.ell_max)
    
    abd_prime.sigma = abd.sigma
    abd_prime.psi4 = abd.psi4
    abd_prime.psi3 = abd.psi3
    abd_prime.psi2 = abd.psi2
    abd_prime.psi1 = abd.psi1
    abd_prime.psi0 = abd.psi0
    
    abd_prime.t -= t_0
    
    return abd_prime

def transformations_to_map_to_superrest_frame(self, t_0=0,\
                                              N_itr_maxes={\
                                                           'supertranslation': 10,\
                                                           'com_transformation': 10,\
                                                           'rotation': 10,\
                                                           },\
                                              rel_err_tols={\
                                                            'supertranslation': 1e-12,\
                                                            'com_transformation': 1e-12,\
                                                            'rotation': 1e-12,\
                                                            },\
                                              ell_max=None,
                                              alpha_ell_max=None,
                                              padding_time=100):
    """
    Compute the transformations necessary to map to the superrest frame
    by iteratively minimizing various BMS charges at a certain time.

    This wholly fixes the BMS frame of the abd object up to a
    time translation and a phase rotation.

    Parameters
    ==========
    t_0 : float, optional
        When to map to the superrest frame.
        Default is 0.
    
    N_itr_maxes : dict, optional
        Maximum number of iterations to perform for each transformation.
        Default is
        N_itr_maxes = {
            'supertranslation':   10,
            'com_transformation': 10,
            'rotation':           10
        }.
 
   rel_err_tols : dict, optional
        Error tolerances for each transformation.
        Default is
        N_itr_maxes = {
            'supertranslation':   1e-12,
            'com_transformation': 1e-12,
            'rotation':           1e-12
        }.
        
    ell_max : int, optional
        Maximum ell to use for Grid/SWSH transformations.
        Default is self.ell_max.

    alpha_ell_max : int, optional
        Maximum ell of the supertranslation to use.
        Default is self.ell_max.

    padding_time : float, optional
        Amount by which to pad around t_0 to speed up computations, i.e.,
        distance from t_0 in each direction to be included in self.interpolate(...)
        Default is 100.

    Returns
    -------
    transformations : dict
        Dictionary of transformations and their relative errors whose keys are
            * 'transformations'
                * 'space_translation'
                * 'supertranslation'
                * 'frame_rotation'
                * 'boost_velocity'
            * 'rel_errs'
                (same as above)

    abd_prime : AsymptoticBondiData
        Result of self.transform(...) where the input transformations are
        the transformations found in the transformations dictionary.
    """

    # apply a time translation so that we're mapping
    # to the superrest frame at u = 0
    abd = time_translation(self, t_0)
    
    if ell_max == None:
        ell_max = abd.ell_max
        
    if alpha_ell_max == None:
        alpha_ell_max = ell_max
    
    abd_interp = abd.interpolate(abd.t[np.argmin(abs(abd.t - (-padding_time))):np.argmin(abs(abd.t - (+padding_time))) + 1])
    
    # space_translation
    space_translation, space_rel_errs = com_transformation_to_map_to_super_rest_frame(abd_interp,\
                                                                                      N_itr_max=N_itr_maxes['com_transformation'],\
                                                                                      rel_err_tol=rel_err_tols['com_transformation'],\
                                                                                      ell_max=ell_max,\
                                                                                      space_translation=True,\
                                                                                      boost_velocity=False)
     
    # supertranslation
    abd_prime = abd_interp.transform(space_translation=space_translation)
    
    alpha, alpha_rel_errs = supertranslation_to_map_to_super_rest_frame(abd_prime,\
                                                                        N_itr_max=N_itr_maxes['supertranslation'],\
                                                                        rel_err_tol=rel_err_tols['supertranslation'],\
                                                                        ell_max=ell_max)
    
    alpha[1:4] = sf.vector_as_ell_1_modes(space_translation)
    
    # rotation
    abd_prime = abd_interp.transform(supertranslation=alpha[:LM(alpha_ell_max, alpha_ell_max, 0) + 1])
    
    rotation, rot_rel_errs = rotation_to_map_to_super_rest_frame(abd_prime,\
                                                                 N_itr_max=N_itr_maxes['rotation'],\
                                                                 rel_err_tol=rel_err_tols['rotation'],\
                                                                 ell_max=ell_max)
    
    # com_transformation
    abd_prime = abd_interp.transform(supertranslation=alpha[:LM(alpha_ell_max, alpha_ell_max, 0) + 1],\
                                     frame_rotation=rotation)
    
    CoM_transformation, CoM_rel_errs = com_transformation_to_map_to_super_rest_frame(abd_prime,\
                                                                                     N_itr_max=N_itr_maxes['com_transformation'],\
                                                                                     rel_err_tol=rel_err_tols['com_transformation'],\
                                                                                     ell_max=ell_max,\
                                                                                     space_translation=True,\
                                                                                     boost_velocity=True)
    
    # transform abd
    abd_prime = abd.transform(supertranslation=alpha[:LM(alpha_ell_max, alpha_ell_max, 0) + 1],\
                              frame_rotation=rotation)
    abd_prime = abd_prime.transform(space_translation=CoM_transformation['space_translation'],\
                                    boost_velocity=CoM_transformation['boost_velocity'])

    # undo the initial time translation
    abd_prime = time_translation(abd_prime, -t_0)
        
    alpha[1:4] = 0
    transformations = {\
                       'transformations': {\
                                           'space_translation': space_translation,\
                                           'supertranslation': alpha,\
                                           'frame_rotation': rotation,\
                                           'boost_velocity': CoM_transformation
                                          },
                       'rel_errs': {\
                                    'space_translation': space_rel_errs,\
                                    'supertranslation': alpha_rel_errs,\
                                    'frame_rotation': rot_rel_errs,\
                                    'boost_velocity': CoM_rel_errs
                                    }
                      }  
    
    return transformations, abd_prime
