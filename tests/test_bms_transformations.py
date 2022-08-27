import pytest
import numpy as np

import scri
import quaternion
import spherical_functions as sf
from scri import bms_transformations

ABD = scri.AsymptoticBondiData

def kerr_schild(mass, spin, ell_max=8):
    psi2 = np.zeros(sf.LM_total_size(0, ell_max), dtype=complex)
    psi1 = np.zeros(sf.LM_total_size(0, ell_max), dtype=complex)
    psi0 = np.zeros(sf.LM_total_size(0, ell_max), dtype=complex)

    # In the Moreschi-Boyle convention
    psi2[0] = -sf.constant_as_ell_0_mode(mass)
    psi1[2] = -np.sqrt(2) * (3j * spin / 2) * (np.sqrt((8 / 3) * np.pi))
    psi0[6] = 2 * (3 * spin**2 / mass / 2) * (np.sqrt((32 / 15) * np.pi))

    return psi2, psi1, psi0

def rotation_matrix(q):
    s, x, y, z = q.components
    R_matrix = \
        [[1, 0                          , 0                          ,                           0],\
         [0, 1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * s * z      , 2 * x * z + 2 * s * y      ],\
         [0, 2 * x * y + 2 * s * z      , 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * s * x      ],\
         [0, 2 * x * z - 2 * s * y      , 2 * y * z + 2 * s * x      , 1 - 2 * x ** 2 - 2 * y ** 2]]
    
    return R_matrix

def boost_matrix(v):
    v_norm = np.linalg.norm(v)
    g = 1 / np.sqrt(1 - v_norm ** 2)
    B_matrix = \
        [[g        , -g * v[0]                            , -g * v[1]                            , -g * v[2]                           ],\
         [-g * v[0], 1 + (g - 1) * v[0] ** 2 / v_norm ** 2, (g - 1) * v[0] * v[1] / v_norm ** 2  , (g - 1) * v[0] * v[2] / v_norm ** 2 ],\
         [-g * v[1], (g - 1) * v[1] * v[0] / v_norm ** 2  , 1 + (g - 1) * v[1] ** 2 / v_norm ** 2, (g - 1) * v[1] * v[2] / v_norm ** 2 ],\
         [-g * v[2], (g - 1) * v[2] * v[0] / v_norm ** 2  , (g - 1) * v[2] * v[1] / v_norm ** 2  , 1 + (g - 1) * v[2] **2 / v_norm ** 2]]
    
    return B_matrix

def test_Lorentz_reorder_consistency():
    q = np.quaternion(1, 2, 3, 4).normalized()
    v = np.array([1, 2, 3]) * 1e-2
    L = bms_transformations.Lorentz_transformation(rotation=q.components,
                                                   boost=v,
                                                   order=['rotation','boost'])

    L_reordered = L.reorder(output_order=L.order[::-1])
    
    assert L == L_reordered.reorder(output_order=L.order)

def test_Lorentz_reorder():
    q = np.quaternion(1, 2, 3, 4).normalized()
    v = np.array([1, 2, 3]) * 1e-2
    L = bms_transformations.Lorentz_transformation(rotation=q.components,
                                                   boost=v,
                                                   order=['rotation','boost'])

    L_reordered = L.reorder(output_order=L.order[::-1])

    vq = np.linalg.multi_dot([boost_matrix(v), rotation_matrix(q)])

    L_reordered_matrix = np.linalg.multi_dot([rotation_matrix(L_reordered.rotation), boost_matrix(L_reordered.boost)])
    
    assert np.allclose(vq, L_reordered_matrix)

def test_rotation_inverse():
    q = np.quaternion(1, 2, 3, 4).normalized()
    L = bms_transformations.Lorentz_transformation(rotation=q.components)
    
    assert np.allclose(q.inverse().components,
                       L.inverse().rotation.components)

def test_boost_inverse():
    v = np.array([1, 2, 3]) * 1e-2
    L = bms_transformations.Lorentz_transformation(boost=v)
    
    assert np.allclose(-v,
                       L.inverse().boost)

def test_Lorentz_inverse_composition_consistency():
    q = np.quaternion(1, 2, 3, 4).normalized()
    v = np.array([1, 2, 3]) * 1e-2
    L = bms_transformations.Lorentz_transformation(rotation=q.components,
                                                   boost=v,
                                                   order=['rotation','boost'])
    L_inv = L.inverse()
    
    assert L.compose(L_inv).is_identity()
    assert L_inv.compose(L).is_identity()

def test_Lorentz_inverse():
    q = np.quaternion(1, 2, 3, 4).normalized()
    v = np.array([1, 2, 3]) * 1e-2
    L = bms_transformations.Lorentz_transformation(rotation=q.components,
                                                   boost=v,
                                                   order=['rotation','boost'])
    L_inv = bms_transformations.Lorentz_transformation(rotation=q.inverse().components,
                                                       boost=-v,
                                                       order=['boost','rotation'])
    
    assert L_inv == L.inverse(output_order=['boost','rotation'])

def test_rotation_composition():
    q1 = np.quaternion(1, 2, 3, 4).normalized()
    q2 = np.quaternion(5, -6, 7, -8).normalized()
    L1 = bms_transformations.Lorentz_transformation(rotation=q1.components)
    L2 = bms_transformations.Lorentz_transformation(rotation=q2.components)

    q1q2 = q2 * q1
    q2q1 = q1 * q2

    L1L2 = L1.compose(L2)
    L2L1 = L2.compose(L1)
    
    assert np.allclose(q1q2.components, L1L2.rotation.components)
    assert np.allclose(q2q1.components, L2L1.rotation.components)
    
def test_boost_composition():
    v1 = np.array([1, 2, 3]) * 1e-2
    v2 = np.array([-4, 5, -6]) * 1e-2

    L1 = bms_transformations.Lorentz_transformation(boost=v1)
    L2 = bms_transformations.Lorentz_transformation(boost=v2)

    # the following Wigner rotation calculation
    # is from Eqs. (65) - (70) of arXiv:1102.2001
    g1 = 1 / np.sqrt(1 - np.linalg.norm(v1) ** 2)
    g2 = 1 / np.sqrt(1 - np.linalg.norm(v2) ** 2)
    g12 = g1 * g2 * (1 + np.dot(v1, v2))
    v1v2 = (v1 + g2 * v2 + (g2 - 1) * np.dot(v1, v2) * v2 / np.linalg.norm(v2) ** 2) / (g2 * (1 + np.dot(v1, v2)))
    v2v1 = (v2 + g1 * v1 + (g1 - 1) * np.dot(v2, v1) * v1 / np.linalg.norm(v1) ** 2) / (g1 * (1 + np.dot(v2, v1)))

    theta = np.arccos((1 + g1 + g2 + g12) ** 2 / ((1 + g1) * (1 + g2) * (1 + g12)) - 1)
    v1v2_q = np.quaternion(np.cos(theta / 2), *(np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2)) * np.sin(theta / 2)))
    v2v1_q = np.quaternion(np.cos(theta / 2), *(np.cross(v2, v1) / np.linalg.norm(np.cross(v2, v1)) * np.sin(theta / 2)))
    
    L1L2 = L1.compose(L2)
    L2L1 = L2.compose(L1)
    
    assert np.allclose(v1v2, L1L2.boost)
    assert np.allclose(v1v2_q.components, L1L2.rotation.components)
    assert np.allclose(v2v1, L2L1.boost)
    assert np.allclose(v2v1_q.components, L2L1.rotation.components)

def test_Lorentz_composition():
    q1 = np.quaternion(1, 2, 3, 4).normalized()
    v1 = np.array([1, 2, 3]) * 1e-2
    q2 = np.quaternion(5, -6, 7, -8).normalized()
    v2 = np.array([-4, 5, -6]) * 1e-2

    L1 = bms_transformations.Lorentz_transformation(rotation=q1.components,
                                                    boost=v1)
    L2 = bms_transformations.Lorentz_transformation(rotation=q2.components,
                                                    boost=v2)

    q1v1q2v2 = np.linalg.multi_dot([boost_matrix(v2), rotation_matrix(q2), boost_matrix(v1), rotation_matrix(q1)])
    q2v2q1v1 = np.linalg.multi_dot([boost_matrix(v1), rotation_matrix(q1), boost_matrix(v2), rotation_matrix(q2)])
        
    L1L2 = L1.compose(L2)
    L1L2_matrix = np.linalg.multi_dot([boost_matrix(L1L2.boost), rotation_matrix(L1L2.rotation)])

    L2L1 = L2.compose(L1)
    L2L1_matrix = np.linalg.multi_dot([boost_matrix(L2L1.boost), rotation_matrix(L2L1.rotation)])
    
    assert np.allclose(q1v1q2v2, L1L2_matrix)
    assert np.allclose(q2v2q1v1, L2L1_matrix)

def test_BMS_reorder_consistency():
    S = np.array([1, 2 + 4j, 3, -2 + 4j, 7 - 5j, -3 - 2j, 4, 3 - 2j, 7 + 5j]) * 1e-3
    q = np.quaternion(1, 2, 3, 4).normalized()
    v = np.array([1, 2, 3]) * 1e-4

    BMS = bms_transformations.BMS_transformation(supertranslation=S,
                                                 rotation=q,
                                                 boost=v)

    check1 = BMS == BMS.reorder(['supertranslation','rotation','boost']).reorder(BMS.order)
    check2 = BMS == BMS.reorder(['rotation','supertranslation','boost']).reorder(BMS.order)
    check3 = BMS == BMS.reorder(['rotation','boost','supertranslation']).reorder(BMS.order)
    check4 = BMS == BMS.reorder(['supertranslation','boost','rotation']).reorder(BMS.order)
    check5 = BMS == BMS.reorder(['boost','supertranslation','rotation']).reorder(BMS.order)
    check6 = BMS == BMS.reorder(['boost','rotation','supertranslation']).reorder(BMS.order)

    assert check1
    assert check2
    assert check3
    assert check4
    assert check5
    assert check6
    
def test_BMS_inverse_composition_consistency():
    S = np.array([1, 2 + 4j, 3, -2 + 4j, 7 - 5j, -3 - 2j, 4, 3 - 2j, 7 + 5j]) * 1e-3
    q = np.quaternion(1, 2, 3, 4).normalized()
    v = np.array([1, 2, 3]) * 1e-4

    BMS = bms_transformations.BMS_transformation(supertranslation=S,
                                                 rotation=q,
                                                 boost=v)
    BMS_inv = BMS.inverse()

    assert BMS.compose(BMS_inv).is_identity()
    assert BMS_inv.compose(BMS).is_identity()

#def test_BMS_inverse():
#    mass = 2.0
#    spin = 0.456
#    ell_max = 8
#    u = np.linspace(-100, 100, num=5000)
#    psi2, psi1, psi0 = kerr_schild(mass, spin, ell_max)
#    abd = ABD.from_initial_values(u, ell_max=ell_max, psi2=psi2, psi1=psi1)
#
#    S = np.array([1, 2 + 4j, 3, -2 + 4j, 7 - 5j, -3 - 2j, 4, 3 - 2j, 7 + 5j]) * 1e-3
#    q = np.quaternion(1, 2, 3, 4).normalized()
#    v = np.array([1, 2, 3]) * 1e-4
#
#    BMS = bms_transformations.BMS_transformation(supertranslation=S,
#                                                 rotation=q,
#                                                 boost=v,
#                                                 order=['supertranslation','rotation','boost'])
#
#    BMS_inv = BMS.inverse(output_order=['supertranslation','rotation','boost'])
#
#    abd_prime = abd.transform(supertranslation=BMS.supertranslation,
#                              frame_rotation=BMS.rotation.components,
#                              boost_velocity=BMS.boost)
#
#    abd_check = abd_prime.transform(supertranslation=BMS_inv.supertranslation,
#                                    frame_rotation=BMS_inv.rotation.components,
#                                    boost_velocity=BMS_inv.boost)
#
#    assert np.allclose(np.array([abd.psi2,abd.psi1,abd.psi0]), np.array([abd_check.psi2, abd_check.psi1, abd_check.psi2]))

    
