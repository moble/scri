import copy
import scipy
import spinsfast
from scri.asymptotic_bondi_data.transformations import *
import numpy as np
import spherical_functions as sf

def _process_transformation_kwargs(input_ell_max, **kwargs):
    original_kwargs = kwargs.copy()

    # Build the supertranslation and spacetime_translation arrays
    supertranslation = np.zeros((4,), dtype=complex)  # For now; may be resized below
    ell_max_supertranslation = 1  # For now; may be increased below
    if "supertranslation" in kwargs:
        supertranslation = np.array(kwargs.pop("supertranslation"), dtype=complex)
        if supertranslation.dtype != "complex" and supertranslation.size > 0:
            # I don't actually think this can ever happen...
            raise TypeError(
                "Input argument `supertranslation` should be a complex array with size>0.  "
                f"Got a {supertranslation.dtype} array of shape {supertranslation.shape}"
            )
        # Make sure the array has size at least 4, by padding with zeros
        if supertranslation.size <= 4:
            supertranslation = np.lib.pad(
                supertranslation, (0, 4 - supertranslation.size), "constant", constant_values=(0.0,)
            )
        # Check that the shape is a possible array of scalar modes with complete (ell,m) data
        ell_max_supertranslation = int(np.sqrt(len(supertranslation))) - 1
        if (ell_max_supertranslation + 1) ** 2 != len(supertranslation):
            raise ValueError(
                "Input supertranslation parameter must contain modes from ell=0 up to some ell_max, "
                "including\n           all relevant m modes in standard order (see `spherical_functions` "
                "documentation for details).\n           Thus, it must be an array with length given by a "
                "perfect square; its length is {len(supertranslation)}"
            )
        # Check that the resulting supertranslation will be real
        for ell in range(ell_max_supertranslation + 1):
            for m in range(ell + 1):
                i_pos = sf.LM_index(ell, m, 0)
                i_neg = sf.LM_index(ell, -m, 0)
                a = supertranslation[i_pos]
                b = supertranslation[i_neg]
                if abs(a - (-1.0) ** m * b.conjugate()) > 3e-16 + 1e-15 * abs(b):
                    raise ValueError(
                        f"\nsupertranslation[{i_pos}]={a}  # (ell,m)=({ell},{m})\n"
                        + "supertranslation[{}]={}  # (ell,m)=({},{})\n".format(i_neg, b, ell, -m)
                        + "Will result in an imaginary supertranslation."
                    )
    spacetime_translation = np.zeros((4,), dtype=float)
    spacetime_translation[0] = sf.constant_from_ell_0_mode(supertranslation[0]).real
    spacetime_translation[1:4] = -sf.vector_from_ell_1_modes(supertranslation[1:4]).real
    if "spacetime_translation" in kwargs:
        st_trans = np.array(kwargs.pop("spacetime_translation"), dtype=float)
        if st_trans.shape != (4,) or st_trans.dtype != "float":
            raise TypeError(
                "\nInput argument `spacetime_translation` should be a float array of shape (4,).\n"
                "Got a {} array of shape {}.".format(st_trans.dtype, st_trans.shape)
            )
        spacetime_translation = st_trans[:]
        supertranslation[0] = sf.constant_as_ell_0_mode(spacetime_translation[0])
        supertranslation[1:4] = sf.vector_as_ell_1_modes(-spacetime_translation[1:4])
    if "space_translation" in kwargs:
        s_trans = np.array(kwargs.pop("space_translation"), dtype=float)
        if s_trans.shape != (3,) or s_trans.dtype != "float":
            raise TypeError(
                "\nInput argument `space_translation` should be an array of floats of shape (3,).\n"
                "Got a {} array of shape {}.".format(s_trans.dtype, s_trans.shape)
            )
        spacetime_translation[1:4] = s_trans[:]
        supertranslation[1:4] = sf.vector_as_ell_1_modes(-spacetime_translation[1:4])
    if "time_translation" in kwargs:
        t_trans = kwargs.pop("time_translation")
        if not isinstance(t_trans, float):
            raise TypeError("Input argument `time_translation` should be a single float.  " f"Got {t_trans}")
        spacetime_translation[0] = t_trans
        supertranslation[0] = sf.constant_as_ell_0_mode(spacetime_translation[0])

    # Decide on the number of points to use in each direction.  A nontrivial supertranslation will
    # introduce power in higher modes, so for best accuracy, we need to account for that.  But we'll
    # make it a firm requirement to have enough points to capture the original waveform, at least
    working_ell_max = 2 * input_ell_max + ell_max_supertranslation
    if working_ell_max < input_ell_max:
        raise ValueError(f"working_ell_max={working_ell_max} is too small; it must be at least ell_max={input_ell_max}")

    # Get the rotor for the frame rotation
    rotation = kwargs.pop("rotation", [1, 0, 0, 0])
    if type(rotation) is not np.quaternion:
        rotation = np.quaternion(*np.array(rotation, dtype=float))
    if rotation.abs() < 3e-16:
        raise ValueError(f"rotation={rotation} should be a single unit quaternion")
    rotation = rotation.normalized()

    # Get the boost velocity vector
    boost = np.array(kwargs.pop("boost", [0.0] * 3), dtype=float)
    beta = np.linalg.norm(boost)
    if boost.dtype != float or boost.shape != (3,) or beta >= 1.0:
        raise ValueError(
            f"Input boost=`{boost}` should be a 3-vector with " "magnitude strictly less than 1.0"
        )

    return rotation, boost, supertranslation

def fourvec_to_spin_matrix(fourvec):
    """Inner product of a four vector and the Pauli matrices,
    as defined by Eq. (1.2.39) of Spinors and Spacetime Vol. 1

    Parameters
    ----------
    fourvec: float array of length 4.
    """
    # Maybe better to define Pauli matrices?
    a = np.cos(fourvec[0] / 2) + 1j * fourvec[3] * np.sin(fourvec[0] / 2)
    b = (-fourvec[2] + 1j * fourvec[1]) * np.sin(fourvec[0] / 2)
    c = (+fourvec[2] + 1j * fourvec[1]) * np.sin(fourvec[0] / 2)
    d = np.cos(fourvec[0] / 2) - 1j * fourvec[3] * np.sin(fourvec[0] / 2)

    return np.array([[a, b], [c, d]])

def Lorentz_to_spin_matrix(Lorentz):
    """Convert a Lorentz transformation to a spin matrix.
    
    Parameters
    ----------
    Lorentz: Lorentz_transformation
    """
    
    psi = 2 * np.arctan2(np.linalg.norm(Lorentz.rotation.components[1:]),
                         Lorentz.rotation.components[0])
    if psi == 0:
        rotation_vec_hat = np.array([0, 0, 0])
    else:
        rotation_vec_hat = Lorentz.rotation.components[1:] \
            / np.linalg.norm(Lorentz.rotation.components[1:])

    chi = 1j * np.arctanh(np.linalg.norm(Lorentz.boost))
    if chi == 0:
        boost_vec_hat = np.array([0, 0, 0])
    else:
        boost_vec_hat = Lorentz.boost \
            / np.linalg.norm(Lorentz.boost)
        
    # Compute product of Lorentz vector and Pauli matrices
    rotation_spin_matrix = fourvec_to_spin_matrix([psi, *rotation_vec_hat])
    boost_spin_matrix = fourvec_to_spin_matrix([chi, *boost_vec_hat])

    if Lorentz.order.index('rotation') < Lorentz.order.index('boost'):
        return np.linalg.multi_dot([boost_spin_matrix, rotation_spin_matrix])
    else:
        return np.linalg.multi_dot([rotation_spin_matrix, boost_spin_matrix])
    
def pure_spin_matrix_to_Lorentz(A, is_rotation=None, tol=1e-14):
    """Convert a pure spin matrix to rotation or a boost.
    
    Parameters
    ----------
    A: 2 by 2 complex array
        2 by 2 array corresponding to the input spin matrix.
    is_rotation: bool
        Whether or not the spin matrix should correspond to
        a Lorentz rotation or a Lorentz boost. Defaults to None,
        i.e., this will be figured out based on the matrix's values.
    tol: float
        Tolerance for figuring out whether the spin matrix
        corresponds to a Lorentz rotation or a Lorentz boost.
    """
    logA = scipy.linalg.logm(A)
    
    # compute the Lorentz vector
    nvec = np.array([(logA[1,0] + logA[0,1]) / 2,
                     (logA[1,0] - logA[0,1]) / 2,
                     (logA[0,0] - logA[1,1]) / 2])
    
    nvec_re = np.array([nvec[0].imag, nvec[1].real, nvec[2].imag])
    nvec_im = np.array([-nvec[0].real, nvec[1].imag, -nvec[2].real])
    
    # figure out if this spin matrix is a rotation or a boost
    if is_rotation is None:
        if np.linalg.norm(nvec_im) < tol:
            nvec = nvec_re
            is_rotation = True
        elif np.linalg.norm(nvec_re) < tol:
            nvec = nvec_im
            is_rotation = False
        else:
            spin_matrix_to_Lorentz(A)
    else:
        if is_rotation:
            nvec = nvec_re
        else:
            nvec = nvec_im
            
    psi = 2 * np.linalg.norm(nvec)
    if psi == 0:
        nvec_hat = np.array([0,0,1])
    else:
        nvec_hat = nvec / np.linalg.norm(nvec)
        
    if is_rotation:
        return np.quaternion(np.cos(psi / 2), *nvec_hat*np.sin(psi / 2)).components
    else:
        return np.tanh(psi) * nvec_hat
    
def spin_matrix_to_Lorentz(A, output_order=['rotation','boost']):
    """Convert a spin matrix to a Lorentz transformation.

    This uses SVD to decompose the spin matrix into a
    spin matrix describing a Lorentz boost and a
    spin matrix describing a Lorentz rotation.
    
    Parameters
    ----------
    A: 2 by 2 complex array
        2 by 2 array corresponding to the input spin matrix.
    output_order: list
        Order in which rotation and boost should be applied.
    """
    if np.allclose(A, np.zeros_like(A)):
        return Lorentz_transformation()
    
    u, s, vh = np.linalg.svd(A)
    
    try:
        rotation_idx = output_order.index('rotation')
    except:
        rotation_idx = np.inf
    try:
        boost_idx = output_order.index('boost')
    except:
        boost_idx = np.inf
        
    if rotation_idx < boost_idx:
        rotation_matrix = np.linalg.multi_dot([u, vh])
        boost_matrix = np.linalg.multi_dot([u, np.diag(s), u.conj().T])
    else:
        rotation_matrix = np.linalg.multi_dot([u, vh])
        boost_matrix = np.linalg.multi_dot([vh.conj().T, np.diag(s), vh])
    rotation = pure_spin_matrix_to_Lorentz(rotation_matrix, is_rotation=True)
    boost = pure_spin_matrix_to_Lorentz(boost_matrix, is_rotation=False)
    return Lorentz_transformation(rotation=rotation, boost=boost, order=output_order)

def compute_conformal_factor(Lorentz, n_theta, n_phi):
    """Compute the conformal factor.

    Note that this is the inverse of Eq. (4.6) of 10.1063/1.1705135.
    """
    thetaprm_phiprm = sf.theta_phi(n_theta, n_phi)

    [[a, b], [c, d]] = Lorentz_to_spin_matrix(Lorentz)

    conformal = np.zeros((n_theta, n_phi), dtype=complex)
    for j in range(n_theta):
        for k in range(n_phi):
            thetaprm_j, phiprm_k = thetaprm_phiprm[j, k]
            if thetaprm_j == 0:
                z = np.inf
                conformal[j, k] = 1/(np.abs(a)**2+np.abs(c)**2)
            else:
                z = np.exp(1j*phiprm_k)/np.tan(thetaprm_j/2)
                conformal[j, k] = (1+np.abs(z)**2)/(np.abs(a*z+b)**2+np.abs(c*z+d)**2)
            
    return conformal

def nCr(n, r):
    """n choose r.
    """
    if n >= 0 and r >= 0 and n - r >= 0:
        return np.math.factorial(n)/(np.math.factorial(r)*np.math.factorial(n-r))
    else:
        return 0

def A_matrix(L, M1, M2, N1, N2, spin_matrix):
    """Compute the matrix that transforms Eq. (4.11) of 10.1063/1.1705135
    to a Lorentz transformed frame.
    """
    A_value = 0
    [[a, b], [c, d]] = spin_matrix
    for i1 in range(L - M1 + 1):
        for i2 in range(M1 + 1):
            if not (i1 + i2) == N1:
                continue
            for i3 in range(L - M2 + 1):
                for i4 in range(M2 + 1):
                    if not (i3 + i4) == N2:
                        continue
                    A_value += nCr(L-M1,i1)*nCr(M1,i2)*nCr(L-M2,i3)*nCr(M2,i4)\
                        *a**(L-M1-i1)*np.conjugate(a)**(L-M2-i3)*b**(i1)*np.conjugate(b)**(i3)\
                        *c**(M1-i2)*np.conjugate(c)**(M2-i4)*d**(i2)*np.conjugate(d)**(i4)
    
    return A_value
    
def B_matrix(Lmax, L, M, M1, M2):
    """Compute Eq. (4.14) of 10.1063/1.1705135.
    """
    if M1 + M != M2:
        return 0
    else:
        ALM = (-1)**(L)*np.sqrt(np.math.factorial(L+M)*np.math.factorial(L-M)*(2*L+1)/(4*np.pi))
        sum_contribution = 0
        for p in range(min(Lmax-M1, L, L + M) + 1):
            sum_contribution += (-1)**(p-M)*nCr(L,p)*nCr(L,p-M)*nCr(Lmax-L,Lmax-M1-p)
        return ALM/np.sqrt(float(np.math.factorial(L)*np.math.factorial(L)))*sum_contribution
    
def transform_Y_LM(Y_LM, Lorentz, ell_max=None):
    """Apply a Lorentz transformation to a spherical harmonic.

    Parameters
    ----------
    f_LM: ndarray, dtype=complex
        Ylms to be transformed.
    Lorentz: Lorentz_transformation
        Lorentz transformation to be used to transform the function.
    ell_max: int
        Maximum ell to use when expressing functions via coordinates on the two-sphere.
        Defaults to the ell_max of the Lorentz transformation.
    """
    if ell_max is None:
        ell_max = Lorentz.ell_max
    
    spin_matrix = Lorentz_to_spin_matrix(Lorentz)

    from_zprime_to_z = []
    from_SH_expansion = []
    for N1 in range(Y_LM[0] + 1):
        for N2 in range(Y_LM[0] + 1):
            factor = 0
            for M1 in range(Y_LM[0] + 1):
                for M2 in range(Y_LM[0] + 1):
                    factor += A_matrix(Y_LM[0], M1, M2, N1, N2, spin_matrix)\
                        *B_matrix(Y_LM[0], Y_LM[0], Y_LM[1], M1, M2)
            from_zprime_to_z.append(factor)

            factors = []
            for L in range(0, Y_LM[0] + 1):
                for M in range(-L, L + 1):
                    factors.append(B_matrix(Y_LM[0], L, M, N1, N2))
            from_SH_expansion.append(factors)
            
    n_theta = 2 * ell_max + 1; n_phi = n_theta;
            
    f_LM = np.matmul(np.linalg.inv(np.array(from_SH_expansion)),np.array(from_zprime_to_z))
    f_LM_padded = np.pad(f_LM, (0,(ell_max + 1)**2 - f_LM.size))
    f_LM_S2 = spinsfast.salm2map(f_LM_padded, 0, ell_max, n_theta, n_phi)
    
    K_S2 = compute_conformal_factor(Lorentz, n_theta, n_phi)

    result_S2 = K_S2**(Y_LM[0]) * f_LM_S2
    
    return spinsfast.map2salm(result_S2, 0, ell_max)

def transform_function(f_LM, Lorentz, ell_max=None, tol=1e-15):
    """Apply a Lorentz transformation to a spin-weight 0 funtion, where the function
    is provided via it's spherical harmonic modes, the Ylms.

    Parameters
    ----------
    f_LM: ndarray, dtype=complex
        Ylms to be transformed.
    Lorentz: Lorentz_transformation
        Lorentz transformation to be used to transform the function.
    ell_max: int
        Maximum ell to use when expressing functions via coordinates on the two-sphere.
        Defaults to the ell_max of the Lorentz transformation.
    tol: float
        Tolerance to use when rewriting the supertranslation's transformed Ylms
        in terms of the untransformed Ylms. Defaults to 1e-15.
    """
    if ell_max is None:
        ell_max = Lorentz.ell_max

    f_LM_prime = np.zeros((ell_max + 1)**2, dtype=complex)
    for L in range(0, int(np.sqrt(f_LM.size) - 1) + 1):
        for M in range(-L, L+1):
            if abs(f_LM[sf.LM_index(L, M, 0)]) < tol:
                continue
            f_LM_prime += f_LM[sf.LM_index(L, M, 0)] * transform_Y_LM((L, M), Lorentz, ell_max)

    return f_LM_prime

def transform_supertranslation(S, Lorentz, ell_max=None, tol=1e-15):
    """Apply a Lorentz transformation to a supertranslation and multiply by
    one over the conformal factor. This produces the supertranslation the appears
    when commuting a supertranslation through a Lorentz transformation.

    The Lorentz transformation is the transformation appearing on
    the RHS of the product, i.e., S' = L^{-1} S L.

    Parameters
    ----------
    S: ndarray, dtype=complex
        supertranslation to be transformed.
    Lorentz: Lorentz_transformation
        Lorentz transformation to be used to transform the supertranslation.
    ell_max: int
        Maximum ell to use when expressing functions via coordinates on the two-sphere.
    tol: float
        Tolerance to use when rewriting the supertranslation's transformed Ylms
        in terms of the untransformed Ylms. Defaults to 1e-15.
    """
    if ell_max is None:
        ell_max = Lorentz.ell_max
    n_theta = 2 * ell_max + 1; n_phi = n_theta;
        
    transformed_S = transform_function(S, Lorentz, ell_max, tol)
    
    transformed_S_S2 = spinsfast.salm2map(transformed_S, 0, ell_max, n_theta, n_phi)
    K_S2 = compute_conformal_factor(Lorentz, n_theta, n_phi)

    S_prime_S2 = K_S2**(-1)*transformed_S_S2

    return spinsfast.map2salm(S_prime_S2, 0, ell_max)
    
class Lorentz_transformation:
    def __init__(self, **kwargs):
        self.ell_max = copy.deepcopy(kwargs.pop("ell_max", 30))
        (
            rotation,
            boost,
            supertranslation,
        ) = _process_transformation_kwargs(self.ell_max, **kwargs)
        self.rotation = copy.deepcopy(rotation)
        self.boost = copy.deepcopy(boost)

        self.order = copy.deepcopy(kwargs.pop("order", ['rotation','boost']))
        if 'supertranslation' in self.order:
            del self.order[self.order.index('supertranslation')]
        for transformation in ['rotation','boost']:
            if transformation not in self.order:
                self.order.append(transformation)

    def __repr__(self):
        Lorentz_output = {}
        for transformation in self.order:
            if transformation == 'rotation':
                Lorentz_output[transformation] = self.rotation
            elif transformation == 'boost':
                Lorentz_output[transformation] = self.boost

        return f'Lorentz_transformation(\n\t{self.order[0]}={Lorentz_output[self.order[0]]}\n\t{self.order[1]}={Lorentz_output[self.order[1]]}\n)'

    def __str__(self):
        Lorentz_output = {}
        for transformation in self.order:
            if transformation == 'rotation':
                Lorentz_output[transformation] = self.rotation
            elif transformation == 'boost':
                Lorentz_output[transformation] = self.boost

        return f'Lorentz_transformation(\n\t{self.order[0]}={Lorentz_output[self.order[0]]}\n\t{self.order[1]}={Lorentz_output[self.order[1]]}\n)'
    
    def copy(self):
        return Lorentz_transformation(rotation=self.rotation,
                                      boost=self.boost,
                                      order=self.order,
                                      ell_max=self.ell_max)

    def reorder(self, output_order):
        """Reorder a Lorentz transformation.
        """
        if not ('rotation' in output_order and\
                'boost' in output_order):
            raise ValueError('Not enough transformations')
        
        if self.order == output_order:
            return self
        else:
            L_spin_matrix = Lorentz_to_spin_matrix(self)
            L_reordered = spin_matrix_to_Lorentz(L_spin_matrix, output_order=output_order)
            return L_reordered

    def is_identity(self, verbose=False):
        """Check if a BMS transformation is the identity element
        """
        rotation_is_identity = np.allclose(self.rotation.components, [1,0,0,0])
        boost_is_identity = np.allclose(self.boost, [0,0,0])
        if verbose:
            print(f"rotation is identity: {rotation_is_identity}")
            print(f"boost is identity: {boost_is_identity}")
        return rotation_is_identity and boost_is_identity

    def inverse(self, output_order=None):
        """Compute the inverse of a Lorentz transformation.
        """
        if output_order is None:
            output_order = self.order[::-1]
        
        L_spin_matrix = Lorentz_to_spin_matrix(self)
        L_spin_matrix_inverse = np.linalg.inv(L_spin_matrix)
        L_inverse = spin_matrix_to_Lorentz(L_spin_matrix_inverse, output_order=output_order)
        return L_inverse

    def __eq__(self, other):
        rotation_eq = np.allclose(self.rotation.components, other.rotation.components)
        boost_eq = np.allclose(self.boost, other.boost)
        return rotation_eq and boost_eq
            
    def compose(self, other, output_order=['rotation','boost']):
        """Compose two Lorentz transformations.
        
        Parameters
        ----------
        other: Lorentz_transformation
            2nd Lorentz transformation to be applied.
        output_order: list
            Order in which rotation and boost should be applied.
        """
        L1_spin_matrix = Lorentz_to_spin_matrix(self)
        L2_spin_matrix = Lorentz_to_spin_matrix(other)
        
        Ls_spin_matrix = np.linalg.multi_dot([L2_spin_matrix, L1_spin_matrix])
    
        return spin_matrix_to_Lorentz(Ls_spin_matrix, output_order=output_order)

    def conjugate(self, other, output_order=['rotation','boost']):
        """Compute the conjugate of two Lorentz transformations.
        
        Parameters
        ----------
        other: Lorentz_transformation
            2nd Lorentz transformation to be applied.
        output_order: list
            Order in which rotation and boost should be applied.
        """
        L1_spin_matrix = Lorentz_to_spin_matrix(self)
        L2_spin_matrix = Lorentz_to_spin_matrix(other)
        
        conjugate_spin_matrix = np.linalg.multi_dot([L2_spin_matrix, L1_spin_matrix])\
            - np.linalg.multi_dot([L1_spin_matrix, L2_spin_matrix])
    
        return spin_matrix_to_Lorentz(conjugate_spin_matrix, output_order=output_order)
    
class BMS_transformation:
    def __init__(self, **kwargs):
        self.ell_max = copy.deepcopy(kwargs.pop("ell_max", 30))
        (
            rotation,
            boost,
            supertranslation,
        ) = _process_transformation_kwargs(self.ell_max, **kwargs)
        self.rotation = copy.deepcopy(rotation)
        self.boost = copy.deepcopy(boost)
        self.supertranslation = np.pad(supertranslation, (0, (self.ell_max + 1)**2 - supertranslation.size))
        
        self.order = copy.deepcopy(kwargs.pop("order", ['supertranslation','rotation','boost']))
        for transformation in ['supertranslation','rotation','boost']:
            if transformation not in self.order:
                self.order.append(transformation)
        
    def __repr__(self):
        BMS_output = {}
        for transformation in self.order:
            if transformation == 'rotation':
                BMS_output[transformation] = self.rotation
            elif transformation == 'boost':
                BMS_output[transformation] = self.boost
            elif transformation == 'supertranslation':
                BMS_output[transformation] = self.supertranslation[:9]

        return f'BMS_transformation(\n\t{self.order[0]}={BMS_output[self.order[0]]}\n\t{self.order[1]}={BMS_output[self.order[1]]}\n\t{self.order[2]}={BMS_output[self.order[2]]}\n)'

    def __str__(self):
        BMS_output = {}
        for transformation in self.order:
            if transformation == 'rotation':
                BMS_output[transformation] = self.rotation
            elif transformation == 'boost':
                BMS_output[transformation] = self.boost
            elif transformation == 'supertranslation':
                BMS_output[transformation] = self.supertranslation[:9]

        return f'BMS_transformation(\n\t{self.order[0]}={BMS_output[self.order[0]]}\n\t{self.order[1]}={BMS_output[self.order[1]]}\n\t{self.order[2]}={BMS_output[self.order[2]]}\n)'
    
    def copy(self):
        return BMS_transformation(rotation=self.rotation,
                                  boost=self.boost,
                                  supertranslation=self.supertranslation,
                                  order=self.order,
                                  ell_max=self.ell_max)
        
    def reorder(self, output_order):
        """Reorder a BMS transformation.
        """
        if not ('supertranslation' in output_order and\
                'rotation' in output_order and\
                'boost' in output_order):
            raise ValueError('Not enough transformations')

        # There's probably a better way to do this, e.g., using the indexes
        # to figure out what transformations are moved through others.
        # But this is easy enough and a resonable first draft.

        # Map to normal order
        normal_order = ['supertranslation','rotation','boost']
        if self.order == normal_order:
            BMS_normal_order = self.copy()
        elif self.order == ['rotation','supertranslation','boost']:
            S_prime = transform_supertranslation(self.supertranslation,
                                                 Lorentz_transformation(rotation=self.rotation,
                                                                        ell_max=self.ell_max,
                                                                        order=self.order))
            
            BMS_normal_order = BMS_transformation(rotation=self.rotation,
                                                  boost=self.boost,
                                                  supertranslation=S_prime,
                                                  ell_max=self.ell_max,
                                                  order=normal_order)
        elif self.order == ['rotation','boost','supertranslation']:
            S_prime = transform_supertranslation(self.supertranslation,
                                                 Lorentz_transformation(rotation=self.rotation,
                                                                        boost=self.boost,
                                                                        ell_max=self.ell_max,
                                                                        order=self.order))
            
            BMS_normal_order = BMS_transformation(rotation=self.rotation,
                                                  boost=self.boost,
                                                  supertranslation=S_prime,
                                                  ell_max=self.ell_max,
                                                  order=normal_order)
        elif self.order == ['supertranslation','boost','rotation']:
            L_prime = Lorentz_transformation(rotation=self.rotation,
                                             boost=self.boost,
                                             ell_max=self.ell_max,
                                             order=self.order).reorder(normal_order)
            
            BMS_normal_order = BMS_transformation(rotation=L_prime.rotation,
                                                  boost=L_prime.boost,
                                                  supertranslation=self.supertranslation,
                                                  ell_max=self.ell_max,
                                                  order=normal_order)
        elif self.order == ['boost','supertranslation','rotation']:
            L_prime = Lorentz_transformation(rotation=self.rotation,
                                             boost=self.boost,
                                             ell_max=self.ell_max,
                                             order=self.order).reorder(normal_order)

            S_prime = transform_supertranslation(self.supertranslation,
                                                 Lorentz_transformation(boost=self.boost,
                                                                        ell_max=self.ell_max,
                                                                        order=self.order))
            
            BMS_normal_order = BMS_transformation(rotation=L_prime.rotation,
                                                  boost=L_prime.boost,
                                                  supertranslation=S_prime,
                                                  ell_max=self.ell_max,
                                                  order=normal_order)
        elif self.order == ['boost','rotation','supertranslation']:
            L_prime = Lorentz_transformation(rotation=self.rotation,
                                             boost=self.boost,
                                             ell_max=self.ell_max,
                                             order=self.order).reorder(normal_order)

            S_prime = transform_supertranslation(self.supertranslation,
                                                 L_prime)
            
            BMS_normal_order = BMS_transformation(rotation=L_prime.rotation,
                                                  boost=L_prime.boost,
                                                  supertranslation=S_prime,
                                                  ell_max=self.ell_max,
                                                  order=normal_order)

        # Map to output order
        if output_order == normal_order:
            BMS_reordered = BMS_normal_order.copy()
        elif output_order == ['rotation','supertranslation','boost']:
            S_prime = transform_supertranslation(BMS_normal_order.supertranslation,
                                                 Lorentz_transformation(rotation=BMS_normal_order.rotation,
                                                                        ell_max=BMS_normal_order.ell_max,
                                                                        order=BMS_normal_order.order).inverse())
            
            BMS_reordered = BMS_transformation(rotation=BMS_normal_order.rotation,
                                               boost=BMS_normal_order.boost,
                                               supertranslation=S_prime,
                                               ell_max=BMS_normal_order.ell_max,
                                               order=output_order)
        elif output_order == ['rotation','boost','supertranslation']:
            S_prime = transform_supertranslation(BMS_normal_order.supertranslation,
                                                 Lorentz_transformation(rotation=BMS_normal_order.rotation,
                                                                        boost=BMS_normal_order.boost,
                                                                        ell_max=BMS_normal_order.ell_max,
                                                                        order=BMS_normal_order.order).inverse())
            
            BMS_reordered = BMS_transformation(rotation=BMS_normal_order.rotation,
                                               boost=BMS_normal_order.boost,
                                               supertranslation=S_prime,
                                               ell_max=BMS_normal_order.ell_max,
                                               order=output_order)
        elif output_order == ['supertranslation','boost','rotation']:
            L_prime = Lorentz_transformation(rotation=BMS_normal_order.rotation,
                                             boost=BMS_normal_order.boost,
                                             ell_max=BMS_normal_order.ell_max,
                                             order=BMS_normal_order.order).reorder(output_order)
            
            BMS_reordered = BMS_transformation(rotation=L_prime.rotation,
                                               boost=L_prime.boost,
                                               supertranslation=BMS_normal_order.supertranslation,
                                               ell_max=BMS_normal_order.ell_max,
                                               order=output_order)
        elif output_order == ['boost','supertranslation','rotation']:
            L_prime = Lorentz_transformation(rotation=BMS_normal_order.rotation,
                                             boost=BMS_normal_order.boost,
                                             ell_max=BMS_normal_order.ell_max,
                                             order=BMS_normal_order.order).reorder(output_order)

            S_prime = transform_supertranslation(BMS_normal_order.supertranslation,
                                                 Lorentz_transformation(boost=L_prime.boost,
                                                                        ell_max=L_prime.ell_max,
                                                                        order=L_prime.order).inverse())
            
            BMS_reordered = BMS_transformation(rotation=L_prime.rotation,
                                               boost=L_prime.boost,
                                               supertranslation=S_prime,
                                               ell_max=BMS_normal_order.ell_max,
                                               order=output_order)
        elif output_order == ['boost','rotation','supertranslation']:
            L_prime = Lorentz_transformation(rotation=BMS_normal_order.rotation,
                                             boost=BMS_normal_order.boost,
                                             ell_max=BMS_normal_order.ell_max,
                                             order=BMS_normal_order.order).reorder(output_order)

            S_prime = transform_supertranslation(BMS_normal_order.supertranslation,
                                                 L_prime.inverse())
            
            BMS_reordered = BMS_transformation(rotation=L_prime.rotation,
                                               boost=L_prime.boost,
                                               supertranslation=S_prime,
                                               ell_max=BMS_normal_order.ell_max,
                                               order=output_order)
            
        return BMS_reordered

    def is_identity(self, verbose=False):
        """Check if a BMS transformation is the identity element.
        """
        rotation_is_identity = np.allclose(self.rotation.components, [1,0,0,0])
        boost_is_identity = np.allclose(self.boost, [0,0,0])
        supertranslation_is_identity = np.allclose(self.supertranslation, np.zeros_like(self.supertranslation))
        if verbose:
            print(f"rotation is identity: {rotation_is_identity}")
            print(f"boost is identity: {boost_is_identity}")
            print(f"supertranslation is identity: {supertranslation_is_identity}")
        return rotation_is_identity and boost_is_identity and supertranslation_is_identity
    
    def inverse(self, output_order=None):
        """Compute the inverse of a BMS transformation.
        """
        if output_order is None:
            output_order = self.order[::-1]
            
        BMS_normal_order = self.reorder(output_order=['supertranslation','rotation','boost'])
        
        L_inverse = Lorentz_transformation(rotation=BMS_normal_order.rotation,
                                           boost=BMS_normal_order.boost,
                                           ell_max=BMS_normal_order.ell_max,
                                           output_order=BMS_normal_order.order).inverse(output_order=['rotation','boost'][::-1])

        n_theta = 2 * self.ell_max + 1; n_phi = n_theta;

        identity_grid_rotors = boosted_grid(np.quaternion(1, 0, 0, 0), [0, 0, 0], n_theta, n_phi)
        
        S_inverse = spinsfast.map2salm(
            sf.Grid(
                -sf.Modes(BMS_normal_order.supertranslation, spin_weight=0).real.evaluate(identity_grid_rotors),
                spin_weight=0
            ).real,
            0,
            self.ell_max)

        BMS_inverse = BMS_transformation(rotation=L_inverse.rotation,
                                         boost=L_inverse.boost,
                                         supertranslation=S_inverse,
                                         ell_max=BMS_normal_order.ell_max,
                                         order=['supertranslation','rotation','boost'][::-1])

        if BMS_inverse.order == output_order:
            return BMS_inverse
        else:
            return BMS_inverse.reorder(output_order=output_order)

    # Functions involving two BMS transformations
    
    def __eq__(self, other):
        rotation_eq = np.allclose(self.rotation.components,
                                  other.rotation.components)
        boost_eq = np.allclose(self.boost,
                               other.boost)
        supertranslation_eq = np.allclose(self.supertranslation,
                                          other.supertranslation)
        return rotation_eq and boost_eq and supertranslation_eq
        
    def compose(self, other, output_order=['supertranslation','rotation','boost']):
        """Compose two BMS transformations.
        
        Parameters
        ----------
        other: BMS_transformation
            2nd BMS transformation to be applied.
        output_order: list
            Order in which rotation and boost should be applied.
        """
        ell_max = max(self.ell_max, other.ell_max)
        
        BMS1_normal_order = self.reorder(output_order=['supertranslation','rotation','boost'])
        BMS2_normal_order = other.reorder(output_order=['supertranslation','rotation','boost'])
        
        L1 = Lorentz_transformation(rotation=BMS1_normal_order.rotation,
                                    boost=BMS1_normal_order.boost,
                                    ell_max=ell_max,
                                    order=BMS1_normal_order.order)
        L2 = Lorentz_transformation(rotation=BMS2_normal_order.rotation,
                                    boost=BMS2_normal_order.boost,
                                    ell_max=ell_max,
                                    order=BMS2_normal_order.order)
        L_composed = L1.compose(L2, output_order=['rotation','boost'])
        
        S1 = np.pad(BMS1_normal_order.supertranslation,
                    (0, (ell_max + 1)**2 - BMS1_normal_order.supertranslation.shape[0]))
        S2 = np.pad(BMS2_normal_order.supertranslation,
                    (0, (ell_max + 1)**2 - BMS2_normal_order.supertranslation.shape[0]))

        S_prime = transform_supertranslation(S2, L1)
        
        S_composed = S1 + S_prime

        BMS_composed = BMS_transformation(rotation=L_composed.rotation,
                                          boost=L_composed.boost,
                                          supertranslation=S_composed,
                                          ell_max=ell_max,
                                          order=['supertranslation','rotation','boost'])
        
        return BMS_composed.reorder(output_order=output_order)
