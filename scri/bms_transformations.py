import copy
import h5py
import scipy
import spinsfast
from scri.asymptotic_bondi_data.transformations import _process_transformation_kwargs, boosted_grid, conformal_factors
import numpy as np
import spherical_functions as sf


def fourvec_to_spin_matrix(fourvec):
    """Inner product of a four vector and the Pauli matrices,
    as defined by Eq. (1.2.39) of Spinors and Spacetime Vol. 1

    Parameters
    ----------
    fourvec: float array of length 4.
    """

    a = np.cos(fourvec[0] / 2) + 1j * fourvec[3] * np.sin(fourvec[0] / 2)
    b = (-fourvec[2] + 1j * fourvec[1]) * np.sin(fourvec[0] / 2)
    c = (+fourvec[2] + 1j * fourvec[1]) * np.sin(fourvec[0] / 2)
    d = np.cos(fourvec[0] / 2) - 1j * fourvec[3] * np.sin(fourvec[0] / 2)

    return np.array([[a, b], [c, d]])


def Lorentz_to_spin_matrix(lorentz):
    """Convert a Lorentz transformation to a spin matrix.

    Parameters
    ----------
    lorentz: LorentzTransformation
    """

    psi = 2 * np.arctan2(np.linalg.norm(lorentz.frame_rotation.components[1:]), lorentz.frame_rotation.components[0])
    if psi == 0:
        frame_rotation_vec_hat = np.array([0, 0, 0])
    else:
        frame_rotation_vec_hat = lorentz.frame_rotation.components[1:] / np.linalg.norm(
            lorentz.frame_rotation.components[1:]
        )

    chi = 1j * np.arctanh(np.linalg.norm(lorentz.boost_velocity))
    if chi == 0:
        boost_velocity_vec_hat = np.array([0, 0, 0])
    else:
        boost_velocity_vec_hat = lorentz.boost_velocity / np.linalg.norm(lorentz.boost_velocity)

    # compute product of Lorentz vector and Pauli matrices
    frame_rotation_spin_matrix = fourvec_to_spin_matrix([psi, *frame_rotation_vec_hat])
    boost_velocity_spin_matrix = fourvec_to_spin_matrix([chi, *boost_velocity_vec_hat])

    if lorentz.order.index("frame_rotation") < lorentz.order.index("boost_velocity"):
        return np.matmul(boost_velocity_spin_matrix, frame_rotation_spin_matrix)
    else:
        return np.matmul(frame_rotation_spin_matrix, boost_velocity_spin_matrix)


def pure_spin_matrix_to_Lorentz(A, is_rotation=None, tol=1e-14):
    """Convert a pure spin matrix to a rotation or a boost.

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
    nvec = np.array([(logA[1, 0] + logA[0, 1]) / 2, (logA[1, 0] - logA[0, 1]) / 2, (logA[0, 0] - logA[1, 1]) / 2])

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
        nvec_hat = np.array([0, 0, 1])
    else:
        nvec_hat = nvec / np.linalg.norm(nvec)

    if is_rotation:
        return np.quaternion(np.cos(psi / 2), *nvec_hat * np.sin(psi / 2))
    else:
        return np.tanh(psi) * nvec_hat


def spin_matrix_to_Lorentz(A, output_order=["frame_rotation", "boost_velocity"]):
    """Convert a spin matrix to a Lorentz transformation.

    This uses SVD to decompose the spin matrix into a
    spin matrix describing a Lorentz boost (positive or negative definite Hermitian) and a
    spin matrix describing a Lorentz rotation (unitary).

    Parameters
    ----------
    A: 2 by 2 complex array
        2 by 2 array corresponding to the input spin matrix.
    output_order: list
        Order in which rotation and boost should be applied.
    """
    if np.allclose(A, np.zeros_like(A)):
        return LorentzTransformation()

    u, s, vh = np.linalg.svd(A)

    try:
        frame_rotation_idx = output_order.index("frame_rotation")
    except:
        frame_rotation_idx = np.inf
    try:
        boost_velocity_idx = output_order.index("boost_velocity")
    except:
        boost_velocity_idx = np.inf

    if frame_rotation_idx < boost_velocity_idx:
        frame_rotation_matrix = np.matmul(u, vh)
        boost_velocity_matrix = np.linalg.multi_dot([u, np.diag(s), u.conj().T])
    else:
        frame_rotation_matrix = np.matmul(u, vh)
        boost_velocity_matrix = np.linalg.multi_dot([vh.conj().T, np.diag(s), vh])
    frame_rotation = pure_spin_matrix_to_Lorentz(frame_rotation_matrix, is_rotation=True)
    boost_velocity = pure_spin_matrix_to_Lorentz(boost_velocity_matrix, is_rotation=False)
    return LorentzTransformation(
        frame_rotation=frame_rotation.components, boost_velocity=boost_velocity, order=output_order
    )


def transform_supertranslation(S, lorentz, ell_max=None):
    """Apply a Lorentz transformation to a supertranslation and multiply by
    the conformal factor. This produces the supertranslation the appears
    when commuting a supertranslation through a Lorentz transformation.

    The Lorentz transformation is the transformation appearing on
    the RHS of the product, i.e., S' = L^{-1} S L.

    Parameters
    ----------
    S: ndarray, dtype=complex
        supertranslation to be transformed.
    lorentz: LorentzTransformation
        Lorentz transformation to be used to transform the supertranslation.
    ell_max: int
        Maximum ell to use when expressing functions via coordinates on the two-sphere.
    """
    if ell_max is None:
        ell_max = lorentz.ell_max
    n_theta = 2 * ell_max + 1
    n_phi = n_theta

    lorentz_inv = lorentz.inverse(output_order=["frame_rotation", "boost_velocity"])

    distorted_grid_rotors = boosted_grid(lorentz_inv.frame_rotation, lorentz_inv.boost_velocity, n_theta, n_phi)
    k, ðk_over_k, one_over_k, one_over_k_cubed = conformal_factors(lorentz_inv.boost_velocity, distorted_grid_rotors)

    return spinsfast.map2salm(
        (k[0] * sf.Grid(sf.Modes(S, spin_weight=0).evaluate(distorted_grid_rotors), spin_weight=0)).real, 0, ell_max
    )


class LorentzTransformation:
    def __init__(self, **kwargs):
        self.ell_max = copy.deepcopy(kwargs.pop("ell_max", 12))
        (
            frame_rotation,
            boost_velocity,
            supertranslation,
            working_ell_max,
            output_ell_max,
        ) = _process_transformation_kwargs(self.ell_max, **kwargs)
        self.frame_rotation = copy.deepcopy(frame_rotation)
        self.boost_velocity = copy.deepcopy(boost_velocity)

        self.order = copy.deepcopy(kwargs.pop("order", ["frame_rotation", "boost_velocity"]))
        if "supertranslation" in self.order:
            del self.order[self.order.index("supertranslation")]
        for transformation in ["frame_rotation", "boost_velocity"]:
            if transformation not in self.order:
                self.order.append(transformation)

    def __repr__(self):
        Lorentz_output = {}
        for transformation in self.order:
            if transformation == "frame_rotation":
                Lorentz_output[transformation] = self.frame_rotation
            elif transformation == "boost_velocity":
                Lorentz_output[transformation] = self.boost_velocity

        return f"LorentzTransformation(\n\t{self.order[0]}={Lorentz_output[self.order[0]]}\n\t{self.order[1]}={Lorentz_output[self.order[1]]}\n)"

    def copy(self):
        return LorentzTransformation(
            frame_rotation=self.frame_rotation.components,
            boost_velocity=self.boost_velocity,
            order=self.order,
            ell_max=self.ell_max,
        )

    def reorder(self, output_order):
        """Reorder a Lorentz transformation."""
        if not ("frame_rotation" in output_order and "boost_velocity" in output_order):
            raise ValueError("Not enough transformations")

        if self.order == output_order:
            return self.copy()
        else:
            L_spin_matrix = Lorentz_to_spin_matrix(self)
            L_reordered = spin_matrix_to_Lorentz(L_spin_matrix, output_order=output_order)
            return L_reordered

    def inverse(self, output_order=None):
        """Compute the inverse of a Lorentz transformation."""
        if output_order is None:
            output_order = self.order[::-1]

        L_spin_matrix = Lorentz_to_spin_matrix(self)
        L_spin_matrix_inverse = np.linalg.inv(L_spin_matrix)
        L_inverse = spin_matrix_to_Lorentz(L_spin_matrix_inverse, output_order=output_order)
        return L_inverse

    def is_close_to(self, other):
        frame_rotation_eq = np.allclose(self.frame_rotation.components, other.frame_rotation.components)
        boost_velocity_eq = np.allclose(self.boost_velocity, other.boost_velocity)
        return frame_rotation_eq and boost_velocity_eq

    def __mul__(self, other):
        """Compose two Lorentz transformations.

        These are composed as other * self, when each is viewed as a (passive)
        Lorentz transformation on the coordinates.

        The output order of the individual transformations is "frame_rotation", "boost_velocity".

        Parameters
        ----------
        other: LorentzTransformation
            2nd Lorentz transformation to be applied.
        """
        L1_spin_matrix = Lorentz_to_spin_matrix(self)
        L2_spin_matrix = Lorentz_to_spin_matrix(other)

        Ls_spin_matrix = np.matmul(L2_spin_matrix, L1_spin_matrix)

        return spin_matrix_to_Lorentz(Ls_spin_matrix, output_order=["frame_rotation", "boost_velocity"])


class BMSTransformation:
    def __init__(self, **kwargs):
        self.ell_max = copy.deepcopy(kwargs.pop("ell_max", 12))
        (
            frame_rotation,
            boost_velocity,
            supertranslation,
            working_ell_max,
            output_ell_max,
        ) = _process_transformation_kwargs(self.ell_max, **kwargs)
        self.frame_rotation = copy.deepcopy(frame_rotation)
        self.boost_velocity = copy.deepcopy(boost_velocity)
        self.supertranslation = np.pad(supertranslation, (0, (self.ell_max + 1) ** 2 - supertranslation.size))

        self.order = copy.deepcopy(kwargs.pop("order", ["supertranslation", "frame_rotation", "boost_velocity"]))
        for transformation in ["supertranslation", "frame_rotation", "boost_velocity"]:
            if transformation not in self.order:
                self.order.append(transformation)

    def __repr__(self):
        BMS_output = {}
        for transformation in self.order:
            if transformation == "frame_rotation":
                BMS_output[transformation] = self.frame_rotation
            elif transformation == "boost_velocity":
                BMS_output[transformation] = self.boost_velocity
            elif transformation == "supertranslation":
                BMS_output[transformation] = self.supertranslation[:9]

        return f"BMSTransformation(\n\t{self.order[0]}={BMS_output[self.order[0]]}\n\t{self.order[1]}={BMS_output[self.order[1]]}\n\t{self.order[2]}={BMS_output[self.order[2]]}\n)"

    def copy(self):
        return BMSTransformation(
            frame_rotation=self.frame_rotation.components,
            boost_velocity=self.boost_velocity,
            supertranslation=self.supertranslation,
            order=self.order,
            ell_max=self.ell_max,
        )

    def reorder(self, output_order):
        """Reorder a BMS transformation."""
        if not (
            "supertranslation" in output_order and "frame_rotation" in output_order and "boost_velocity" in output_order
        ):
            raise ValueError("Not enough transformations")

        # map to normal order
        normal_order = ["supertranslation", "frame_rotation", "boost_velocity"]
        if self.order == normal_order:
            BMS_normal_order = self.copy()
        elif self.order == ["frame_rotation", "supertranslation", "boost_velocity"]:
            S_prime = transform_supertranslation(
                self.supertranslation,
                LorentzTransformation(
                    frame_rotation=self.frame_rotation.components, ell_max=self.ell_max, order=self.order
                ),
            )

            BMS_normal_order = BMSTransformation(
                frame_rotation=self.frame_rotation.components,
                boost_velocity=self.boost_velocity,
                supertranslation=S_prime,
                ell_max=self.ell_max,
                order=normal_order,
            )
        elif self.order == ["frame_rotation", "boost_velocity", "supertranslation"]:
            S_prime = transform_supertranslation(
                self.supertranslation,
                LorentzTransformation(
                    frame_rotation=self.frame_rotation.components,
                    boost_velocity=self.boost_velocity,
                    ell_max=self.ell_max,
                    order=self.order,
                ),
            )

            BMS_normal_order = BMSTransformation(
                frame_rotation=self.frame_rotation.components,
                boost_velocity=self.boost_velocity,
                supertranslation=S_prime,
                ell_max=self.ell_max,
                order=normal_order,
            )
        elif self.order == ["supertranslation", "boost_velocity", "frame_rotation"]:
            L_prime = LorentzTransformation(
                frame_rotation=self.frame_rotation.components,
                boost_velocity=self.boost_velocity,
                ell_max=self.ell_max,
                order=self.order,
            ).reorder(normal_order)

            BMS_normal_order = BMSTransformation(
                frame_rotation=L_prime.frame_rotation.components,
                boost_velocity=L_prime.boost_velocity,
                supertranslation=self.supertranslation,
                ell_max=self.ell_max,
                order=normal_order,
            )
        elif self.order == ["boost_velocity", "supertranslation", "frame_rotation"]:
            L_prime = LorentzTransformation(
                frame_rotation=self.frame_rotation.components,
                boost_velocity=self.boost_velocity,
                ell_max=self.ell_max,
                order=self.order,
            ).reorder(normal_order)

            S_prime = transform_supertranslation(
                self.supertranslation,
                LorentzTransformation(boost_velocity=self.boost_velocity, ell_max=self.ell_max, order=self.order),
            )

            BMS_normal_order = BMSTransformation(
                frame_rotation=L_prime.frame_rotation.components,
                boost_velocity=L_prime.boost_velocity,
                supertranslation=S_prime,
                ell_max=self.ell_max,
                order=normal_order,
            )
        elif self.order == ["boost_velocity", "frame_rotation", "supertranslation"]:
            L_prime = LorentzTransformation(
                frame_rotation=self.frame_rotation.components,
                boost_velocity=self.boost_velocity,
                ell_max=self.ell_max,
                order=self.order,
            ).reorder(normal_order)

            S_prime = transform_supertranslation(self.supertranslation, L_prime)

            BMS_normal_order = BMSTransformation(
                frame_rotation=L_prime.frame_rotation.components,
                boost_velocity=L_prime.boost_velocity,
                supertranslation=S_prime,
                ell_max=self.ell_max,
                order=normal_order,
            )

        # map to output order
        if output_order == normal_order:
            BMS_reordered = BMS_normal_order.copy()
        elif output_order == ["frame_rotation", "supertranslation", "boost_velocity"]:
            S_prime = transform_supertranslation(
                BMS_normal_order.supertranslation,
                LorentzTransformation(
                    frame_rotation=BMS_normal_order.frame_rotation.components,
                    ell_max=BMS_normal_order.ell_max,
                    order=BMS_normal_order.order,
                ).inverse(),
            )

            BMS_reordered = BMSTransformation(
                frame_rotation=BMS_normal_order.frame_rotation.components,
                boost_velocity=BMS_normal_order.boost_velocity,
                supertranslation=S_prime,
                ell_max=BMS_normal_order.ell_max,
                order=output_order,
            )
        elif output_order == ["frame_rotation", "boost_velocity", "supertranslation"]:
            S_prime = transform_supertranslation(
                BMS_normal_order.supertranslation,
                LorentzTransformation(
                    frame_rotation=BMS_normal_order.frame_rotation.components,
                    boost_velocity=BMS_normal_order.boost_velocity,
                    ell_max=BMS_normal_order.ell_max,
                    order=BMS_normal_order.order,
                ).inverse(),
            )

            BMS_reordered = BMSTransformation(
                frame_rotation=BMS_normal_order.frame_rotation.components,
                boost_velocity=BMS_normal_order.boost_velocity,
                supertranslation=S_prime,
                ell_max=BMS_normal_order.ell_max,
                order=output_order,
            )
        elif output_order == ["supertranslation", "boost_velocity", "frame_rotation"]:
            L_prime = LorentzTransformation(
                frame_rotation=BMS_normal_order.frame_rotation.components,
                boost_velocity=BMS_normal_order.boost_velocity,
                ell_max=BMS_normal_order.ell_max,
                order=BMS_normal_order.order,
            ).reorder(output_order)

            BMS_reordered = BMSTransformation(
                frame_rotation=L_prime.frame_rotation.components,
                boost_velocity=L_prime.boost_velocity,
                supertranslation=BMS_normal_order.supertranslation,
                ell_max=BMS_normal_order.ell_max,
                order=output_order,
            )
        elif output_order == ["boost_velocity", "supertranslation", "frame_rotation"]:
            L_prime = LorentzTransformation(
                frame_rotation=BMS_normal_order.frame_rotation.components,
                boost_velocity=BMS_normal_order.boost_velocity,
                ell_max=BMS_normal_order.ell_max,
                order=BMS_normal_order.order,
            ).reorder(output_order)

            S_prime = transform_supertranslation(
                BMS_normal_order.supertranslation,
                LorentzTransformation(
                    boost_velocity=L_prime.boost_velocity, ell_max=L_prime.ell_max, order=L_prime.order
                ).inverse(),
            )

            BMS_reordered = BMSTransformation(
                frame_rotation=L_prime.frame_rotation.components,
                boost_velocity=L_prime.boost_velocity,
                supertranslation=S_prime,
                ell_max=BMS_normal_order.ell_max,
                order=output_order,
            )
        elif output_order == ["boost_velocity", "frame_rotation", "supertranslation"]:
            L_prime = LorentzTransformation(
                frame_rotation=BMS_normal_order.frame_rotation.components,
                boost_velocity=BMS_normal_order.boost_velocity,
                ell_max=BMS_normal_order.ell_max,
                order=BMS_normal_order.order,
            ).reorder(output_order)

            S_prime = transform_supertranslation(BMS_normal_order.supertranslation, L_prime.inverse())

            BMS_reordered = BMSTransformation(
                frame_rotation=L_prime.frame_rotation.components,
                boost_velocity=L_prime.boost_velocity,
                supertranslation=S_prime,
                ell_max=BMS_normal_order.ell_max,
                order=output_order,
            )

        return BMS_reordered

    def inverse(self, output_order=None):
        """Compute the inverse of a BMS transformation."""
        if output_order is None:
            output_order = self.order[::-1]

        bms_normal_order = self.reorder(output_order=["supertranslation", "frame_rotation", "boost_velocity"])

        L_inverse = LorentzTransformation(
            frame_rotation=bms_normal_order.frame_rotation.components,
            boost_velocity=bms_normal_order.boost_velocity,
            ell_max=bms_normal_order.ell_max,
            output_order=bms_normal_order.order,
        ).inverse(output_order=["frame_rotation", "boost_velocity"][::-1])

        n_theta = 2 * self.ell_max + 1
        n_phi = n_theta

        S_inverse = -bms_normal_order.supertranslation

        bms_inverse = BMSTransformation(
            frame_rotation=L_inverse.frame_rotation.components,
            boost_velocity=L_inverse.boost_velocity,
            supertranslation=S_inverse,
            ell_max=bms_normal_order.ell_max,
            order=["supertranslation", "frame_rotation", "boost_velocity"][::-1],
        )

        if bms_inverse.order == output_order:
            return bms_inverse
        else:
            return bms_inverse.reorder(output_order=output_order)

    def is_close_to(self, other):
        frame_rotation_eq = np.allclose(self.frame_rotation.components, other.frame_rotation.components)
        boost_velocity_eq = np.allclose(self.boost_velocity, other.boost_velocity)
        supertranslation_eq = np.allclose(self.supertranslation, other.supertranslation)
        return frame_rotation_eq and boost_velocity_eq and supertranslation_eq

    def __mul__(self, other):
        """Compose two BMS transformations.

        These are composed as other * self, when each is viewed as a (passive)
        BMS transformation on the coordinates.

        The output order of the individual transformations is "supertranslation", "frame_rotation", "boost_velocity".

        For more on this, see the documentation.

        Parameters
        ----------
        other: BMSTransformation
            2nd BMS transformation to be applied.
        """
        ell_max = max(self.ell_max, other.ell_max)

        bms1_normal_order = self.reorder(output_order=["supertranslation", "frame_rotation", "boost_velocity"])
        bms2_normal_order = other.reorder(output_order=["supertranslation", "frame_rotation", "boost_velocity"])

        L1 = LorentzTransformation(
            frame_rotation=bms1_normal_order.frame_rotation.components,
            boost_velocity=bms1_normal_order.boost_velocity,
            ell_max=ell_max,
            order=bms1_normal_order.order,
        )
        L2 = LorentzTransformation(
            frame_rotation=bms2_normal_order.frame_rotation.components,
            boost_velocity=bms2_normal_order.boost_velocity,
            ell_max=ell_max,
            order=bms2_normal_order.order,
        )
        L_composed = L1 * L2

        S1 = np.pad(
            bms1_normal_order.supertranslation, (0, (ell_max + 1) ** 2 - bms1_normal_order.supertranslation.shape[0])
        )
        S2 = np.pad(
            bms2_normal_order.supertranslation, (0, (ell_max + 1) ** 2 - bms2_normal_order.supertranslation.shape[0])
        )

        S_prime = transform_supertranslation(S1, L2)

        S_composed = S_prime + S2

        bms_composed = BMSTransformation(
            frame_rotation=L_composed.frame_rotation.components,
            boost_velocity=L_composed.boost_velocity,
            supertranslation=S_composed,
            ell_max=ell_max,
            order=["supertranslation", "frame_rotation", "boost_velocity"],
        )

        return bms_composed

    def to_file(self, filename, file_write_mode="w", group=None):
        dt = h5py.special_dtype(vlen=str)
        with h5py.File(filename, file_write_mode) as hf:
            if group is not None:
                g = hf.create_group(group)
            else:
                g = hf
            g.create_dataset("supertranslation", data=self.supertranslation)
            g.create_dataset("frame_rotation", data=self.frame_rotation.components)
            g.create_dataset("boost_velocity", data=self.boost_velocity)
            g.create_dataset("order", data=self.order)
            g.create_dataset("ell_max", data=self.ell_max)

        return

    def from_file(self, filename, group=None):
        with h5py.File(filename, "r") as hf:
            if group is not None:
                g = hf[group]
            else:
                g = hf
            supertranslation = np.array(g.get("supertranslation"))
            frame_rotation = np.array(g.get("frame_rotation"))
            boost_velocity = np.array(g.get("boost_velocity"))
            order = [x.decode("utf-8") for x in np.array(g.get("order"))]
            ell_max = int(np.array(g.get("ell_max")))

        BMS = BMSTransformation(
            frame_rotation=frame_rotation,
            boost_velocity=boost_velocity,
            supertranslation=supertranslation,
            order=order,
            ell_max=ell_max,
        )

        self.__dict__.update(BMS.__dict__)

        return
