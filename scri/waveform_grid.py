# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

from . import Inertial, WaveformModes, SpinWeights, DataNames
from . import h, hdot, sigma, news, psi0, psi1, psi2, psi3, psi4
from .waveform_base import WaveformBase, waveform_alterations

import sys
import warnings
import pprint
import numbers
import math
import numpy as np
from scipy import interpolate
import quaternion
import spherical_functions as sf
import spinsfast


def process_transformation_kwargs(ell_max, **kwargs):
    # Build the supertranslation and spacetime_translation arrays
    supertranslation = np.zeros((4,), dtype=complex)  # For now; may be resized below
    ell_max_supertranslation = 1  # For now; may be increased below
    if "supertranslation" in kwargs:
        supertranslation = np.array(kwargs.pop("supertranslation"), dtype=complex)
        if supertranslation.dtype != "complex" and supertranslation.size > 0:
            # I don't actually think this can ever happen...
            raise TypeError(
                "\nInput argument `supertranslation` should be a complex array with size>0.\n"
                "Got a {} array of shape {}.".format(supertranslation.dtype, supertranslation.shape)
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
                "\nInput supertranslation parameter must contain modes from ell=0 up to some ell_max, "
                "including\nall relevant m modes in standard order (see `spherical_functions` "
                "documentation for details).\nThus, it must be an array with length given by a "
                "perfect square; its length is {}".format(len(supertranslation))
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
            raise TypeError("\nInput argument `time_translation` should be a single float.\n" "Got {}.".format(t_trans))
        spacetime_translation[0] = t_trans
        supertranslation[0] = sf.constant_as_ell_0_mode(spacetime_translation[0])

    # Decide on the number of points to use in each direction.  A nontrivial supertranslation will introduce
    # power in higher modes, so for best accuracy, we need to account for that.  But we'll make it a firm
    # requirement to have enough points to capture the original waveform, at least
    w_ell_max = ell_max
    ell_max = w_ell_max + ell_max_supertranslation
    n_theta = kwargs.pop("n_theta", 2 * ell_max + 1)
    n_phi = kwargs.pop("n_phi", 2 * ell_max + 1)
    if n_theta < 2 * ell_max + 1 and abs(supertranslation[1:]).max() > 0.0:
        warning = (
            f"n_theta={n_theta} is small; because of the supertranslation, "
            + f"it will lose accuracy for anything less than 2*ell+1={ell_max}"
        )
        warnings.warn(warning)
    if n_theta < 2 * w_ell_max + 1:
        raise ValueError(f"n_theta={n_theta} is too small; " + "must be at least 2*ell+1={}".format(2 * w_ell_max + 1))
    if n_phi < 2 * ell_max + 1 and abs(supertranslation[1:]).max() > 0.0:
        warning = (
            f"n_phi={n_phi} is small; because of the supertranslation, "
            + f"it will lose accuracy for anything less than 2*ell+1={ell_max}"
        )
        warnings.warn(warning)
    if n_phi < 2 * w_ell_max + 1:
        raise ValueError(f"n_phi={n_phi} is too small; " + "must be at least 2*ell+1={}".format(2 * w_ell_max + 1))

    # Get the rotor for the frame rotation
    frame_rotation = np.quaternion(*np.array(kwargs.pop("frame_rotation", [1, 0, 0, 0]), dtype=float))
    if frame_rotation.abs() < 3e-16:
        raise ValueError(f"frame_rotation={frame_rotation} should be a unit quaternion")
    frame_rotation = frame_rotation.normalized()

    # Get the boost velocity vector
    boost_velocity = np.array(kwargs.pop("boost_velocity", [0.0] * 3), dtype=float)
    beta = np.linalg.norm(boost_velocity)
    if boost_velocity.shape != (3,) or beta >= 1.0:
        raise ValueError(
            "Input boost_velocity=`{}` should be a 3-vector with "
            "magnitude strictly less than 1.0.".format(boost_velocity)
        )
    gamma = 1 / math.sqrt(1 - beta ** 2)
    varphi = math.atanh(beta)

    # These are the angles in the transformed system at which we need to know the function values
    thetaprm_j_phiprm_k = np.array(
        [
            [[thetaprm_j, phiprm_k] for phiprm_k in np.linspace(0.0, 2 * np.pi, num=n_phi, endpoint=False)]
            for thetaprm_j in np.linspace(0.0, np.pi, num=n_theta, endpoint=True)
        ]
    )

    # Construct the function that modifies our rotor grid to account for the boost
    if beta > 3e-14:  # Tolerance for beta; any smaller and numerical errors will have greater effect
        vhat = boost_velocity / beta

        def Bprm_j_k(thetaprm, phiprm):
            """Construct rotor taking r' to r

            I derived this result in a different way, but I've also found it described in Penrose-Rindler Vol. 1,
            around Eq. (1.3.5).  Note, however, that their discussion is for the past celestial sphere,
            so there's a sign difference.

            """
            # Note: It doesn't matter which we use -- r' or r; all we need is the direction of the bivector
            # spanned by v and r', which is the same as the direction of the bivector spanned by v and r,
            # since either will be normalized, and one cross product is zero iff the other is zero.
            rprm = np.array(
                [math.cos(phiprm) * math.sin(thetaprm), math.sin(phiprm) * math.sin(thetaprm), math.cos(thetaprm)]
            )
            Thetaprm = math.acos(np.dot(vhat, rprm))
            Theta = 2 * math.atan(math.exp(-varphi) * math.tan(Thetaprm / 2.0))
            rprm_cross_vhat = np.quaternion(0.0, *np.cross(rprm, vhat))
            if rprm_cross_vhat.abs() > 1e-200:
                return (rprm_cross_vhat.normalized() * (Thetaprm - Theta) / 2).exp()
            else:
                return quaternion.one

    else:

        def Bprm_j_k(thetaprm, phiprm):
            return quaternion.one

    # Set up rotors that we can use to evaluate the SWSHs in the original frame
    R_j_k = np.empty(thetaprm_j_phiprm_k.shape[:2], dtype=np.quaternion)
    for j in range(thetaprm_j_phiprm_k.shape[0]):
        for k in range(thetaprm_j_phiprm_k.shape[1]):
            thetaprm_j, phiprm_k = thetaprm_j_phiprm_k[j, k]
            R_j_k[j, k] = (
                Bprm_j_k(thetaprm_j, phiprm_k) * frame_rotation * quaternion.from_spherical_coords(thetaprm_j, phiprm_k)
            )

    return (
        supertranslation,
        ell_max_supertranslation,
        ell_max,
        n_theta,
        n_phi,
        boost_velocity,
        beta,
        gamma,
        varphi,
        R_j_k,
        Bprm_j_k,
        thetaprm_j_phiprm_k,
        kwargs,
    )


class WaveformGrid(WaveformBase):
    def __init__(self, *args, **kwargs):
        """Initializer for WaveformGrid object"""
        # Do not directly access __n_theta or __n_phi; use n_theta or n_phi instead
        self.__n_theta = kwargs.pop("n_theta", 0)
        self.__n_phi = kwargs.pop("n_phi", 0)
        super().__init__(*args, **kwargs)

    @waveform_alterations
    def ensure_validity(self, alter=True, assertions=False):
        """Try to ensure that the `WaveformGrid` object is valid

        See `WaveformBase.ensure_validity` for the basic tests.  This function also includes tests that `data` is
        complex, and consistent with the n_theta and n_phi values.

        """
        import numbers

        errors = []
        alterations = []

        if assertions:
            from .waveform_base import test_with_assertions

            test = test_with_assertions
        else:
            from .waveform_base import test_without_assertions

            test = test_without_assertions

        test(
            errors,
            isinstance(self.__n_theta, numbers.Integral),
            "isinstance(self.__n_theta, numbers.Integral)  # type(self.__n_theta)={}".format(type(self.__n_theta)),
        )
        test(
            errors,
            isinstance(self.__n_phi, numbers.Integral),
            "isinstance(self.__n_phi, numbers.Integral)  # type(self.__n_phi)={}".format(type(self.__n_phi)),
        )
        test(errors, self.__n_theta >= 0, f"self.__n_theta>=0 # {self.__n_theta}")
        test(errors, self.__n_phi >= 0, f"self.__n_phi>=0 # {self.__n_phi}")

        test(
            errors,
            self.data.dtype == np.dtype(complex),
            f"self.data.dtype == np.dtype(complex)  # self.data.dtype={self.data.dtype}",
        )
        test(errors, self.data.ndim >= 2, f"self.data.ndim >= 2 # self.data.ndim={self.data.ndim}")
        test(
            errors,
            self.data.shape[1] == self.__n_theta * self.__n_phi,
            "self.data.shape[1] == self.__n_theta * self.__n_phi  "
            "# self.data.shape={}; self.__n_theta * self.__n_phi={}".format(
                self.data.shape[1], self.__n_theta * self.__n_phi
            ),
        )

        if alterations:
            self._append_history(alterations)
            print("The following alterations were made:\n\t" + "\n\t".join(alterations))
        if errors:
            print("The following conditions were found to be incorrectly False:\n\t" + "\n\t".join(errors))
            return False

        # Call the base class's version
        super().ensure_validity(alter, assertions)

        self.__history_depth__ -= 1
        self._append_history("WaveformModes.ensure_validity" + f"({self}, alter={alter}, assertions={assertions})")

        return True

    @property
    def n_theta(self):
        return self.__n_theta

    @property
    def n_phi(self):
        return self.__n_phi

    def to_modes(self, ell_max=None, ell_min=None):
        """Transform to modes of a spin-weighted spherical harmonic expansion

        Parameters
        ----------
        self : WaveformGrid object
            This is the object to be transformed to SWSH modes
        ell_max : int, optional
            The largest ell value to include in the output data.  Default value
            is deduced from n_theta and n_phi.
        ell_min : int, optional
            The smallest ell value to include in the output data.  Default value
            is abs(spin_weight).

        """
        s = SpinWeights[self.dataType]
        if ell_max is None:
            ell_max = int((max(self.n_theta, self.n_phi) - 1) // 2)
        if ell_min is None:
            ell_min = abs(s)
        if not isinstance(ell_max, numbers.Integral) or ell_max < 0:
            raise ValueError(f"Input `ell_max` should be a nonnegative integer; got `{ell_max}`.")
        if not isinstance(ell_min, numbers.Integral) or ell_min < 0 or ell_min > ell_max:
            raise ValueError(f"Input `ell_min` should be an integer between 0 and {ell_max}; got `{ell_min}`.")

        final_dim = int(np.prod(self.data.shape[2:]))
        old_data = self.data.reshape((self.n_times, self.n_theta, self.n_phi, final_dim))
        new_data = np.empty((self.n_times, sf.LM_total_size(ell_min, ell_max), final_dim), dtype=complex)
        # Note that spinsfast returns all modes, including ell<abs(s).  So we just chop those off
        for i_time in range(self.n_times):
            for i_final in range(final_dim):
                new_data[i_time, :, i_final] = spinsfast.map2salm(old_data[i_time, :, :, i_final], s, ell_max)[
                    sf.LM_index(ell_min, -ell_min, 0) :
                ]
        new_data = new_data.reshape((self.n_times, sf.LM_total_size(ell_min, ell_max)) + self.data.shape[2:])

        # old_data = self.data.reshape((self.n_times, self.n_theta, self.n_phi)+self.data.shape[2:])
        # new_data = np.empty((self.n_times, sf.LM_total_size(ell_min, ell_max))+self.data.shape[2:], dtype=complex)
        # # Note that spinsfast returns all modes, including ell<abs(s).  So we just chop those off
        # for i_time in range(self.n_times):
        #     new_data[i_time, :] = spinsfast.map2salm(old_data[i_time, :, :], s, ell_max)\
        #         [sf.LM_index(ell_min, -ell_min, 0):]

        m = WaveformModes(
            t=self.t,
            data=new_data,
            history=self.history,
            ell_min=ell_min,
            ell_max=ell_max,
            frameType=self.frameType,
            dataType=self.dataType,
            r_is_scaled_out=self.r_is_scaled_out,
            m_is_scaled_out=self.m_is_scaled_out,
            constructor_statement=f"{self}.to_modes({ell_max})",
        )
        return m

    @classmethod
    def from_modes(cls, w_modes, **kwargs):
        """Construct grid object from modes, with optional BMS transformation

        This "constructor" is designed with the goal of transforming the frame in which the modes are measured.  If
        this is not desired, it can be called without those parameters.

        It is important to note that the input transformation parameters are applied in the order listed in the
        parameter list below:
          1. (Super)Translations
          2. Rotation (about the origin of the translated system)
          3. Boost
        All input parameters refer to the transformation required to take the mode's inertial frame onto the inertial
        frame of the grid's inertial observers.  In what follows, the inertial frame of the modes will be unprimed,
        while the inertial frame of the grid will be primed.  NOTE: These are passive transformations, e.g. supplying
        the option space_translation=[0, 0, 5] to a Schwarzschild spacetime will move the coordinates to z'=z+5 and so
        the center of mass will be shifted in the negative z direction by 5 in the new coordinates.

        The translations (space, time, spacetime, or super) can be given in various ways, which may override each
        other.  Ultimately, however, they are essentially combined into a single function `alpha`, representing the
        supertranslation, which transforms the asymptotic time variable `u` as
          u'(theta, phi) = u - alpha(theta, phi)
        A simple time translation would correspond to
          alpha(theta, phi) = time_translation
        A pure spatial translation would correspond to
          alpha(theta, phi) = np.dot(space_translation, -nhat(theta, phi))
        where `np.dot` is the usual dot product, and `nhat` is the unit vector in the given direction.


        Parameters
        ----------
        w_modes : WaveformModes
            The object storing the modes of the original waveform, which will be converted to values on a grid in
            this function.  This is the only required argument to this function.
        n_theta : int, optional
        n_phi : int, optional
            Number of points in the equi-angular grid in the colatitude (theta) and azimuth (phi) directions. Each
            defaults to 2*ell_max+1, which is optimal for accuracy and speed.  However, this ell_max will typically
            be greater than the input waveform's ell_max by at least one, or the ell_max of the input
            supertranslation (whichever is greater).  This is to minimally account for the power at higher orders
            that such a supertranslation introduces.  You may wish to increase this further if the spatial size of
            your supertranslation is large compared to the smallest wavelength you wish to capture in your data
            [e.g., ell_max*Omega_orbital_max/speed_of_light], or if your boost speed is close to the speed of light.
        time_translation : float, optional
            Defaults to zero.  Nonzero overrides spacetime_translation and supertranslation.
        space_translation : float array of length 3, optional
            Defaults to empty (no translation).  Non-empty overrides spacetime_translation and supertranslation.
        spacetime_translation : float array of length 4, optional
            Defaults to empty (no translation).  Non-empty overrides supertranslation.
        supertranslation : complex array, optional
            This gives the complex components of the spherical-harmonic expansion of the supertranslation in standard
            form, starting from ell=0 up to some ell_max, which may be different from the ell_max of the input
            WaveformModes object.  Supertranslations must be real, so these values must obey the condition
              alpha^{ell,m} = (-1)^m \bar{alpha}^{ell,-m}
            Defaults to empty, which causes no supertranslation.
        frame_rotation : quaternion, optional
            Transformation applied to (x,y,z) basis of the mode's inertial frame.  For example, the basis z vector of
            the new grid frame may be written as
              z' = frame_rotation * z * frame_rotation.inverse()
            Defaults to 1, corresponding to the identity transformation (no frame_rotation).
        boost_velocity : float array of length 3, optional
            This is the three-velocity vector of the grid frame relative to the mode frame.  The norm of this vector
            is checked to ensure that it is smaller than 1.  Defaults to [], corresponding to no boost.
        psi4_modes : WaveformModes, required only if w_modes is type psi3, psi2, psi1, or psi0
        psi3_modes : WaveformModes, required only if w_modes is type psi2, psi1, or psi0
        psi2_modes : WaveformModes, required only if w_modes is type psi1 or psi0
        psi1_modes : WaveformModes, required only if w_modes is type psi0
            The objects storing the modes of the original waveforms of the same spacetime that
            w_modes belongs to. A BMS transformation of an asymptotic Weyl scalar requires mode
            information from all higher index Weyl scalars. E.g. if w_modes is of type scri.psi2,
            then psi4_modes and psi3_modes will be required. Note: the WaveformModes objects
            supplied to these arguments will themselves NOT be transformed. Please use the
            AsymptoticBondiData class to efficiently transform all asymptotic data at once.

        Returns
        -------
        WaveformGrid

        """
        # Check input object type and frame type
        #
        # The data in `w_modes` is given in the original frame.  We need to get the value of the field on a grid of
        # points corresponding to the points in the new grid frame.  But we must also remember that this is a
        # spin-weighted and boost-weighted field, which means that we need to account for the frame_rotation due to
        # `frame_rotation` and `boost_velocity`.  The way to do this is to use rotors to transform the points as needed,
        # and evaluate the SWSHs.  But for now, let's just reject any waveforms in a non-inertial frame
        if not isinstance(w_modes, WaveformModes):
            raise TypeError(
                "\nInput waveform object must be an instance of `WaveformModes`; "
                "this is of type `{}`".format(type(w_modes).__name__)
            )
        if w_modes.frameType != Inertial:
            raise ValueError(
                "\nInput waveform object must be in an inertial frame; "
                "this is in a frame of type `{}`".format(w_modes.frame_type_string)
            )

        # The first task is to establish a set of constant u' slices on which the new grid should be evaluated.  This
        # is done simply by translating the original set of slices by the time translation (the lowest moment of the
        # supertranslation).  But some of these slices (at the beginning and end) will not have complete data,
        # because of the direction-dependence of the rest of the supertranslation.  That is, in some directions,
        # the data for the complete slice (data in all directions on the sphere) of constant u' will actually refer to
        # spacetime events that were not in the original set of time slices; we would have to extrapolate the original
        # data.  So, for nontrivial supertranslations, the output data will actually represent a proper subset of the
        # input data.
        #
        # We can invert the equation for u' to obtain u as a function of angle assuming constant u'
        #   u'(theta, phi) = u + alpha(theta, phi) + u * np.dot(boost_velocity, nhat(theta, phi))
        #   u(theta, phi) = (u' - alpha(theta, phi)) / (1 + np.dot(boost_velocity, nhat(theta, phi)))
        # But really, we want u'(theta', phi') for given values
        #
        # Note that `space_translation` (and the spatial part of `spacetime_translation`) get reversed signs when
        # transformed into supertranslation modes, because these pieces enter the time transformation with opposite
        # sign compared to the time translation, as can be seen by looking at the retarded time: `t-r`.

        original_kwargs = kwargs.copy()

        (
            supertranslation,
            ell_max_supertranslation,
            ell_max,
            n_theta,
            n_phi,
            boost_velocity,
            beta,
            gamma,
            varphi,
            R_j_k,
            Bprm_j_k,
            thetaprm_j_phiprm_k,
            kwargs,
        ) = process_transformation_kwargs(w_modes.ell_max, **kwargs)

        # TODO: Incorporate the w_modes.frame information into rotors, which will require time dependence throughout
        # It would be best to leave the waveform in its frame.  But we'll have to apply the frame_rotation to the BMS
        # elements, which will be a little tricky.  Still probably not as tricky as applying to the waveform...

        # We need values of (1) waveform, (2) conformal factor, and (3) supertranslation, at each point of the
        # transformed grid, at each instant of time.
        SWSH_j_k = sf.SWSH_grid(R_j_k, w_modes.spin_weight, ell_max)
        SH_j_k = sf.SWSH_grid(R_j_k, 0, ell_max_supertranslation)  # standard (spin-zero) spherical harmonics
        r_j_k = np.array([(R * quaternion.z * R.inverse()).vec for R in R_j_k.flat]).T
        kconformal_j_k = 1.0 / (gamma * (1 - np.dot(boost_velocity, r_j_k).reshape(R_j_k.shape)))
        alphasupertranslation_j_k = np.tensordot(supertranslation, SH_j_k, axes=([0], [2])).real
        fprm_i_j_k = np.tensordot(
            w_modes.data,
            SWSH_j_k[
                :,
                :,
                sf.LM_index(w_modes.ell_min, -w_modes.ell_min, 0) : sf.LM_index(w_modes.ell_max, w_modes.ell_max, 0)
                + 1,
            ],
            axes=([1], [2]),
        )
        if beta != 0 or (supertranslation[1:] != 0).any():
            if w_modes.dataType == h:
                # Note that SWSH_j_k will use s=-2 in this case, so it can be used in the tensordot correctly
                supertranslation_deriv = 0.5 * sf.ethbar_GHP(sf.ethbar_GHP(supertranslation, 0, 0), -1, 0)
                supertranslation_deriv_values = np.tensordot(
                    supertranslation_deriv,
                    SWSH_j_k[:, :, : sf.LM_index(ell_max_supertranslation, ell_max_supertranslation, 0) + 1],
                    axes=([0], [2]),
                )
                fprm_i_j_k -= supertranslation_deriv_values[np.newaxis, :, :]
            elif w_modes.dataType == sigma:
                # Note that SWSH_j_k will use s=+2 in this case, so it can be used in the tensordot correctly
                supertranslation_deriv = 0.5 * sf.eth_GHP(sf.eth_GHP(supertranslation, 0, 0), 1, 0)
                supertranslation_deriv_values = np.tensordot(
                    supertranslation_deriv,
                    SWSH_j_k[:, :, : sf.LM_index(ell_max_supertranslation, ell_max_supertranslation, 0) + 1],
                    axes=([0], [2]),
                )
                fprm_i_j_k -= supertranslation_deriv_values[np.newaxis, :, :]
            elif w_modes.dataType in [psi0, psi1, psi2, psi3]:
                from scipy.special import comb

                eth_alphasupertranslation_j_k = np.tensordot(
                    1 / np.sqrt(2) * sf.eth_GHP(supertranslation, spin_weight=0),
                    sf.SWSH_grid(R_j_k, 1, ell_max_supertranslation),
                    axes=([0], [2]),
                )
                v_dot_rhat = np.insert(sf.vector_as_ell_1_modes(boost_velocity), 0, 0.0)
                eth_v_dot_rhat_j_k = np.tensordot(
                    1 / np.sqrt(2) * v_dot_rhat, sf.SWSH_grid(R_j_k, 1, 1), axes=([0], [2])
                )
                eth_uprm_over_kconformal_i_j_k = (
                    (w_modes.t[:, np.newaxis, np.newaxis] - alphasupertranslation_j_k[np.newaxis, :, :])
                    * gamma
                    * kconformal_j_k[np.newaxis, :, :]
                    * eth_v_dot_rhat_j_k[np.newaxis, :, :]
                    - eth_alphasupertranslation_j_k[np.newaxis, :, :]
                )
                # Loop over the Weyl scalars of higher index than w_modes, and sum
                # them with appropriate prefactors.
                for DT in range(w_modes.dataType + 1, psi4 + 1):
                    try:
                        w_modes_temp = kwargs.pop("psi{}_modes".format(DataNames[DT][-1]))
                    except KeyError:
                        raise ValueError(
                            "\nA BMS transformation of {} requires information from {}, which "
                            "has not been supplied.".format(w_modes.data_type_string, DataNames[DT])
                        )
                    SWSH_temp_j_k = sf.SWSH_grid(R_j_k, w_modes_temp.spin_weight, w_modes_temp.ell_max)
                    f_i_j_k = np.tensordot(
                        w_modes_temp.data,
                        SWSH_temp_j_k[
                            :,
                            :,
                            sf.LM_index(w_modes_temp.ell_min, -w_modes_temp.ell_min, 0) : sf.LM_index(
                                w_modes_temp.ell_max, w_modes_temp.ell_max, 0
                            )
                            + 1,
                        ],
                        axes=([1], [2]),
                    )
                    fprm_i_j_k += (
                        comb(5 - w_modes.dataType, 5 - DT)
                        * f_i_j_k
                        * eth_uprm_over_kconformal_i_j_k ** (DT - w_modes.dataType)
                    )
            elif w_modes.dataType not in [psi4, hdot, news]:
                warning = (
                    "\nNo BMS transformation is implemented for waveform objects "
                    "of dataType '{}'. Proceeding with the transformation as if it "
                    "were dataType 'Psi4'.".format(w_modes.data_type_string)
                )
                warnings.warn(warning)

        fprm_i_j_k *= (kconformal_j_k ** w_modes.conformal_weight)[np.newaxis, :, :]

        # Determine the new time slices.  The set u' is chosen so that on each slice of constant u'_i, the average value
        # of u is precisely u_i.  But then, we have to narrow that set down, so that every physical point on all the
        # u'_i' slices correspond to data in the range of input data.
        time_translation = sf.constant_from_ell_0_mode(supertranslation[0]).real
        uprm_i = (1 / gamma) * (w_modes.t - time_translation)
        uprm_min = (kconformal_j_k * (w_modes.t[0] - alphasupertranslation_j_k)).max()
        uprm_max = (kconformal_j_k * (w_modes.t[-1] - alphasupertranslation_j_k)).min()
        uprm_iprm = uprm_i[(uprm_i >= uprm_min) & (uprm_i <= uprm_max)]

        # Interpolate along each grid line to the new time in that direction.  Note that if there are additional
        # dimensions in the waveform data, InterpolatedUnivariateSpline will not be able to handle them automatically,
        # so we have to loop over them explicitly; an Ellipsis can't handle them.  Also, we are doing all time steps in
        # one go, for each j,k,... value, which means that we can overwrite the original data
        final_dim = int(np.prod(fprm_i_j_k.shape[3:]))
        fprm_i_j_k = fprm_i_j_k.reshape(fprm_i_j_k.shape[:3] + (final_dim,))
        for j in range(n_theta):
            for k in range(n_phi):
                uprm_i_j_k = kconformal_j_k[j, k] * (w_modes.t - alphasupertranslation_j_k[j, k])
                for final_indices in range(final_dim):
                    re_fprm_iprm_j_k = interpolate.InterpolatedUnivariateSpline(
                        uprm_i_j_k, fprm_i_j_k[:, j, k, final_indices].real
                    )
                    im_fprm_iprm_j_k = interpolate.InterpolatedUnivariateSpline(
                        uprm_i_j_k, fprm_i_j_k[:, j, k, final_indices].imag
                    )
                    fprm_i_j_k[: len(uprm_iprm), j, k, final_indices] = re_fprm_iprm_j_k(
                        uprm_iprm
                    ) + 1j * im_fprm_iprm_j_k(uprm_iprm)

        # Delete the extra rows from fprm_i_j_k, corresponding to values of u' outside of [u'min, u'max]
        fprm_iprm_j_k = np.delete(fprm_i_j_k, np.s_[len(uprm_iprm) :], 0)

        # Reshape, to have correct final dimensions
        fprm_iprm_j_k = fprm_iprm_j_k.reshape((fprm_iprm_j_k.shape[0], n_theta * n_phi) + w_modes.data.shape[2:])

        # Encapsulate into a new grid waveform
        g = cls(
            t=uprm_iprm,
            data=fprm_iprm_j_k,
            history=w_modes.history,
            n_theta=n_theta,
            n_phi=n_phi,
            frameType=w_modes.frameType,
            dataType=w_modes.dataType,
            r_is_scaled_out=w_modes.r_is_scaled_out,
            m_is_scaled_out=w_modes.m_is_scaled_out,
            constructor_statement=f"{cls.__name__}.from_modes({w_modes}, **{original_kwargs})",
        )

        if kwargs:
            warnings.warn("\nUnused kwargs passed to this function:\n{}".format(pprint.pformat(kwargs, width=1)))

        return g

    @classmethod
    def transform(cls, w_modes, **kwargs):
        """Transform modes by some BMS transformation

        This simply applies the `WaveformGrid.from_modes` function, followed by the `WaveformGrid.to_modes` function.
        See their respective docstrings for more details.  However, note that the `ell_max` parameter used in the
        second function call defaults here to the `ell_max` value in the input waveform.  This is slightly different
        from the usual default, because `WaveformGrid.from_modes` usually increases the effective ell value by 1.

        """
        if not isinstance(w_modes, WaveformModes):
            raise TypeError(
                "Expected WaveformModes object in argument 1; " "got `{}` instead.".format(type(w_modes).__name__)
            )
        ell_max = kwargs.pop("ell_max", w_modes.ell_max)
        return WaveformGrid.from_modes(w_modes, **kwargs).to_modes(ell_max)

    def __repr__(self):
        # "The goal of __str__ is to be readable; the goal of __repr__ is to be unambiguous." --- stackoverflow
        rep = super().__repr__()
        rep += f"\n# n_theta={self.n_theta}, n_phi={self.n_phi}"
        return rep


# Now, we can assign WaveformModes objects new capabilities based on WaveformGrid functions
WaveformModes.to_grid = lambda w_modes, **kwargs: WaveformGrid.from_modes(w_modes, **kwargs)
WaveformModes.from_grid = classmethod(lambda cls, w_grid, ell_max: WaveformGrid.to_modes(w_grid, ell_max))
#WaveformModes.transform = lambda w_mode, **kwargs: WaveformGrid.transform(w_mode, **kwargs)  # Move to WaveformModes class
if sys.version_info[0] == 2:
    WaveformModes.to_grid.__func__.__doc__ = WaveformGrid.from_modes.__doc__
    WaveformModes.from_grid.__func__.__doc__ = WaveformGrid.to_modes.__doc__
    # WaveformModes.transform.__func__.__doc__ = WaveformGrid.transform.__doc__
else:
    WaveformModes.to_grid.__doc__ = WaveformGrid.from_modes.__doc__
    WaveformModes.from_grid.__func__.__doc__ = WaveformGrid.to_modes.__doc__
    # WaveformModes.transform.__doc__ = WaveformGrid.transform.__doc__
