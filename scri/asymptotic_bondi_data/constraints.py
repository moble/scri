# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

### NOTE: The functions in this file are intended purely for inclusion in the AsymptoticBondData
### class.  In particular, they assume that the first argument, `self` is an instance of
### AsymptoticBondData.  They should probably not be used outside of that class.


def bondi_constraints(self, lhs=True, rhs=True):
    """Compute Bondi-gauge constraint equations

    Bondi gauge establishes some relations that the data must satisfy:

        \dot{\Psi0} = -\eth\Psi_1 + 3\sigma \Psi_2
        \dot{\Psi1} = -\eth\Psi_2 + 2\sigma \Psi_3
        \dot{\Psi2} = -\eth\Psi_3 + \sigma \Psi_4
        \Psi_3 = \eth \dot{\bar{\sigma}}
        \Psi_4 = - \ddot{\bar{\sigma}}
        Im[\Psi_2] = -Im[\eth^2\bar{\sigma} + \sigma \dot{\bar{\sigma}}]

    This function returns a 6-tuple of 2-tuples, corresponding to these 6 equations and their
    left- and right-hand sides.

    """
    return (
        self.bianchi_0(lhs, rhs),
        self.bianchi_1(lhs, rhs),
        self.bianchi_2(lhs, rhs),
        self.constraint_3(lhs, rhs),
        self.constraint_4(lhs, rhs),
        self.constraint_mass_aspect(lhs, rhs),
    )


@property
def bondi_violations(self):
    """Compute violations of Bondi-gauge constraints

    Bondi gauge establishes some relations that the data must satisfy:

        \dot{\Psi0} = -\eth\Psi_1 + 3\sigma \Psi_2
        \dot{\Psi1} = -\eth\Psi_2 + 2\sigma \Psi_3
        \dot{\Psi2} = -\eth\Psi_3 + \sigma \Psi_4
        \Psi_3 = \eth \dot{\bar{\sigma}}
        \Psi_4 = - \ddot{\bar{\sigma}}
        Im[\Psi_2] = -Im[\eth^2\bar{\sigma} + \sigma \dot{\bar{\sigma}}]

    This function returns a tuple of 6 arrays, corresponding to these 6 equations, in which the
    right-hand side is subtracted from the left-hand side.  No norms are taken.

    """
    constraints = self.bondi_constraints(True, True)
    return (lhs-rhs for (lhs, rhs) in constraints)


@property
def bondi_violation_norms(self):
    """Compute norms of violations of Bondi-gauge conditions

    Bondi gauge establishes some relations that the data must satisfy:

        \dot{\Psi0} = -\eth\Psi_1 + 3\sigma \Psi_2
        \dot{\Psi1} = -\eth\Psi_2 + 2\sigma \Psi_3
        \dot{\Psi2} = -\eth\Psi_3 + \sigma \Psi_4
        \Psi_3 = \eth \dot{\bar{\sigma}}
        \Psi_4 = - \ddot{\bar{\sigma}}
        Im[\Psi_2] = -Im[\eth^2\bar{\sigma} + \sigma \dot{\bar{\sigma}}]

    This function returns a tuple of 6 arrays, corresponding to the norms of these 6 equations,
    in which the right-hand side is subtracted from the left-hand side, and then the squared
    magnitude of that result is integrated over the sphere.  No integration is performed over
    time.

    """
    violations = self.bondi_violations
    return (v.norm() for v in violations)


def bianchi_0(self, lhs=True, rhs=True):
    """Return the left- and/or right-hand sides of the Psi0 component of the Bianchi identity

    In Bondi coordinates, the Bianchi identities simplify, resulting in this expression (among
    others) for the time derivative of Psi0:

        \dot{\Psi0} = -\eth\Psi_1 + 3\sigma \Psi_2

    Parameters
    ==========
    lhs: bool [defaults to True]
        If True, return the left-hand side of the equation above
    rhs: bool [defaults to True]
        If True, return the right-hand side of the equation above

    If both parameters are True, a tuple with elements (lhs_value, rhs_value) is returned;
    otherwise just the requested value is returned.

    """
    if lhs:
        lhs_value = self.psi0.dot
    if rhs:
        rhs_value = -self.psi1.eth + 3 * self.sigma * self.psi2
    if lhs and rhs:
        return (lhs_value, rhs_value)
    elif lhs:
        return lhs_value
    elif rhs:
        return rhs_value


def bianchi_1(self, lhs=True, rhs=True):
    """Return the left- and/or right-hand sides of the Psi1 component of the Bianchi identity

    In Bondi coordinates, the Bianchi identities simplify, resulting in this expression (among
    others) for the time derivative of Psi1:

        \dot{\Psi1} = -\eth\Psi_2 + 2\sigma \Psi_3

    Parameters
    ==========
    lhs: bool [defaults to True]
        If True, return the left-hand side of the equation above
    rhs: bool [defaults to True]
        If True, return the right-hand side of the equation above

    If both parameters are True, a tuple with elements (lhs_value, rhs_value) is returned;
    otherwise just the requested value is returned.

    """
    if lhs:
        lhs_value = self.psi1.dot
    if rhs:
        rhs_value = -self.psi2.eth + 2 * self.sigma * self.psi3
    if lhs and rhs:
        return (lhs_value, rhs_value)
    elif lhs:
        return lhs_value
    elif rhs:
        return rhs_value


def bianchi_2(self, lhs=True, rhs=True):
    """Return the left- and/or right-hand sides of the Psi2 component of the Bianchi identity

    In Bondi coordinates, the Bianchi identities simplify, resulting in this expression (among
    others) for the time derivative of Psi2:

        \dot{\Psi2} = -\eth\Psi_3 + \sigma \Psi_4

    Parameters
    ==========
    lhs: bool [defaults to True]
        If True, return the left-hand side of the equation above
    rhs: bool [defaults to True]
        If True, return the right-hand side of the equation above

    If both parameters are True, a tuple with elements (lhs_value, rhs_value) is returned;
    otherwise just the requested value is returned.

    """
    if lhs:
        lhs_value = self.psi2.dot
    if rhs:
        rhs_value = -self.psi3.eth + self.sigma * self.psi4
    if lhs and rhs:
        return (lhs_value, rhs_value)
    elif lhs:
        return lhs_value
    elif rhs:
        return rhs_value


def constraint_3(self, lhs=True, rhs=True):
    """Return the left- and/or right-hand sides of the Psi3 expression in Bondi gauge

    In Bondi coordinates, the value of Psi3 is given by a time derivative and an angular
    derivative of the (conjugate) shear:

        \Psi3 = \eth \dot{\bar{\sigma}}

    Parameters
    ==========
    lhs: bool [defaults to True]
        If True, return the left-hand side of the equation above
    rhs: bool [defaults to True]
        If True, return the right-hand side of the equation above

    If both parameters are True, a tuple with elements (lhs_value, rhs_value) is returned;
    otherwise just the requested value is returned.

    """
    if lhs:
        lhs_value = self.psi3
    if rhs:
        rhs_value = self.sigma.bar.dot.eth
    if lhs and rhs:
        return (lhs_value, rhs_value)
    elif lhs:
        return lhs_value
    elif rhs:
        return rhs_value


def constraint_4(self, lhs=True, rhs=True):
    """Return the left- and/or right-hand sides of the Psi4 expression in Bondi gauge

    In Bondi coordinates, the value of Psi4 is given by two time derivatives of the (conjugate)
    shear:

        \Psi4 = - \ddot{\bar{\sigma}}

    Parameters
    ==========
    lhs: bool [defaults to True]
        If True, return the left-hand side of the equation above
    rhs: bool [defaults to True]
        If True, return the right-hand side of the equation above

    If both parameters are True, a tuple with elements (lhs_value, rhs_value) is returned;
    otherwise just the requested value is returned.

    """
    if lhs:
        lhs_value = self.psi4
    if rhs:
        rhs_value = -self.sigma.bar.ddot
    if lhs and rhs:
        return (lhs_value, rhs_value)
    elif lhs:
        return lhs_value
    elif rhs:
        return rhs_value


def constraint_mass_aspect(self, lhs=True, rhs=True):
    """Return the left- and/or right-hand sides of the mass-aspect reality condition in Bondi gauge

    In Bondi coordinates, the Bondi mass aspect is always real, resulting in this relationship:

        \Im[\Psi2] = - \Im[\eth^2\bar{\sigma} + \sigma \dot{\bar{\sigma}}]

    Parameters
    ==========
    lhs: bool [defaults to True]
        If True, return the left-hand side of the equation above
    rhs: bool [defaults to True]
        If True, return the right-hand side of the equation above

    If both parameters are True, a tuple with elements (lhs_value, rhs_value) is returned;
    otherwise just the requested value is returned.

    """
    if lhs:
        lhs_value = np.imag(self.psi2.view(np.ndarray))
    if rhs:
        rhs_value = -np.imag(self.sigma.bar.eth.eth + self.sigma * self.sigma.bar.dot)
    if lhs and rhs:
        return (lhs_value, rhs_value)
    elif lhs:
        return lhs_value
    elif rhs:
        return rhs_value
