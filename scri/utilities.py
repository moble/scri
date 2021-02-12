# Copyright (c) 2019, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

import functools
import numpy as np
from . import jit

maxexp = np.finfo(float).maxexp * np.log(2) * 0.99


@jit
def _transition_function(x, x0, x1, y0, y1):
    transition = np.empty_like(x)
    ydiff = y1 - y0
    i = 0
    while x[i] <= x0:
        i += 1
    i0 = i
    transition[:i0] = y0
    while x[i] < x1:
        tau = (x[i] - x0) / (x1 - x0)
        exponent = 1.0 / tau - 1.0 / (1.0 - tau)
        if exponent >= maxexp:
            transition[i] = y0
        else:
            transition[i] = y0 + ydiff / (1.0 + np.exp(exponent))
        i += 1
    i1 = i
    transition[i1:] = y1
    return transition, i0, i1


def transition_function(x, x0, x1, y0=0.0, y1=1.0, return_indices=False):
    """Return a smooth function that is constant outside (x0, x1).

    This uses the standard smooth (C^infinity) function with derivatives of compact support to
    transition between the two values, being constant outside of the transition region (x0, x1).

    Parameters
    ==========
    x: array_like
        One-dimensional monotonic array of floats.
    x0: float
        Value before which the output will equal `y0`.
    x1: float
        Value after which the output will equal `y1`.
    y0: float [defaults to 0.0]
        Value of the output before `x0`.
    y1: float [defaults to 1.0]
        Value of the output after `x1`.
    return_indices: bool [defaults to False]
        If True, return the array and the indices (i0, i1) at which the transition occurs, such that
        t[:i0]==y0 and t[i1:]==y1.

    """
    if return_indices:
        return _transition_function(x, x0, x1, y0, y1)
    return _transition_function(x, x0, x1, y0, y1)[0]


@jit
def transition_function_derivative(x, x0, x1, y0=0.0, y1=1.0):
    """Return derivative of the transition function

    This function simply returns the derivative of `transition_function` with respect to the `x`
    parameter.  The parameters to this function are identical to those of that function.

    Parameters
    ==========
    x: array_like
        One-dimensional monotonic array of floats.
    x0: float
        Value before which the output will equal `y0`.
    x1: float
        Value after which the output will equal `y1`.
    y0: float [defaults to 0.0]
        Value of the output before `x0`.
    y1: float [defaults to 1.0]
        Value of the output after `x1`.

    """
    transition_prime = np.zeros_like(x)
    ydiff = y1 - y0
    i = 0
    while x[i] <= x0:
        i += 1
    while x[i] < x1:
        tau = (x[i] - x0) / (x1 - x0)
        exponent = 1.0 / tau - 1.0 / (1.0 - tau)
        if exponent >= maxexp:
            transition_prime[i] = 0.0
        else:
            exponential = np.exp(1.0 / tau - 1.0 / (1.0 - tau))
            transition_prime[i] = (
                -ydiff
                * exponential
                * (-1.0 / tau ** 2 - 1.0 / (1.0 - tau) ** 2)
                * (1 / (x1 - x0))
                / (1.0 + exponential) ** 2
            )
        i += 1
    return transition_prime


@jit
def bump_function(x, x0, x1, x2, x3, y0=0.0, y12=1.0, y3=0.0):
    """Return a smooth bump function that is constant outside (x0, x3) and inside (x1, x2).

    This uses the standard C^infinity function with derivatives of compact support to transition
    between the the given values.  By default, this is a standard bump function that is 0 outside of
    (x0, x3), and is 1 inside (x1, x2), but the constant values can all be adjusted optionally.

    Parameters
    ==========
    x: array_like
        One-dimensional monotonic array of floats.
    x0: float
        Value before which the output will equal `y0`.
    x1, x2: float
        Values between which the output will equal `y12`.
    x3: float
        Value after which the output will equal `y3`.
    y0: float [defaults to 0.0]
        Value of the output before `x0`.
    y12: float [defaults to 1.0]
        Value of the output after `x1` but before `x2`.
    y3: float [defaults to 0.0]
        Value of the output after `x3`.

    """
    bump = np.empty_like(x)
    ydiff01 = y12 - y0
    ydiff23 = y3 - y12
    i = 0
    while x[i] <= x0:
        i += 1
    bump[:i] = y0
    while x[i] < x1:
        tau = (x[i] - x0) / (x1 - x0)
        exponent = 1.0 / tau - 1.0 / (1.0 - tau)
        if exponent >= maxexp:
            bump[i] = y0
        else:
            bump[i] = y0 + ydiff01 / (1.0 + np.exp(exponent))
        i += 1
    i1 = i
    while x[i] <= x2:
        i += 1
    bump[i1:i] = y12
    while x[i] < x3:
        tau = (x[i] - x2) / (x3 - x2)
        exponent = 1.0 / tau - 1.0 / (1.0 - tau)
        if exponent >= maxexp:
            bump[i] = y12
        else:
            bump[i] = y12 + ydiff23 / (1.0 + np.exp(exponent))
        i += 1
    bump[i:] = y3
    return bump


def transition_to_constant(f, t, t1, t2):
    """Smoothly transition from the function to a constant.

    This works (implicitly) by multiplying the derivative of `f` with the transition function, and
    then integrating.  Using integration by parts, this simplifies to multiplying `f` itself by the
    transition function, and then subtracting the integral of `f` times the derivative of the
    transition function.  This integral is effectively restricted to the region (t1, t2).  Note that
    the final value (after t2) will depend on the precise values of `t1` and `t2`, and the behavior
    of `f` in between.

    Parameters
    ==========
    f: array_like
        One-dimensional array corresponding to the following `t` parameter.
    t: array_like
        One-dimensional monotonic array of floats.
    t1: float
        Value before which the output will equal `f`.
    t2: float
        Value after which the output will be constant.

    """
    from quaternion import indefinite_integral

    transition, i1, i2 = transition_function(t, t1, t2, y0=1.0, y1=0.0, return_indices=True)
    transition_dot = transition_function_derivative(t, t1, t2, y0=1.0, y1=0.0)
    f_transitioned = f * transition
    f_transitioned[i1:i2] -= indefinite_integral(f[i1:i2] * transition_dot[i1:i2], t[i1:i2])
    f_transitioned[i2:] = f_transitioned[i2 - 1]
    return f_transitioned


@jit
def xor_timeseries(c):
    """XOR a time-series of data in place

    Assumes time varies along the first dimension of the input array, but any number of other
    dimensions are supported.

    This function leaves the first time step unchanged, but successive timesteps are the XOR from
    the preceding time step — storing only the bits that have changed.  This transformation is
    useful when storing the data because it allows for greater compression in many cases.

    Note that the direction in which this operation is done matters.  This function starts with the
    last time, changes that data in place, and proceeds to earlier times.  To undo this
    transformation, we need to start at early times and proceed to later times.

    The function `xor_timeseries_reverse` achieves the opposite transformation, recovering the
    original data with bit-for-bit accuracy.

    """
    u = c.view(np.uint64)
    for i in range(u.shape[0] - 1, 0, -1):
        u[i] = np.bitwise_xor(u[i - 1], u[i])
    return c


@jit
def xor_timeseries_reverse(c):
    """XOR a time-series of data in place

    This function reverses the effects of `xor_timeseries`.  See that function's docstring for
    details.

    """
    u = c.view(np.uint64)
    for i in range(1, u.shape[0]):
        u[i] = np.bitwise_xor(u[i - 1], u[i])
    return c


@jit
def fletcher32(data):
    """Compute the Fletcher-32 checksum of an array

    This checksum is very easy to implement from scratch and very fast.

    Note that it's not entirely clear that everyone agrees on the naming of
    these functions.  This version uses 16-bit input, 32-bit accumulators,
    block sizes of 360, and a modulus of 65_535.

    Parameters
    ==========
    data: ndarray
        This array can have any dtype, but must be able to be viewed as uint16.

    Returns
    =======
    checksum: uint32

    """
    data = data.reshape((data.size,)).view(np.uint16)
    size = data.size
    c0 = np.uint32(0)
    c1 = np.uint32(0)
    j = 0
    block_size = 360  # largest number of sums that can be performed without overflow
    while j < size:
        block_length = min(block_size, size - j)
        for i in range(block_length):
            c0 += data[j]
            c1 += c0
            j += 1
        c0 %= np.uint32(65535)
        c1 %= np.uint32(65535)
    return c1 << np.uint32(16) | c0


@functools.lru_cache()
def multishuffle(shuffle_widths, forward=True):
    """Construct functions to "multi-shuffle" data

    The standard "shuffle" algorithm (as found in HDF5, for example) takes an
    array of numbers and shuffles their bytes so that all bytes of a given
    significance are stored together — the first byte of all the numbers are
    stored contiguously, then the second byte of all the numbers, and so on.
    The motivation for this is that — with reasonably smooth data — bytes in
    the same byte position in sequential numbers are usually more related to
    each other than they are to other bytes within the same number, which means
    that shuffling results in better compression of the data.

    There is no reason that shuffling can only work byte-by-byte, however.
    There is also a "bitshuffle" algorithm, which works in the same way, but
    collecting bits rather than bytes.  More generally, we could vary the
    number of bits stored together as we move along the numbers.  For example,
    we could store the first 8 bits of each number, followed by the next 4 bits
    of each number, etc.  This is the "multi-shuffle" algorithm.

    With certain types of data, this can reduce the compressed data size
    significantly.  For example, with float data for which successive values
    have been XOR-ed, the sign bit will very rarely change, the next 11 bits
    (representing the exponent) and a few of the following bits (representing
    the highest-significance digits) will typically be highly correlated, while
    as we move to lower significance there will be less correlation.  Thus, we
    might shuffle the first 8 bits together, followed by the next 8, then the
    next 4, the next 4, the next 2, and so on — decreasing the shuffle width as
    we go.  The `shuffle_widths` input might look like [8, 8, 4, 4, 2, 2, 1, 1,
    1, 1, ...].

    There are also some cases where we see correlation *increasing* again at
    low significance.  For example, if a number results from cancellation — the
    subtraction of two numbers much larger than their difference — then its
    lower-significance bits will be 0.  If we then multiply that by some
    integer (e.g., for normalization), there may be some very correlated but
    nonzero pattern.  In either case, compression might improve if the values
    at the end of our shuffle_widths list increase.

    Parameters
    ==========
    shuffle_widths: list of integers
        These integers represent the number of bits in each piece of each
        number that is shuffled, starting from the highest significance, and
        proceeding to the lowest.  The sum of these numbers must be the total
        bit width of the numbers that will be given as input — which must
        currently be 8, 16, 32, or 64.  There is no restriction on the
        individual widths, but note that if they do not fit evenly into 8-bit
        bytes, the result is unlikely to compress well.
    forward: bool [defaults to True]
        If True, the returned function will shuffle data; if False, the
        returned function will reverse this process — unshuffle.

    Returns
    =======
    shuffle_func: numba JIT function
        This function takes just one parameter — the array to be shuffled — and
        returns the shuffled array.  Note that the input array *must* be flat
        (have just one dimension), and will be viewed as an array of unsigned
        integers of the input bit width.  This can affect the shape of the
        array and order of elements.  You should ensure that this process will
        result in an array of numbers in the order that you want.  For example,
        if you have a 2-d array of floats `a` that are more continuous along
        the first dimension, you might pass `np.ravel(a.view(np.uint64), 'F')`,
        where F represents Fortran order, which varies more quickly in the
        first dimension.

    """
    import numpy as np
    import numba as nb

    bit_width = np.sum(shuffle_widths, dtype=np.int64)  # uint64 casts to float under floor division...
    if bit_width not in [8, 16, 32, 64]:
        raise ValueError(f"Total bit width must be one of [8, 16, 32, 64], not {bit_width}")
    dtype = np.dtype(f"u{bit_width//8}")
    bit_width = dtype.type(bit_width)
    reversed_shuffle_widths = np.array(list(reversed(shuffle_widths)), dtype=dtype)
    one = dtype.type(1)

    if forward:

        def shuffle(a):
            a = a.view(dtype)
            if a.ndim != 1:
                raise ValueError(
                    "\nThis function only accepts flat arrays.  Make sure you flatten "
                    "(using ravel, reshape, or flatten)\n in a way that keeps your data"
                    "contiguous in the order you want."
                )
            b = np.zeros_like(a)
            b_array_bit = np.uint64(0)
            for i, shuffle_width in enumerate(reversed_shuffle_widths):
                mask_shift = np.sum(reversed_shuffle_widths[:i])
                mask = dtype.type(2 ** shuffle_width - 1)
                pieces_per_element = bit_width // shuffle_width
                for a_array_index in range(a.size):
                    b_array_index = b_array_bit // bit_width
                    b_element_bit = dtype.type(b_array_bit % bit_width)
                    masked = (a[a_array_index] >> mask_shift) & mask
                    b[b_array_index] += masked << b_element_bit
                    if b_element_bit + shuffle_width > bit_width:
                        b[b_array_index + one] += masked >> (bit_width - b_element_bit)
                    b_array_bit += shuffle_width
            return b

        return nb.jit(shuffle)

    else:
        # This function is almost the same as above, except for:
        # 1) swap a <-> b in input and output
        # 2) reverse the effect of the line in which b was set from a
        def unshuffle(b):
            b = b.view(dtype)
            if b.ndim != 1:
                raise ValueError(
                    "\nThis function only accepts flat arrays.  Make sure you flatten "
                    "(using ravel, reshape, or flatten)\n in a way that keeps your data"
                    "contiguous in the order you want."
                )
            a = np.zeros_like(b)
            b_array_bit = dtype.type(0)
            for i, shuffle_width in enumerate(reversed_shuffle_widths):
                mask_shift = np.sum(reversed_shuffle_widths[:i])
                mask = dtype.type(2 ** shuffle_width - 1)
                pieces_per_element = bit_width // shuffle_width
                for a_array_index in range(a.size):
                    b_array_index = b_array_bit // bit_width
                    b_element_bit = b_array_bit % bit_width
                    masked = (b[b_array_index] >> b_element_bit) & mask
                    a[a_array_index] += masked << mask_shift
                    if b_element_bit + shuffle_width > bit_width:
                        a[a_array_index] += (
                            (b[b_array_index + one] << (bit_width - b_element_bit)) & mask
                        ) << mask_shift
                    b_array_bit += shuffle_width
            return a

        return nb.jit(unshuffle)
