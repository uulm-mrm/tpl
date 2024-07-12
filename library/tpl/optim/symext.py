"""
This file defines convenient symbolic extensions
(i.e. helper functions for stuff not found in sympy)
for definition and generation of optimizers via genopt.
"""

import copy
import sympy as sp

from sympy.core import cache


class ArraySymbol(sp.Symbol):
    """
    This represents an opaque array of doubles with undetermined shape.
    Its elements cannot be accessed in sympy methods.
    However, it is handled by the code generation accordingly.
    """

    def __new__(cls, name, **kwargs):
        self = super().__new__(cls, name, **kwargs)
        return self


class get_array_value(sp.Function):
    """
    This represents an array getter function by index.

    While this object has no special function in sympy,
    it gets used by the C++ code generation accordingly.

    Use like: value = array_get(i, j, k)
    """
    pass


class box_interp(sp.Function):
    """
    This represents a box "interpolation" function.

    While this object has no special function in sympy,
    it gets used by the C code generation accordingly.

    Arguments:
    dx, x, arr
    """

    nargs = (3,)


class lerp(sp.Function):
    """
    This represents a linear interpolation function.

    While this object has no special function in sympy,
    it gets used by the C code generation accordingly.

    Arguments:
    x0, dx, x, arr
    """

    nargs = (4,)


class lerp_angle(sp.Function):
    """
    This represents a linear interpolation function to be used for angles.

    While this object has no special function in sympy,
    it gets used by the C code generation accordingly.

    Arguments:
    x0, dx, x, arr
    """

    nargs = (4,)


class blerp(sp.Function):
    """
    This represents a bi-linear interpolation function.

    While this object has no special function in sympy,
    it gets used by the C++ code generation accordingly.

    Arguments:
    x0, y0, dx, dy, x, y, arr
    """

    nargs = (7,)


class lerp_wrap(sp.Function):
    """
    This represents a wrapping linear interpolation function.

    While this object has no special function in sympy,
    it gets used by the C++ code generation accordingly.
    """

    nargs = (5,)


def clear_cache():
    """
    The cache does weird things sometimes, so clearing becomes necessary.
    """

    cache.clear_cache()


def clone(expr):
    """
    So it turns out copying sympy symbols is officially broken.
    https://github.com/sympy/sympy/issues/7672

    As caching is the issue, we clear the cache before copy.
    """

    clear_cache()
    result = copy.deepcopy(expr)

    return result


def fixed(expr):
    """
    Prepends "fixed_" to the names of all free symbols in the
    expressions.

    This can be used for keeping certain parts of an expression
    "fixed" during differentiation.
    """

    for s in expr.free_symbols:
        if not s.name.startswith("fixed"):
            fixed_symbol = sp.Symbol("fixed_" + s.name)
            expr = expr.subs(s, fixed_symbol)

    return expr


def unfixed(expr):
    """
    Removes "fixed" from all free symbols in the expression.
    """

    for s in expr.free_symbols:

        if s.name.startswith("fixed"):

            # drop the "fixed_" prefix
            unfixed_name = "_".join(s.name.split("_")[1:])
            unfixed_symbol = sp.Symbol(unfixed_name)
            expr = expr.subs(s, unfixed_symbol)

    return expr
