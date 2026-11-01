"""Field expressions for Tack: VTK calculator syntax over tack.dataset fields, and the same
expressions written with Python operators on td.Fields."""

from .lazy import contour, field
from .ops import (
    Expr,
    cells,
    clamp,
    compile_many,
    coords,
    cross,
    curl,
    div,
    dot,
    fsum,
    grad,
    laplacian,
    mag,
    mag2,
    maximum,
    minimum,
    neighbor,
    norm,
    owner,
    pi,
    points,
    qcrit,
    ref,
    vec,
    where,
)
from .parser import (
    Compiled,
    ExprError,
    Function,
    clear_cache,
    compile_expression,
    evaluate,
    for_each,
    function,
)

__all__ = ["Compiled", "Expr", "ExprError", "Function", "cells", "clamp", "clear_cache", "compile_expression",
           "compile_many", "contour", "coords", "cross", "curl", "div", "dot", "evaluate", "field", "for_each",
           "fsum", "function", "grad", "laplacian", "mag", "mag2", "maximum", "minimum", "neighbor", "norm",
           "owner", "pi", "points", "qcrit", "ref", "vec", "where"]
