"""Derivatives of expressions against float64 central differences of the exact fields.

u = x^2 + y z and v = 2x - y + 3z are scalars, U = (x y, y^2 + z, x - 2 y z) a vector,
all held exactly by order-2 H1 fields, so every expression's derivative is known.
"""

import sys

import numpy as np
import tack
import tack.dataset as td

import tack_expr as fx

device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
tack.set_default_device(device)
results = []


@tack.func
def f_u(x):
    return x[0] * x[0] + x[1] * x[2]


@tack.func
def f_v(x):
    return 2.0 * x[0] - x[1] + 3.0 * x[2]


@tack.func
def f_U(x):
    return tack.vec3(x[0] * x[1], x[1] * x[1] + x[2], x[0] - 2.0 * x[2] * x[1])


def u(x):
    return x[..., 0] ** 2 + x[..., 1] * x[..., 2]


def v(x):
    return 2 * x[..., 0] - x[..., 1] + 3 * x[..., 2]


def U(x):
    return np.stack([x[..., 0] * x[..., 1], x[..., 1] ** 2 + x[..., 2], x[..., 0] - 2 * x[..., 2] * x[..., 1]], -1)


def coords(x):
    return x


def jac(f, x, h=1e-5):
    """d f / d x_k by central differences, in float64: (..., 3) for a scalar f, (..., 3, 3) for a vector."""
    cols = []
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        cols.append((f(x + e) - f(x - e)) / (2 * h))
    return np.stack(cols, -1)


def div(F):
    return lambda x: np.trace(jac(F, x), axis1=-2, axis2=-1)


def curl(F):
    def c(x):
        J = jac(F, x)
        return np.stack([J[..., 2, 1] - J[..., 1, 2], J[..., 0, 2] - J[..., 2, 0], J[..., 1, 0] - J[..., 0, 1]], -1)
    return c


def qcrit(F):
    def q(x):
        J = jac(F, x)
        return (-0.5 * (J[..., 0, 0] ** 2 + J[..., 1, 1] ** 2 + J[..., 2, 2] ** 2)
                - (J[..., 0, 1] * J[..., 1, 0] + J[..., 0, 2] * J[..., 2, 0] + J[..., 1, 2] * J[..., 2, 1]))
    return q


def mag(a):
    return np.linalg.norm(a, axis=-1)


def check(label, got, ref, mask=None, tol=2e-5):
    got = np.asarray(got, np.float64).reshape(ref.shape)
    if mask is not None:
        got, ref = got[mask], ref[mask]
    err = float(np.max(np.abs(got - ref)) / max(1.0, float(np.max(np.abs(ref)))))
    ok = err < tol
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label:62} max error {err:.1e} of the largest value")


data = td.uniform_grid((7, 6, 5), origin=(0.2, 0.3, 0.4), spacing=(0.25, 0.2, 0.3))
data.fields["u"] = td.interpolate(data, f_u, td.H1(data, order=2))
data.fields["v"] = td.interpolate(data, f_v, td.H1(data, order=2))
data.fields["U"] = td.interpolate(data, f_U, td.H1(data, order=2), dtype=tack.vec3())
l2 = td.L2(data, order=2)
at_sites = {"cells": fx.evaluate("coords", data).values.numpy().astype(np.float64),
            "L2 order-2 nodes": fx.evaluate("coords", data, at=l2).values.numpy().astype(np.float64)}

cases = [
    ("grad(u * u)", lambda x: jac(lambda y: u(y) ** 2, x), None),
    ("grad(sqrt(u + 2) * v)", lambda x: jac(lambda y: np.sqrt(u(y) + 2) * v(y), x), None),
    ("grad(exp(-u) + sin(v))", lambda x: jac(lambda y: np.exp(-u(y)) + np.sin(v(y)), x), None),
    ("grad(u ^ 3 / (1 + v * v))", lambda x: jac(lambda y: u(y) ** 3 / (1 + v(y) ** 2), x), None),
    ("grad(atan2(u, v) + log10(u + 1))", lambda x: jac(lambda y: np.arctan2(u(y), v(y)) + np.log10(u(y) + 1), x), None),
    ("grad(mag(U))", lambda x: jac(lambda y: mag(U(y)), x), None),
    ("grad(dot(U, coords) - mag2(U))", lambda x: jac(lambda y: (U(y) * y).sum(-1) - (U(y) ** 2).sum(-1), x), None),
    ("grad(U_X * U_Z + coordsY)", lambda x: jac(lambda y: U(y)[..., 0] * U(y)[..., 2] + y[..., 1], x), None),
    ("grad(u > v ? u : v * v)", lambda x: jac(lambda y: np.where(u(y) > v(y), u(y), v(y) ** 2), x),
     lambda x: np.abs(u(x) - v(x)) > 1e-3),
    ("grad(min(u, v) + clamp(1, u, 3))", lambda x: jac(lambda y: np.minimum(u(y), v(y)) + np.clip(u(y), 1, 3), x),
     lambda x: (np.abs(u(x) - v(x)) > 1e-3) & (np.abs(u(x) - 1) > 1e-3) & (np.abs(u(x) - 3) > 1e-3)),
    ("div(U * u)", lambda x: div(lambda y: U(y) * u(y)[..., None])(x), None),
    ("div(cross(U, coords))", lambda x: div(lambda y: np.cross(U(y), y))(x), None),
    ("curl(U * v + coords * u)", lambda x: curl(lambda y: U(y) * v(y)[..., None] + y * u(y)[..., None])(x), None),
    ("curl(norm(U))", lambda x: curl(lambda y: U(y) / mag(U(y))[..., None])(x), None),
    ("qcrit(U + {u, v, u * v})", lambda x: qcrit(lambda y: U(y) + np.stack([u(y), v(y), u(y) * v(y)], -1))(x), None),
    ("qcrit(U / (1 + u))", lambda x: qcrit(lambda y: U(y) / (1 + u(y))[..., None])(x), None),
]

print(f"device={device}: {data.num_cells} voxels, order-2 fields")
for where, x in at_sites.items():
    print(f"at {where}:")
    for expr, exact, valid in cases:
        got = fx.evaluate(expr, data, at="cells" if where == "cells" else l2).values.numpy()
        check(expr, got, exact(x), None if valid is None else valid(x))

print("compared with td.gradients:")
check("grad(u) and grad(u * 1)", fx.evaluate("grad(u * 1)", data).values.numpy(),
      td.gradients(data, "u").values.numpy().astype(np.float64))
xc = at_sites["cells"]
cu = fx.evaluate("cells(u)", data).values.numpy().astype(np.float64)
check("grad(cells(u) * u) = cells(u) grad(u)", fx.evaluate("grad(cells(u) * u)", data).values.numpy(),
      cu[:, None] * jac(u, xc))

print("refused:")
for expr in ("grad(laplacian(u))", "grad(U * 2)", "div(u * v)", "grad(u + 1)"):
    try:
        fx.compile_expression(expr, data, at="cells")
        ok = expr == "grad(u + 1)"
        print(f"  {'PASS' if ok else 'FAIL'}  {expr!r} accepted")
    except fx.ExprError as e:
        ok = expr != "grad(u + 1)"
        print(f"  {'PASS' if ok else 'FAIL'}  {expr!r}: {str(e).splitlines()[0]}")
    results.append(ok)
try:
    fx.compile_expression("grad(u * v)", data, at="points")
    results.append(False)
except fx.ExprError as e:
    print(f"  PASS  'grad(u * v)' at points: {str(e).splitlines()[0]}")
    results.append(True)

print(f"\n{sum(results)} of {len(results)} checks passed")
print("\ngenerated kernel for qcrit(U / (1 + u)) at cells:")
print(fx.compile_expression("qcrit(U / (1 + u))", data).source)

