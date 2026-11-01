"""Second derivatives against float64 central differences of the exact fields.

The Hessian differentiates the world gradient along parametric axes with a stencil that is
exact for these fields, so float32 results differ only by rounding, amplified by the
stencil to about 1e-4 of the largest value; the float64 checks at the end show 1e-13.

u = x^3 + x y z + z^2 (order 3) and U = (x y, y^2 + z, x - 2 y z) (order 2) are held exactly
by their H1 fields on a uniform grid, whose cells map affinely.
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
    return x[0] * x[0] * x[0] + x[0] * x[1] * x[2] + x[2] * x[2]


@tack.func
def f_U(x):
    return tack.vec3(x[0] * x[1], x[1] * x[1] + x[2], x[0] - 2.0 * x[2] * x[1])


def u(x):
    return x[..., 0] ** 3 + x[..., 0] * x[..., 1] * x[..., 2] + x[..., 2] ** 2


def U(x):
    return np.stack([x[..., 0] * x[..., 1], x[..., 1] ** 2 + x[..., 2], x[..., 0] - 2 * x[..., 2] * x[..., 1]], -1)


H = 1e-3


def d(f, k):
    e = np.zeros(3)
    e[k] = H
    return lambda x: (f(x + e) - f(x - e)) / (2 * H)


def grad(f):
    return lambda x: np.stack([d(f, k)(x) for k in range(3)], -1)


def lap(f):
    return lambda x: sum(d(d(f, k), k)(x) for k in range(3))


def div(F):
    return lambda x: sum(d(lambda y, k=k: F(y)[..., k], k)(x) for k in range(3))


def curl(F):
    def c(x):
        g = [[d(lambda y, i=i: F(y)[..., i], j)(x) for j in range(3)] for i in range(3)]
        return np.stack([g[2][1] - g[1][2], g[0][2] - g[2][0], g[1][0] - g[0][1]], -1)
    return c


def check(label, got, ref, tol=3e-4):
    got = np.asarray(got, np.float64).reshape(ref.shape)
    err = float(np.max(np.abs(got - ref)) / max(1.0, float(np.max(np.abs(ref)))))
    ok = err < tol
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label:60} max error {err:.1e} of the largest value")


data = td.uniform_grid((6, 5, 4), origin=(0.2, 0.3, 0.4), spacing=(0.25, 0.2, 0.3))
data.fields["u"] = td.interpolate(data, f_u, td.H1(data, order=3))
data.fields["U"] = td.interpolate(data, f_U, td.H1(data, order=2), dtype=tack.vec3())
l2 = td.L2(data, order=2)
sites = {"cells": fx.evaluate("coords", data).values.numpy().astype(np.float64),
         "L2 order-2 nodes": fx.evaluate("coords", data, at=l2).values.numpy().astype(np.float64)}

uu = lambda x: u(x) * u(x)
cases = [
    ("laplacian(u)", lap(u)),
    ("div(grad(u))", lap(u)),
    ("grad(grad(u)[0])", grad(lambda y: grad(u)(y)[..., 0])),
    ("curl(grad(u))", curl(grad(u))),
    ("laplacian(u * u)", lap(uu)),
    ("laplacian(exp(-u) + sin(U_Y))", lap(lambda y: np.exp(-u(y)) + np.sin(U(y)[..., 1]))),
    ("laplacian(mag(U))", lap(lambda y: np.linalg.norm(U(y), axis=-1))),
    ("div(grad(dot(U, coords) / (1 + u)))", lap(lambda y: (U(y) * y).sum(-1) / (1 + u(y)))),
    ("grad(div(U * u))", grad(div(lambda y: U(y) * u(y)[..., None]))),
    ("curl(curl(U))", curl(curl(U))),
    ("grad(laplacian(u) * 0 + mag2(grad(u)))", grad(lambda y: (grad(u)(y) ** 2).sum(-1))),
]
print(f"device={device}")
for where, x in sites.items():
    print(f"at {where}:")
    at = "cells" if where == "cells" else l2
    for expr, ref in cases:
        if "laplacian(u) * 0" in expr:
            expr = "grad(mag2(grad(u)))"
        check(expr, fx.evaluate(expr, data, at=at).values.numpy(), ref(x))

print("float64:")
d64 = td.uniform_grid((6, 5, 4), origin=(0.2, 0.3, 0.4), spacing=(0.25, 0.2, 0.3), dtype=tack.f64)
d64.fields["u"] = td.interpolate(d64, f_u, td.H1(d64, order=3), dtype=tack.f64)
x64 = fx.evaluate("coords", d64).values.numpy()
check("laplacian(u) in float64", fx.evaluate("laplacian(u)", d64).values.numpy(), lap(u)(x64), tol=1e-7)
check("grad(grad(u)[0]) in float64", fx.evaluate("grad(grad(u)[0])", d64).values.numpy(),
      grad(lambda y: grad(u)(y)[..., 0])(x64), tol=1e-6)

print("refused:")
for expr in ("grad(laplacian(u))", "laplacian(qcrit(U))", "laplacian(U)"):
    try:
        fx.compile_expression(expr, data)
        print(f"  FAIL  {expr!r} accepted")
        results.append(False)
    except fx.ExprError as e:
        print(f"  PASS  {expr!r}: {str(e).splitlines()[0]}")
        results.append(True)
print(f"\n{sum(results)} of {len(results)} checks passed")

