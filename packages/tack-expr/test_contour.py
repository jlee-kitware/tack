"""Contours of expressions evaluated where the contour reads them, against stored fields."""

import sys

import numpy as np

import tack
import tack.dataset as td
import tack_expr as fx

device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
tack.set_default_device(device)
results = []


def check(label, ok, detail=""):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {label:64} {detail}")


def positions(d):
    return np.asarray(d.positions(), dtype=np.float64)


def same_surface(a, b):
    pa, pb = np.round(positions(a), 5), np.round(positions(b), 5)
    pa, pb = pa[np.lexsort(pa.T)], pb[np.lexsort(pb.T)]
    return a.num_cells == b.num_cells and a.num_points == b.num_points and np.array_equal(pa, pb)


print(f"device={device}")
print("linear fields:")
data = td.wavelet(10)
expr = "RTData * 2 + 40 * coordsX"
lazy = fx.contour(expr, data, 330.0)
stored = td.contour(data, fx.evaluate(expr, data, at="points"), 330.0)
check("RTData * 2 + 40 coordsX: same triangles and shared points as stored",
      same_surface(lazy, stored), f"{lazy.num_cells} triangles, {lazy.num_points} points")
check("nothing stored in data.fields", set(data.fields) == {"RTData"})
f = fx.field(expr, data)
check("the expression field declares itself continuous", f.space.continuous)

print("higher order (order-2 fields, subdivided by their order):")


@tack.func
def quadratic(x):
    return x[0] * x[0] + x[1] * x[2]


h = td.uniform_grid((7, 6, 5), origin=(-0.6, -0.5, -0.4), spacing=(0.2, 0.2, 0.2))
h.fields["u"] = td.interpolate(h, quadratic, td.H1(h, order=2))
def level_error(s):
    x = positions(s)
    uu = x[:, 0] ** 2 + x[:, 1] * x[:, 2]
    return float(np.abs(uu * uu + uu - 0.3).max())


exact = fx.contour("u * u + u", h, 0.3)
print(f"  INFO  contour of u * u + u: {exact.num_cells} triangles, {exact.num_points} points")
check("exact crossings: vertices on the true level", level_error(exact) < 1e-5, f"max |f - 0.3| {level_error(exact):.1e}")
lattice = fx.contour("u * u + u", h, 0.3, exact=False)
err = level_error(lattice)
check("lattice crossings (exact=False): near it, steps of 1/2 cell", err < 2e-2, f"max |f - 0.3| {err:.1e}")
finer = fx.contour("u * u + u", h, 0.3, subdivision=8, exact=False)
xf = positions(finer)
uf = xf[:, 0] ** 2 + xf[:, 1] * xf[:, 2]
errf = float(np.abs(uf * uf + uf - 0.3).max())
check("a finer subdivision brings them closer", errf < err / 4, f"max |f - 0.3| {errf:.1e}")

print("expressions with derivatives (not continuous: points shared within each cell):")
g = fx.field("mag(grad(u))", h)
check("mag(grad(u)) field is not continuous", not g.space.continuous)
s = td.contour(h, g, 0.8, subdivision=2)
check("its contour has triangles", s.num_cells > 0, f"{s.num_cells} triangles")
xs = positions(s)
gm = np.linalg.norm(np.stack([2 * xs[:, 0], xs[:, 2], xs[:, 1]], 1), axis=1)
check("exact crossings: vertices on |grad u| = 0.8", float(np.abs(gm - 0.8).max()) < 1e-5,
      f"max error {np.abs(gm - 0.8).max():.1e}")
print(f"\n{sum(results)} of {len(results)} checks passed")

