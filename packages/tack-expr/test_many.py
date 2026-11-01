"""compile_many: several expressions in one kernel, against one kernel each, and its timing."""

import sys
import time

import numpy as np
import tack
import tack.dataset as td

import tack_expr as fx

device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
tack.set_default_device(device)
rng = np.random.default_rng(3)
results = []


def check(label, got, ref):
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    err = float(np.max(np.abs(got - ref)) / max(1.0, float(np.max(np.abs(ref)))))
    ok = err == 0.0 or err < 1e-6
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label:60} max error {err:.1e}")


def grid(dims):
    data = td.uniform_grid(dims, spacing=(0.1, 0.1, 0.1))
    n = data.num_points
    v = tack.empty((n,), tack.vec3())
    tack.copyto(v, (rng.random((n, 3), dtype=np.float32) - 0.5))
    data.fields["velocity"] = td.Field(td.H1(data), v)
    data.fields["pressure"] = td.Field(td.H1(data), tack.array(rng.random(n, dtype=np.float32)))
    data.fields["speed"] = td.magnitude(data, "velocity")
    return data


data = grid((20, 16, 12))
sets = {
    "points": {"cp": "2 * pressure / 0.0064", "ke": "0.5 * mag2(velocity)", "u2": "velocity * pressure"},
    "cells": {"vorticity": "mag(curl(velocity))", "q": "qcrit(velocity)", "helicity": "dot(velocity, curl(velocity))",
              "dp": "grad(pressure * speed)", "pc": "cells(pressure) + 1"},
    "L2 order 1": {"vorticity": "mag(curl(velocity))", "w": "velocity * 2 - coords"},
}
print(f"device={device}")
for where, exprs in sets.items():
    at = td.L2(data, order=1) if where.startswith("L2") else where
    many = fx.compile_many(exprs, data, at=at)
    fields = many(data)
    print(f"{where}: one kernel for {len(exprs)} expressions")
    for name, text in exprs.items():
        check(f"{name} = {text}", fields[name].values.numpy(), fx.evaluate(text, data, at=at).values.numpy())

src = fx.compile_many(sets["cells"], data).source
reads = src.count("_world_gradient(dom, ")
print(f"  INFO  cells kernel: {reads} world gradients for curl, qcrit, helicity and grad(pressure * speed) "
      f"(velocity's 3 components and pressure and speed: 5 when shared)")
results.append(reads == 5)

out = {"cp": tack.empty((data.num_points,))}
filled = fx.compile_many(sets["points"], data, at="points")(data, out=out)
results.append(filled["cp"].values is out["cp"])
print(f"  {'PASS' if results[-1] else 'FAIL'}  an output array passed in is filled in place")

print("timing, tack-lbm's output grid (192 x 64 x 32 points), four derived fields:")
big = grid((192, 64, 32))
four = {"vorticity": "mag(curl(velocity))", "q_criterion": "qcrit(velocity)",
        "helicity": "dot(velocity, curl(velocity))", "omega_z": "curl(velocity)[2]"}
one_each = [fx.compile_expression(t, big) for t in four.values()]
fused = fx.compile_many(four, big)


def timed(fn, reps=20):
    fn()
    tack.sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    tack.sync()
    return (time.perf_counter() - t0) / reps * 1e3


separate = timed(lambda: [c(big) for c in one_each])
together = timed(lambda: fused(big))
print(f"  INFO  one kernel each {separate:.2f} ms, one kernel for all four {together:.2f} ms")
print(f"\n{sum(results)} of {len(results)} checks passed")

