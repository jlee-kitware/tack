"""Ghost refresh between chained evaluations, projection into H1, and cache cleanup."""

import os
import sys
import time

import numpy as np

import tack
import tack.dataset as td
import tack_expr as fx

device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
tack.set_default_device(device)
results = []


def check(label, ok, detail=""):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {label:62} {detail}")


chain = (("cells(RTData)", "cells", "c"), ("points(c)", "points", "p"),
         ("mag(grad(p))", "cells", "g"), ("points(g)", "points", "q"))


def run(data, refresh):
    for expr, at, name in chain:
        fx.evaluate(expr, data, at=at, name=name, update_ghosts=refresh)


print(f"device={device}")
print("a chain of four evaluations on 4 pieces with a ghost layer, owned values vs the whole:")
whole = td.wavelet(10)
run(whole, False)
ref = td.statistics(whole, "q")
for refresh in (False, True):
    pieces = td.ghost_cells(td.split(td.wavelet(10), 4))
    run(pieces, refresh)
    got = td.statistics(pieces, "q")
    err = max(abs(got.mean - ref.mean) / abs(ref.mean), abs(got.max - ref.max) / abs(ref.max))
    if refresh:
        check("update_ghosts=True: points(g) owned (mean, max) as the whole's", err < 1e-5, f"error {err:.1e}")
    else:
        check("without the refresh the ghost rim reaches the owned values", err > 1e-4,
              f"error {err:.1e} (the reason for update_ghosts)")

print("projection into H1:")
data = td.uniform_grid((6, 5, 4), spacing=(0.2, 0.2, 0.2))
rng = np.random.default_rng(1)
data.fields["C"] = td.Field(td.Constant(data), tack.array(rng.random(data.num_cells, dtype=np.float32)))
h1 = td.H1(data, order=2)
try:
    fx.evaluate("C * 2", data, at=h1)
    check("cell data into H1 refused without project=True", False)
except fx.ExprError:
    check("cell data into H1 refused without project=True", True)
got = fx.evaluate("C * 2", data, at=h1, project=True).values.numpy()
data.fields["C2"] = td.Field(td.Constant(data), tack.array(data.fields["C"].values.numpy() * 2))
ref = td.project(data, "C2", h1).values.numpy()
err = float(np.abs(got - ref).max())
check("C * 2 with project=True vs td.project of 2 C into H1", err < 1e-6, f"max error {err:.1e}")

print("cache cleanup:")
old = os.path.join(fx.parser.CACHE_DIR, "fx_0000000000000000.py")
with open(old, "w") as f:
    f.write("# a stale module\n")
past = time.time() - 40 * 86400
os.utime(old, (past, past))
fx.evaluate("C * 3 + 0.125", data)
fresh = [n for n in os.listdir(fx.parser.CACHE_DIR) if n.endswith(".py")]
removed = fx.clear_cache(older_than_days=30)
check("modules older than 30 days removed, ones just used kept",
      removed == 1 and not os.path.exists(old) and len(os.listdir(fx.parser.CACHE_DIR)) >= len(fresh) - 1,
      f"removed {removed}")
print(f"  INFO  TACK_EXPR_CACHE_DAYS (default {fx.parser.CACHE_DAYS:g}) runs this cleanup at import")
print(f"\n{sum(results)} of {len(results)} checks passed")

