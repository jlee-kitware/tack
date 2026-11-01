"""points() of cell data inside the kernel on every mesh kind, against td.project and VTK."""

import sys
import time

import numpy as np
import tack
import tack.dataset as td
from tack.interop.vtk import vtk_to_dataset
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkIdList, vtkObject
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkAppendFilter, vtkCellDataToPointData

try:
    from vtkmodules.vtkFiltersCore import vtkExtractCells
except ImportError:
    from vtkmodules.vtkFiltersExtraction import vtkExtractCells
from vtkmodules.vtkFiltersGeneral import vtkDataSetTriangleFilter

import tack_expr as fx

device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
tack.set_default_device(device)
vtkObject.GlobalWarningDisplayOff()
rng = np.random.default_rng(5)
results = []
projections = {"n": 0}
_project = td.project


def counting_project(*args, **kwargs):
    projections["n"] += 1
    return _project(*args, **kwargs)


td.project = counting_project


def check(label, got, ref, tol=1e-5):
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64).reshape(np.shape(got))
    err = float(np.max(np.abs(got - ref)) / max(1.0, float(np.max(np.abs(ref)))))
    ok = err < tol
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label:62} max error {err:.1e}")


def with_cell_data(data):
    n = data.num_cells
    data.fields["C"] = td.Field(td.Constant(data), tack.array(rng.random(n, dtype=np.float32)))
    v = tack.empty((n,), tack.vec3())
    tack.copyto(v, rng.random((n, 3), dtype=np.float32))
    data.fields["W"] = td.Field(td.Constant(data), v)
    return data


def run(label, data, fused=True):
    with_cell_data(data)
    projections["n"] = 0
    got = fx.evaluate("points(C) * 2 + 1", data, at="points").values.numpy()
    gotv = fx.evaluate("points(W)", data, at="points").values.numpy()
    made = projections["n"]
    ref = _project(data, "C", td.H1(data)).values.numpy() * 2 + 1
    refv = _project(data, "W", td.H1(data)).values.numpy()
    check(f"{label}: points(C) * 2 + 1 vs td.project", got, ref)
    check(f"{label}: points(W), a vector, vs td.project", gotv, refv)
    want = 0 if fused else 2
    ok = made == want
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: td.project calls during evaluation: {made} (expected {want})")


print(f"device={device}")
run("uniform grid 13 x 9 x 7", td.uniform_grid((13, 9, 7), spacing=(0.1, 0.2, 0.3)))
run("2D grid 17 x 11", td.uniform_grid((17, 11, 1)))
run("rectilinear grid", td.rectilinear_grid(np.cumsum(rng.random(9)) , np.cumsum(rng.random(6)), np.cumsum(rng.random(5))))
run("grid 1 x 9 x 7 (its topology's axes renumbered)", td.uniform_grid((1, 9, 7)))

# A mixed mesh: half the voxels of a grid, the other half cut into tetrahedra, sharing points.
img = vtkImageData()
img.SetDimensions(11, 8, 6)
img.SetSpacing(0.1, 0.1, 0.1)
half = img.GetNumberOfCells() // 2
parts = []
for lo, hi in ((0, half), (half, img.GetNumberOfCells())):
    ids = vtkIdList()
    for i in range(lo, hi):
        ids.InsertNextId(i)
    ex = vtkExtractCells()
    ex.SetInputData(img)
    ex.SetCellList(ids)
    ex.Update()
    parts.append(ex.GetOutput())
tets = vtkDataSetTriangleFilter()
tets.SetInputData(parts[1])
tets.Update()
app = vtkAppendFilter()
app.MergePointsOn()
app.AddInputData(parts[0])
app.AddInputData(tets.GetOutput())
app.Update()
mixed_vtk = app.GetOutput()
mixed = vtk_to_dataset(mixed_vtk)
shapes = sorted({g.shape.__name__ for g in mixed.topology.groups() if g.count})
run(f"mixed mesh ({', '.join(shapes)})", mixed)
values = rng.random(mixed_vtk.GetNumberOfCells(), dtype=np.float32)
arr = numpy_to_vtk(values, deep=True)
arr.SetName("C")
mixed_vtk.GetCellData().AddArray(arr)
c2p = vtkCellDataToPointData()
c2p.SetInputData(mixed_vtk)
c2p.Update()
mixed2 = vtk_to_dataset(mixed_vtk)
check("mixed mesh: points(C) vs vtkCellDataToPointData",
      fx.evaluate("points(C)", mixed2, at="points").values.numpy(),
      vtk_to_numpy(c2p.GetOutput().GetPointData().GetArray("C")))

print("timing, points(C) * 2 on a 192 x 64 x 32 grid:")
big = with_cell_data(td.uniform_grid((192, 64, 32)))
fused_c = fx.compile_expression("points(C) * 2", big, at="points")
big.fields["P"] = _project(big, "C", td.H1(big))
times_two = fx.compile_expression("P * 2", big, at="points")


def two_pass():
    big.fields["P"] = _project(big, "C", td.H1(big))
    return times_two(big)


def timed(fn, reps=20):
    fn()
    tack.sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    tack.sync()
    return (time.perf_counter() - t0) / reps * 1e3


print(f"  INFO  in the kernel {timed(lambda: fused_c(big)):.2f} ms; td.project then the kernel {timed(two_pass):.2f} ms")
print(f"\n{sum(results)} of {len(results)} checks passed")

