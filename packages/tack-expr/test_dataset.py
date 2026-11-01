"""tack_expr over tack.dataset fields, checked against VTK and tack.dataset's filters."""

import sys

import numpy as np
import tack
import tack.dataset as td
from tack.interop.vtk import vtk_to_dataset
from vtkmodules.vtkCommonCore import VTK_DOUBLE, vtkObject
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import (vtkAppendFilter, vtkArrayCalculator,
                                       vtkCellDataToPointData, vtkPointDataToCellData)
from vtkmodules.vtkFiltersGeneral import vtkDataSetTriangleFilter
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import tack_expr as fx

device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
tack.set_default_device(device)
vtkObject.GlobalWarningDisplayOff()
rng = np.random.default_rng(1)
results = []


def check(label, got, ref, tol=1e-4):
    got = np.asarray(got, dtype=np.float64).reshape(np.shape(ref))
    ref = np.asarray(ref, dtype=np.float64)
    err = float(np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref))))
    ok = err < tol
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label:52} max rel err {err:.1e}")


def vtk_arrays(n, names=("T", "P", "V", "W")):
    return {"T": rng.random(n, dtype=np.float32), "P": rng.random(n, dtype=np.float32),
            "V": rng.random((n, 3), dtype=np.float32) * 2 - 1,
            "W": rng.random((n, 3), dtype=np.float32) * 2 - 1}


def image(dims, origin, spacing, point=None, cell=None):
    img = vtkImageData()
    img.SetDimensions(dims)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)
    for attrs, arrays in ((img.GetPointData(), point), (img.GetCellData(), cell)):
        for name, a in (arrays or {}).items():
            arr = numpy_to_vtk(a, deep=True)
            arr.SetName(name)
            attrs.AddArray(arr)
    return img


def unstructured(vtk_data):
    f = vtkAppendFilter()
    f.SetInputData(vtk_data)
    f.Update()
    return f.GetOutput()


def calc(vtk_data, expr, cells=False):
    c = vtkArrayCalculator()
    c.SetInputData(vtk_data)
    c.SetAttributeTypeToCellData() if cells else c.SetAttributeTypeToPointData()
    c.AddScalarArrayName("T")
    c.AddScalarArrayName("P")
    c.AddVectorArrayName("V")
    c.AddVectorArrayName("W")
    for k, comp in enumerate("XYZ"):
        c.AddScalarVariable(f"V_{comp}", "V", k)
        c.AddScalarVariable(f"W_{comp}", "W", k)
        if not cells:
            c.AddCoordinateScalarVariable(f"coords{comp}", k)
    if not cells:
        c.AddCoordinateVectorVariable("coords")
    c.SetFunction(expr)
    c.SetResultArrayName("result")
    c.SetResultArrayType(VTK_DOUBLE)
    c.Update()
    out = c.GetOutput()
    arr = (out.GetCellData() if cells else out.GetPointData()).GetArray("result")
    return None if arr is None else vtk_to_numpy(arr)


dims, origin, spacing = (20, 15, 10), (0.5, -1.0, 2.0), (0.1, 0.2, 0.05)
npts = dims[0] * dims[1] * dims[2]
ncells = (dims[0] - 1) * (dims[1] - 1) * (dims[2] - 1)

print(f"device={device}")
# 1. the calculator syntax on point data of an unstructured mesh of voxels
pimg = image(dims, origin, spacing, point=vtk_arrays(npts))
pgrid = unstructured(pimg)
pdata = vtk_to_dataset(pgrid)
print(f"point data: {pdata.num_points} points, {pdata.num_cells} cells; fields "
      + ", ".join(f"{k}:{type(v.space).__name__}" for k, v in pdata.fields.items()))
cases = [
    "T + P", "T - 2*P", "T * P / (P + 1)", "T ^ 2", "2 ^ 3 ^ 2", "-T ^ 2", "T + P ^ 2",
    "T % 0.3", "-T % 0.3", "(T + 1) * -P", "abs(T - 0.5)", "acos(T)", "asin(T)", "atan(T)",
    "ceil(10*T)", "floor(10*T)", "cos(T)", "cosh(T)", "exp(T)", "ln(P + 1)", "log10(P + 1)",
    "sin(T)", "sinh(T)", "sqrt(T)", "tan(T)", "tanh(T)", "sign(T - 0.5)", "min(T, P)",
    "max(T, P)", "pow(T, 3)", "atan2(T, P)", "clamp(0.2, T, 0.8)", "pi * T", "V_X + V_Y * V_Z",
    "coordsX + coordsY - coordsZ", "T > 0.5 ? T : P", "if(T > 0.5, T, P)",
    "(T > 0.3) and (P < 0.7)", "(T < 0.5) | (P > 0.5)", "not(T > 0.5)", "T = T", "T <> P",
    "mag(V)", "dot(V, W)", "V + W", "2 * V", "V * T", "V / 2", "cross(V, W)", "norm(V)",
    "iHat * T + jHat * P + kHat", "coords", "{T, P, 1}", "V * W", "V[1]", "T > 0.5 ? V : W",
    "mag(cross(V, W)) / (mag(V) * mag(W))",
]
passed = 0
for expr in cases:
    ref = calc(pgrid, expr)
    got = fx.evaluate(expr, pdata, at="points").values.numpy()
    err = float(np.max(np.abs(got.reshape(ref.shape) - ref) / np.maximum(1.0, np.abs(ref))))
    passed += err < 1e-4
    if err >= 1e-4:
        print(f"  FAIL  {expr}: {err:.1e}")
results.append(passed == len(cases))
print(f"  {'PASS' if passed == len(cases) else 'FAIL'}  calculator syntax at points: "
      f"{passed} of {len(cases)} match vtkArrayCalculator")

# 2. cell data
cimg = image(dims, origin, spacing, cell=vtk_arrays(ncells))
cgrid = unstructured(cimg)
cdata = vtk_to_dataset(cgrid)
print("cell data: fields " + ", ".join(f"{k}:{type(v.space).__name__}" for k, v in cdata.fields.items()))
for expr in ("T + P * 2", "mag(V) * T", "T > 0.5 ? V : W", "cross(V, W) / (1 + T)"):
    check(f"'{expr}' at cells vs vtkArrayCalculator", fx.evaluate(expr, cdata, at="cells").values.numpy(),
          calc(cgrid, expr, cells=True))

# 3. conversions, against VTK's filters
p2c = vtkPointDataToCellData()
p2c.SetInputData(pgrid)
p2c.Update()
check("cells(T) vs vtkPointDataToCellData", fx.evaluate("cells(T)", pdata, at="cells").values.numpy(),
      vtk_to_numpy(p2c.GetOutput().GetCellData().GetArray("T")))
check("T at cells (basis at center) vs vtkPointDataToCellData",
      fx.evaluate("T", pdata, at="cells").values.numpy(),
      vtk_to_numpy(p2c.GetOutput().GetCellData().GetArray("T")))
c2p = vtkCellDataToPointData()
c2p.SetInputData(cgrid)
c2p.Update()
check("points(T) vs vtkCellDataToPointData", fx.evaluate("points(T)", cdata, at="points").values.numpy(),
      vtk_to_numpy(c2p.GetOutput().GetPointData().GetArray("T")))
check("points(V) * 2 vs 2 * vtkCellDataToPointData",
      fx.evaluate("points(V) * 2", cdata, at="points").values.numpy(),
      2 * vtk_to_numpy(c2p.GetOutput().GetPointData().GetArray("V")))

# 4. a tetrahedral mesh
tf = vtkDataSetTriangleFilter()
tf.SetInputData(pimg)
tf.Update()
tdata = vtk_to_dataset(tf.GetOutput())
print(f"tetrahedra: {tdata.num_cells} cells")
check("T * 2 + P at points on tets vs vtkArrayCalculator",
      fx.evaluate("T * 2 + P", tdata, at="points").values.numpy(), calc(tf.GetOutput(), "T * 2 + P"))
p2c.SetInputData(tf.GetOutput())
p2c.Update()
check("cells(T) on tets vs vtkPointDataToCellData", fx.evaluate("cells(T)", tdata, at="cells").values.numpy(),
      vtk_to_numpy(p2c.GetOutput().GetCellData().GetArray("T")))
check("grad(T) at cells on tets vs td.gradients", fx.evaluate("grad(T)", tdata, at="cells").values.numpy(),
      td.gradients(tdata, "T", at="cells").values.numpy())


# 5. higher order: an order-2 H1 field of a quadratic, exact in its space
@tack.func
def quadratic(x):
    return x[0] * x[0] + x[1] * x[2]


hdata = td.uniform_grid((6, 5, 4), origin=(0.2, -0.5, 1.0), spacing=(0.3, 0.25, 0.4))
hdata.fields["u"] = td.interpolate(hdata, quadratic, td.H1(hdata, order=2))
print(f"order 2: H1 field u = x^2 + y z on {hdata.num_cells} voxels, {hdata.fields['u'].space.size} values")
l2 = td.L2(hdata, order=2)
x = fx.evaluate("coords", hdata, at=l2).values.numpy()
uq = x[:, 0] ** 2 + x[:, 1] * x[:, 2]
check("u at L2 order-2 nodes vs td.project", fx.evaluate("u", hdata, at=l2).values.numpy(),
      td.project(hdata, "u", l2).values.numpy())
check("u * u + 2 at L2 order-2 nodes vs exact", fx.evaluate("u * u + 2", hdata, at=l2).values.numpy(),
      uq * uq + 2)
xc = fx.evaluate("coords", hdata, at="cells").values.numpy()
check("grad(u) at cells vs td.gradients", fx.evaluate("grad(u)", hdata, at="cells").values.numpy(),
      td.gradients(hdata, "u", at="cells").values.numpy())
check("grad(u) at cells vs exact (2x, z, y)", fx.evaluate("grad(u)", hdata, at="cells").values.numpy(),
      np.stack([2 * xc[:, 0], xc[:, 2], xc[:, 1]], axis=1))
check("mag(grad(u)) at L2 order-2 nodes vs exact",
      fx.evaluate("mag(grad(u))", hdata, at=l2).values.numpy(),
      np.sqrt((2 * x[:, 0]) ** 2 + x[:, 2] ** 2 + x[:, 1] ** 2))


# 6. faces: an integrated flux of v = (x, y, z), its outward sum, and face jumps
@tack.kernel
def sides(faces, s0, s1):
    for f in faces:
        e = faces.entity_id(f)
        s0[e] = faces.side_cell(f, 0)
        s1[e] = faces.side_cell(f, 1)


fdata = td.uniform_grid((6, 5, 4), spacing=(0.5, 0.4, 0.3))
nf = fdata.topology.faces().num_faces
geometry = td.face_geometry(fdata)
centers = td.face_centers(fdata).values.numpy()
flux = (np.sum(centers * geometry.normal.values.numpy(), axis=1) * geometry.area.values.numpy()).astype(np.float32)
fdata.fields["flux"] = td.Field(td.Values(fdata, "faces", oriented=True), tack.array(flux))
check("fsum(flux) at cells vs td.net_flux", fx.evaluate("fsum(flux)", fdata, at="cells").values.numpy(),
      td.net_flux(fdata, "flux").values.numpy())
volume = td.cell_geometry(fdata).measure.values.numpy()
check("fsum(flux) / vol vs divergence of (x, y, z) = 3",
      fx.evaluate("fsum(flux)", fdata, at="cells").values.numpy() / volume, np.full(fdata.num_cells, 3.0))
tc = rng.random(fdata.num_cells, dtype=np.float32)
fdata.fields["TC"] = td.Field(td.Constant(fdata), tack.array(tc))
s0, s1 = tack.zeros((nf,), tack.i32), tack.zeros((nf,), tack.i32)
td.for_each(sides, fdata, "faces", s0, s1)
s0, s1 = s0.numpy(), s1.numpy()
jump = np.where(s1 >= 0, tc[np.maximum(s1, 0)] - tc[s0], 0.0)
check("neighbor(TC) - owner(TC) at faces vs side cells",
      fx.evaluate("neighbor(TC) - owner(TC)", fdata, at="faces").values.numpy(), jump)
check("flux * owner(TC) at faces", fx.evaluate("flux * owner(TC)", fdata, at="faces").values.numpy(),
      flux * tc[s0])

# 7. what each location refuses, with the fix in the message
print("refused, with the fix:")
for expr, data, at in (("TC + 1", fdata, "points"), ("flux * 2", fdata, "cells"),
                       ("TC * 2", fdata, "faces"), ("grad(u)", hdata, "points"),
                       ("T + 1", cdata, "points")):
    try:
        fx.compile_expression(expr, data, at)
        print(f"  accepted (unexpected): {expr} at {at}")
        results.append(False)
    except fx.ExprError as e:
        print(f"  {expr!r} at {at}: {str(e).splitlines()[0]}")

print(f"\n{sum(results)} of {len(results)} checks passed")
print("\ngenerated kernel for 'mag(grad(u)) * u' at L2 order-2 nodes:")
print(fx.compile_expression("mag(grad(u)) * u", hdata, at=l2).source)

