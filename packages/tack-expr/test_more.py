"""Checks for div/curl/qcrit, fused conversions, functions in kernels, node outputs,
polyhedra, float64, invalid values, staged gradients, reuse after updates, the cache."""

import sys

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import VTK_DOUBLE, vtkObject
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import (
    vtkAppendFilter,
    vtkArrayCalculator,
    vtkCellDataToPointData,
    vtkPointDataToCellData,
)
from vtkmodules.vtkFiltersGeneral import vtkDataSetTriangleFilter

import tack
import tack.dataset as td
import tack_expr as fx
from tack.interop.vtk import vtk_to_dataset

device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
tack.set_default_device(device)
vtkObject.GlobalWarningDisplayOff()
rng = np.random.default_rng(2)
results = []


def check(label, got, ref, tol=1e-4):
    got = np.asarray(got, dtype=np.float64).reshape(np.shape(ref))
    ref = np.asarray(ref, dtype=np.float64)
    err = float(np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref))))
    ok = err < tol
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label:60} max rel err {err:.1e}")


def image(dims, point=None, cell=None):
    img = vtkImageData()
    img.SetDimensions(dims)
    img.SetOrigin(0.5, -1.0, 2.0)
    img.SetSpacing(0.1, 0.2, 0.05)
    for attrs, arrays in ((img.GetPointData(), point), (img.GetCellData(), cell)):
        for name, a in (arrays or {}).items():
            arr = numpy_to_vtk(a, deep=True)
            arr.SetName(name)
            attrs.AddArray(arr)
    f = vtkAppendFilter()
    f.SetInputData(img)
    f.Update()
    return f.GetOutput()


def calc(grid, expr, cells=False, replace=None):
    c = vtkArrayCalculator()
    c.SetInputData(grid)
    c.SetAttributeTypeToCellData() if cells else c.SetAttributeTypeToPointData()
    c.AddScalarArrayName("T")
    c.AddScalarArrayName("P")
    c.AddVectorArrayName("V")
    c.SetFunction(expr)
    c.SetResultArrayName("r")
    c.SetResultArrayType(VTK_DOUBLE)
    if replace is not None:
        c.ReplaceInvalidValuesOn()
        c.SetReplacementValue(replace)
    c.Update()
    out = c.GetOutput()
    return vtk_to_numpy((out.GetCellData() if cells else out.GetPointData()).GetArray("r"))


dims = (16, 12, 9)
npts, ncells = dims[0] * dims[1] * dims[2], (dims[0] - 1) * (dims[1] - 1) * (dims[2] - 1)
pointvals = {"T": rng.random(npts, dtype=np.float32), "P": rng.random(npts, dtype=np.float32),
             "V": rng.random((npts, 3), dtype=np.float32) * 2 - 1}
cellvals = {"T": rng.random(ncells, dtype=np.float32), "P": rng.random(ncells, dtype=np.float32),
            "V": rng.random((ncells, 3), dtype=np.float32) * 2 - 1}
pgrid, cgrid = image(dims, point=pointvals), image(dims, cell=cellvals)
pdata, cdata = vtk_to_dataset(pgrid), vtk_to_dataset(cgrid)
print(f"device={device}")


# 1. div, curl, qcrit of an order-2 vector field
@tack.func
def vfun(x):
    return tack.vec3(x[0] * x[1], x[1] * x[1] + x[2], x[0] - 2.0 * x[2] * x[1])


@tack.func
def quadratic(x):
    return x[0] * x[0] + x[1] * x[2]


h = td.uniform_grid((6, 5, 4), origin=(0.2, -0.5, 1.0), spacing=(0.3, 0.25, 0.4))
h.fields["U"] = td.interpolate(h, vfun, td.H1(h, order=2), dtype=tack.vec3())
h.fields["u"] = td.interpolate(h, quadratic, td.H1(h, order=2))
fq = td.flow_quantities(h, td.gradients(h, "U", at="cells"))
print("vector derivatives (order-2 H1 field on voxels):")
check("div(U) at cells vs td.flow_quantities", fx.evaluate("div(U)", h).values.numpy(), fq.divergence.values.numpy())
check("curl(U) at cells vs td.flow_quantities", fx.evaluate("curl(U)", h).values.numpy(), fq.vorticity.values.numpy())
check("qcrit(U) at cells vs td.flow_quantities", fx.evaluate("qcrit(U)", h).values.numpy(), fq.q_criterion.values.numpy())
xc = fx.evaluate("coords", h).values.numpy()
check("div(U) vs exact y + 2y - 2y", fx.evaluate("div(U)", h).values.numpy(), xc[:, 1])
l2 = td.L2(h, order=2)
x2 = fx.evaluate("coords", h, at=l2).values.numpy()
check("mag(curl(U)) at L2 order-2 nodes vs exact",
      fx.evaluate("mag(curl(U))", h, at=l2).values.numpy(),
      np.linalg.norm(np.stack([-2 * x2[:, 2] - 1, 0 * x2[:, 0] - 1, -x2[:, 0]], axis=1), axis=1))

# 2. staged gradients of expressions
print("gradients of expressions:")
gu = td.gradients(h, "u", at="cells").values.numpy()
check("grad(2 * u + 1) vs 2 * td.gradients(u)", fx.evaluate("grad(2 * u + 1)", h).values.numpy(), 2 * gu)
uu = xc[:, 0] ** 2 + xc[:, 1] * xc[:, 2]
exact = 2 * uu[:, None] * np.stack([2 * xc[:, 0], xc[:, 2], xc[:, 1]], axis=1)
check("grad(u * u) by the chain rule vs exact 2 u grad(u)", fx.evaluate("grad(u * u)", h).values.numpy(), exact)

# 3. fused conversions
print("conversions inside the kernel:")
p2c = vtkPointDataToCellData()
p2c.SetInputData(pgrid)
p2c.Update()
src = fx.compile_expression("cells(T)", pdata).source
check("cells(T) in the kernel vs vtkPointDataToCellData", fx.evaluate("cells(T)", pdata).values.numpy(),
      vtk_to_numpy(p2c.GetOutput().GetCellData().GetArray("T")))
print(f"  INFO  cells(T) kernel calls td.project: {'project' in src}; corner loop in kernel: {'range(1, dom.NUM_POINTS)' in src}")
c2p = vtkCellDataToPointData()
c2p.SetInputData(cgrid)
c2p.Update()
pc = fx.compile_expression("points(T) * 2 + points(P)", cdata, at="points")
check("points(T) * 2 + points(P) through point links vs VTK",
      pc(cdata).values.numpy(),
      2 * vtk_to_numpy(c2p.GetOutput().GetPointData().GetArray("T"))
      + vtk_to_numpy(c2p.GetOutput().GetPointData().GetArray("P")))
print(f"  INFO  point expression is a loop over points: {pc.mode == 'pointloop'}")

# 4. an expression inside another kernel
print("expressions inside other kernels:")
q = fx.function("mag(V) * T + cells(P)", pdata)


@tack.kernel
def above(cells, q, out, level):
    for c in cells:
        out[cells.entity_id(c)] = 1.0 if q.value(cells, c, cells.parametric_center()) > level else 0.0


mask = tack.zeros((pdata.num_cells,))
fx.for_each(above, pdata, q, mask, 0.6)
ref = fx.evaluate("mag(V) * T + cells(P)", pdata).values.numpy()
check("threshold kernel calling the expression vs evaluate", mask.numpy(), (ref > 0.6).astype(float))

qf = fx.function("neighbor(T) - owner(T) * 2", cdata, at="faces")


@tack.kernel
def face_values(faces, q, out):
    for f in faces:
        out[faces.entity_id(f)] = q.value(faces, f)


face_out = tack.zeros((cdata.topology.faces().num_faces,))
fx.for_each(face_values, cdata, qf, face_out)
check("face kernel calling a face expression vs evaluate at faces", face_out.numpy(),
      fx.evaluate("neighbor(T) - owner(T) * 2", cdata, at="faces").values.numpy())

# 5. node outputs: simplices and H1 targets
print("node outputs:")
tf = vtkDataSetTriangleFilter()
tf.SetInputData(pgrid)
tf.Update()
tets = vtk_to_dataset(tf.GetOutput())
tl2 = td.L2(tets, order=2, basis="equispaced")
check("T at L2 order-2 nodes of tetrahedra vs td.project", fx.evaluate("T", tets, at=tl2).values.numpy(),
      td.project(tets, "T", tl2).values.numpy())
check("T * T + P at L2 order-1 nodes of tetrahedra vs node values",
      fx.evaluate("T * T + P", tets, at=td.L2(tets, order=1)).values.numpy(),
      (lambda t, p: t * t + p)(td.project(tets, "T", td.L2(tets, order=1)).values.numpy(),
                               td.project(tets, "P", td.L2(tets, order=1)).values.numpy()))
h1 = h.fields["u"].space
check("u * 2 + 1 into the H1 order-2 space vs its values", fx.evaluate("u * 2 + 1", h, at=h1).values.numpy(),
      h.fields["u"].values.numpy() * 2 + 1)
try:
    th1 = td.H1(tets, order=2)
    tets.fields["w"] = td.project(tets, "T", th1)
    check("w * 3 into H1 order 2 on tetrahedra vs its values",
          fx.evaluate("w * 3", tets, at=th1).values.numpy(), tets.fields["w"].values.numpy() * 3)
except Exception as e:
    results.append(False)
    print(f"  FAIL  H1 order 2 on tetrahedra: {type(e).__name__}: {str(e).splitlines()[0][:100]}")
try:
    tets.fields["C"] = td.Field(td.Constant(tets), tack.array(rng.random(tets.num_cells, dtype=np.float32)))
    fx.compile_expression("C + 1", tets, at=th1)
    print("  FAIL  cell data into an H1 space was accepted")
    results.append(False)
except fx.ExprError as e:
    print(f"  PASS  refused: {str(e).splitlines()[0]}")
    results.append(True)

# 6. polyhedra
print("polyhedral cells:")
poly = vtk_to_dataset(pgrid, polyhedral=True)
cpoly = vtk_to_dataset(cgrid, polyhedral=True)
print(f"  INFO  reference cells: {getattr(poly.topology, 'reference_cells', True)}")
check("T * 2 + P at points on polyhedra vs VTK", fx.evaluate("T * 2 + P", poly, at="points").values.numpy(),
      calc(pgrid, "T * 2 + P"))
check("cells(T) on polyhedra vs vtkPointDataToCellData", fx.evaluate("cells(T)", poly).values.numpy(),
      vtk_to_numpy(p2c.GetOutput().GetCellData().GetArray("T")))
check("T + P * 2 at cells on polyhedral cell data vs VTK", fx.evaluate("T + P * 2", cpoly).values.numpy(),
      calc(cgrid, "T + P * 2", cells=True))
geometry = td.face_geometry(cpoly)
centers = td.face_centers(cpoly).values.numpy()
flux = (np.sum(centers * geometry.normal.values.numpy(), axis=1) * geometry.area.values.numpy()).astype(np.float32)
cpoly.fields["flux"] = td.Field(td.Values(cpoly, "faces", oriented=True), tack.array(flux))
check("fsum(flux) on polyhedra vs td.net_flux", fx.evaluate("fsum(flux)", cpoly).values.numpy(),
      td.net_flux(cpoly, "flux").values.numpy())

# 7. float64
print("float64:")
p64 = vtk_to_dataset(pgrid, dtype=tack.f64)
for name in ("T", "P", "V"):
    f = p64.fields[name]
    p64.fields[name] = td.Field(f.space, f.values.astype(tack.f64) if hasattr(f.values, "astype") else f.values)
r64 = fx.evaluate("T % 0.3 + sqrt(P) * 1.1 + mag(V)", p64, at="points")
print(f"  INFO  output dtype {r64.values.dtype}")
check("T % 0.3 + sqrt(P) * 1.1 + mag(V) in float64 vs VTK", r64.values.numpy(),
      calc(pgrid, "T % 0.3 + sqrt(P) * 1.1 + mag(V)"), tol=1e-12)

# 8. invalid values
print("invalid values:")
check("ln(T - 0.5) with replacement -1 vs VTK",
      fx.evaluate("ln(T - 0.5)", pdata, at="points", replace_invalid=-1.0).values.numpy(),
      calc(pgrid, "ln(T - 0.5)", replace=-1.0))

# 9. reuse after the solver updates its arrays in place
print("reuse after updates:")
c_div = fx.compile_expression("div(U)", h)
c_pts = fx.compile_expression("points(T) + 1", cdata, at="points")
c_div(h)
c_pts(cdata)
newU = (rng.random((h.fields["U"].space.size, 3), dtype=np.float32) - 0.5)
U = tack.empty((h.fields["U"].space.size,), tack.vec3())
tack.copyto(U, newU)
h.fields["U"] = td.Field(h.fields["U"].space, U)
fq2 = td.flow_quantities(h, td.gradients(h, "U", at="cells"))
check("div(U) after replacing U's values", c_div(h).values.numpy(), fq2.divergence.values.numpy())
tack.copyto(cdata.fields["T"].values, rng.random(ncells, dtype=np.float32))
check("points(T) + 1 after updating T in place",
      c_pts(cdata).values.numpy(), td.project(cdata, "T", td.H1(cdata)).values.numpy() + 1)

# 10. the cache directory
print("cache:")
removed = fx.clear_cache()
again = fx.evaluate("T + P", pdata, at="points").values.numpy()
check(f"clear_cache removed {removed} modules; recompiled T + P", again, calc(pgrid, "T + P"))

# 11. pieces with a ghost layer: what each piece owns is the whole's
print("partitioned data (4 pieces, one ghost layer):")
whole = td.wavelet(10)
pieces = td.split(whole, 4, ghost_layers=1)
for data in [whole, *pieces]:
    g = td.face_geometry(data)
    ctr = td.face_centers(data).values.numpy()
    fl = (np.sum(ctr * g.normal.values.numpy(), axis=1) * g.area.values.numpy()).astype(np.float32)
    data.fields["flux"] = td.Field(td.Values(data, "faces", oriented=True), tack.array(fl))
for expr, at, name in (("RTData * 2 + 1", "points", "r1"), ("cells(RTData)", "cells", "C"),
                       ("mag(grad(RTData))", "cells", "r3"), ("points(C) - RTData", "points", "r4"),
                       ("fsum(flux) * C", "cells", "r5")):
    fx.evaluate(expr, whole, at=at, name=name)
    fx.evaluate(expr, pieces, at=at, name=name)
    a, b = td.statistics(whole, name), td.statistics(pieces, name)
    check(f"{expr} at {at}: owned (mean, min, max, n) vs whole", [b.mean, b.min, b.max, b.n],
          [a.mean, a.min, a.max, a.n], tol=1e-5)

# 12. HDiv and HCurl fields (the Tack branch's lowest-order vector spaces)
if hasattr(td, "HDiv"):
    print("HDiv and HCurl fields:")

    @tack.func
    def normal_linear(x):
        return tack.vec3(1.0 + 2.0 * x[0], 3.0 - x[1], 0.5 + 4.0 * x[2])

    @tack.kernel
    def face_fluxes(faces, centers, normals, areas, out):
        for f in faces:
            e = faces.entity_id(f)
            out[e] = normal_linear(centers[e]).dot(normals[e]) * areas[e]

    @tack.kernel
    def edge_integrals(edges, out):
        for i in edges:
            x0 = edges.corner_position(i, 0)
            x1 = edges.corner_position(i, 1)
            out[edges.entity_id(i)] = tack.vec3(1.0, -2.0, 0.5).dot(x1 - x0)

    v = td.uniform_grid((6, 5, 4), origin=(0.3, -0.2, 1.0), spacing=(0.5, 0.3, 0.7))
    g = td.face_geometry(v)
    fl = tack.empty((td.HDiv(v).size,), v.dtype)
    td.for_each(face_fluxes, v, "faces", td.face_centers(v).values, g.normal.values, g.area.values, fl)
    v.fields["F"] = td.Field(td.HDiv(v), fl)
    ei = tack.empty((td.HCurl(v).size,), v.dtype)
    td.for_each(edge_integrals, v, "edges", ei)
    v.fields["G"] = td.Field(td.HCurl(v), ei)
    xv = fx.evaluate("coords", v).values.numpy()
    check("F (HDiv) at cells vs the field it holds", fx.evaluate("F", v).values.numpy(),
          np.stack([1 + 2 * xv[:, 0], 3 - xv[:, 1], 0.5 + 4 * xv[:, 2]], axis=1))
    check("div(F) vs exact 2 - 1 + 4", fx.evaluate("div(F)", v).values.numpy(), np.full(v.num_cells, 5.0))
    x1 = fx.evaluate("coords", v, at=td.L2(v, order=1)).values.numpy()
    check("mag(F) at L2 order-1 nodes vs exact", fx.evaluate("mag(F)", v, at=td.L2(v, order=1)).values.numpy(),
          np.linalg.norm(np.stack([1 + 2 * x1[:, 0], 3 - x1[:, 1], 0.5 + 4 * x1[:, 2]], axis=1), axis=1))
    check("G * 2 (HCurl) at cells vs (2, -4, 1)", fx.evaluate("G * 2", v).values.numpy(),
          np.tile([2.0, -4.0, 1.0], (v.num_cells, 1)))
    if hasattr(td, "hcurl_curl"):
        @tack.kernel
        def rotation_integrals(edges, out):
            for i in edges:
                x0 = edges.corner_position(i, 0)
                x1 = edges.corner_position(i, 1)
                m = 0.5 * (x0 + x1)
                out[edges.entity_id(i)] = tack.vec3(-m[1], m[0], 0.0).dot(x1 - x0)

        ri = tack.empty((td.HCurl(v).size,), v.dtype)
        td.for_each(rotation_integrals, v, "edges", ri)
        v.fields["R"] = td.Field(td.HCurl(v), ri)
        check("curl(R) of the HCurl rotation (-y, x, 0) vs (0, 0, 2)", fx.evaluate("curl(R)", v).values.numpy(),
              np.tile([0.0, 0.0, 2.0], (v.num_cells, 1)))
    if hasattr(td, "hdiv_div"):
        @tack.func
        def rt(x):
            return tack.vec3(1.0 + 2.0 * x[0], 3.0 + 2.0 * x[1], 0.5 + 2.0 * x[2])

        @tack.kernel
        def rt_fluxes(faces, centers, normals, areas, out):
            for f in faces:
                e = faces.entity_id(f)
                out[e] = rt(centers[e]).dot(normals[e]) * areas[e]

        gt = td.face_geometry(tets)
        tf_ = tack.empty((td.HDiv(tets).size,), tets.dtype)
        td.for_each(rt_fluxes, tets, "faces", td.face_centers(tets).values, gt.normal.values, gt.area.values, tf_)
        tets.fields["F"] = td.Field(td.HDiv(tets), tf_)
        xt = fx.evaluate("coords", tets).values.numpy()
        check("F (HDiv) on tetrahedra vs the field it holds", fx.evaluate("F", tets).values.numpy(),
              np.stack([1 + 2 * xt[:, 0], 3 + 2 * xt[:, 1], 0.5 + 2 * xt[:, 2]], axis=1))
        check("div(F) on tetrahedra vs exact 6", fx.evaluate("div(F)", tets).values.numpy(),
              np.full(tets.num_cells, 6.0))
    try:
        fx.compile_expression("F", v, at="points")
        print("  FAIL  F at points was accepted")
        results.append(False)
    except fx.ExprError as e:
        print(f"  PASS  refused: {str(e).splitlines()[0]}")
        results.append(True)

print(f"\n{sum(results)} of {len(results)} checks passed")

