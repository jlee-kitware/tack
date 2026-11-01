# tack-expr

Usage guide for `tack-expr`. The design, the checks and the measurements are in [README.md](README.md).

Field expressions for [Tack](https://github.com/berkgeveci/tack): VTK array calculator syntax over
`tack.dataset` fields, compiled to one Tack kernel per expression.

```python
import tack.dataset as td
import tack_expr as fx

speed_t = fx.evaluate("mag(V) * T", data, at="points")              # a td.Field in H1
outflow = fx.evaluate("fsum(flux)", data, at="cells")               # a td.Field in Constant
detail = fx.evaluate("mag(grad(u)) * u", data, at=td.L2(data, order=2))
```

The parser checks scalar and vector types and where each field can be read, and writes the
kernel's Python source, one assignment per operation. The source goes into a module in
`~/.cache/tack_expr` (`TACK_EXPR_CACHE` to change it), named by a hash of the source, and is
imported; modules unused for 30 days are removed at import (`TACK_EXPR_CACHE_DAYS`, 0 to keep them), so `@tack.kernel` compiles it as any other kernel. Constants are kernel arguments, so
expressions that differ only in their numbers share a kernel; no text the user typed reaches the
source.

## Python operators

The same expressions can be written with Python operators on `td.Field`s, which importing
`tack_expr` gives them:

```python
import numpy as np
import tack_expr as fx

U, V, W, T = (data.fields[n] for n in ("U", "V", "W", "T"))
expr = U + V / fx.curl(W)
result = expr.eval(data)                                    # at cell centers
speed = fx.mag(U).eval(data, at="points")
wave = (np.sin(T) * 2 + fx.where(T > 0.5, T, 0)).eval(data, at="points")
q = fx.qcrit(U).compile(data)                               # call q(data) as often as U changes
surface = (T * T + fx.mag2(U)).contour(data, 1.0)           # evaluated where the contour reads it
```

An expression is a tree of operations on the functions its fields represent, evaluated at the
location asked for, so `U * V` of order-2 fields is the product of the two functions at each
output node. `eval`, `compile`, `field`, `contour` and `function` write it in the calculator
syntax and compile it as text is compiled; `str(expr)` shows the text. The dataset passed in
gives the topology and the geometry, and every field must be on its topology.

A `td.Field` in an expression is that field, read wherever the expression runs, whether or not
`data.fields` holds it. `fx.ref("T")` is whatever `data.fields["T"]` is when it runs: a field the
solver replaces between calls, or each piece's own field over a `td.PartitionedDataSet`, which
takes refs only.

`+ - * / ** %`, unary `-`, `abs()`, `[k]` for a component and `.x .y .z` on expressions, and the
comparisons `< <= > >=` make expressions; `&`, `|` and `~` combine conditions, and
`fx.where(condition, a, b)` chooses. `==`, `and`, `or`, `not` and chained comparisons are
Python's own and are not expressions. NumPy's ufuncs (`np.sqrt`, `np.sin`, `np.arctan2`,
`np.minimum`, ...) make expressions of fields. The calculator's functions are `fx.grad`,
`fx.div`, `fx.curl`, `fx.qcrit`, `fx.laplacian`, `fx.mag`, `fx.mag2`, `fx.norm`, `fx.dot`,
`fx.cross`, `fx.clamp`, `fx.minimum`, `fx.maximum`, `fx.vec`, `fx.coords` and `fx.pi`, and
`fx.cells`, `fx.points`, `fx.owner`, `fx.neighbor` and `fx.fsum` of a field. `fx.compile_many`
takes expressions and text alike. `tests/test_expr_operators.py` checks each against the text.

## Locations

`at` is `"cells"`, `"points"`, `"faces"`, or a space: `L2` of any order, or `H1` for an expression
of continuous fields (`project=True` evaluates other expressions into `L2` and averages them into
`H1` with `td.project`). A field with a basis (`H1` of any order, `L2`, `Constant`) is evaluated
through its basis at the output's parametric coordinate. Values on cells, points or faces are read
by entity. Data on other entities needs an explicit operator:

| Operator | Meaning |
| --- | --- |
| `points(T)` | cell data averaged at each point, as `vtkCellDataToPointData`, inside the kernel on every mesh |
| `cells(T)` | point data averaged over each cell, as `vtkPointDataToCellData` |
| `owner(T)`, `neighbor(T)` | in a face expression, the cell value on side 0 or 1 |
| `fsum(F)` | in a cell expression, the outward sum of a face field over the cell's faces |
| `grad(u)`, `div(V)`, `curl(V)`, `qcrit(V)` | derivatives through each cell's basis and geometry |
| `laplacian(u)` | the trace of the Hessian; `div(grad(u))` is the same |

`grad()` and the others also take an expression, such as `grad(T * P)` or `qcrit(U / (1 + u))`.
The kernel carries each value's derivative alongside it (a gradient for a scalar, a Jacobian for a
vector) and applies the chain rule, so the result is exact wherever the expression is
differentiable, at any order. `tests/test_expr_derivatives.py` checks 16 such expressions against central
differences of the exact fields.

Second derivatives nest: `div(grad(u))`, `grad(grad(u)[0])`, `curl(curl(U))` and `laplacian(u * v)`
carry each scalar's Hessian as well. On a Tack with `tack.dataset.world_jet`, a field's value,
gradient and Hessian come from one pass over its cell's values, through its basis' second
derivatives and the geometry's, so they are exact on curved cells too, and a kernel with several
Hessians compiles in about 5 seconds the first time. On a Tack without it, a field's Hessian
differentiates its world gradient along the parametric axes with a fourth-order stencil, exact for
fields up to order 4 on cells that map affinely (uniform grids, tetrahedra) and fourth-order
accurate on curved ones, and such kernels take up to a minute and a half to compile. In float32
second derivatives round to about 1e-4 of the largest value either way. They are taken in solid
cells; on surfaces they are refused.
`tests/test_expr_second.py` checks them, in float32 and float64. `HDiv` and `HCurl` fields, on a Tack with those spaces, are read
through their Piola maps, and `div()` of an `HDiv` field is its outward flux over the cell's
volume.

A `td.PartitionedDataSet` runs piece by piece; with a ghost layer, what each piece owns is the
whole mesh's result. For pieces made by `td.ghost_cells` or `td.ghost_points`,
`update_ghosts=True` refreshes the ghost values of the fields an expression reads from the pieces
that own them (across MPI ranks too, through Tack's exchange), so a chain of evaluations that
average over neighbours stays right; `tests/test_expr_pieces.py` shows the difference.

## Several at once

`fx.compile_many({"vorticity": "mag(curl(V))", "q": "qcrit(V)", ...}, data, at="cells")` compiles
several expressions at one location into one kernel. Each field read, derivative and conversion
is done once for all of them, and calling the result gives `{name: td.Field}`. Four derived fields
of a velocity field on a 192 x 64 x 32 grid take 0.20 ms this way, 0.62 ms as four kernels.

## Without storing it

`fx.field(expression, data)` is a `td.Field` that computes the expression wherever an algorithm
reads it, through its view's `value` and `corner_value`; nothing is stored. `td.contour` takes it
as it takes a stored field, and `fx.contour(expression, data, value)` does so with the subdivision
its inputs' order calls for. Each crossing is the expression's own root along its sub-edge, and an
expression of continuous fields shares its points between cells, so its surface is watertight.
With no Bernstein bound to skip cells by, a subdivided contour refines every cell.
`tests/test_expr_contour.py` compares with contours of stored fields and with the exact level.

## In other kernels

`fx.function(expression, data)` returns a device class with `value(cells, c, pc)`, or with
`at="faces"` one with `value(faces, f)`, and
`fx.for_each(kernel, data, function, *args)` runs a kernel that calls it, so a threshold or a
reduction can use an expression without storing it.

## Checks and examples

The tests in `tests/` run with Tack's suite, once per available backend:

```bash
uv run pytest packages/tack-expr/tests
uv run pytest packages/tack-expr/tests/test_expr_vtk.py    # the comparisons with VTK
```

- `test_expr_vtk.py`: the calculator syntax against `vtkArrayCalculator`, conversions against
  VTK's filters, tetrahedra, polyhedra, float64, invalid values.
- `test_expr_spaces.py`: order-2 fields, face operators, `H1` targets, `HDiv` and `HCurl`.
- `test_expr_derivatives.py`, `test_expr_second.py`: first and second derivatives against
  central differences, `td.gradients` and `td.flow_quantities`.
- `test_expr_many.py`: `compile_many` against one kernel each.
- `test_expr_points.py`: `points()` on every mesh kind, with no `td.project` call.
- `test_expr_pieces.py`: pieces with a ghost layer, and the ghost refresh.
- `test_expr_contour.py`: contours of expressions.
- `test_expr_kernels.py`: expressions in other kernels, reuse after updates, the module cache.

The examples run on a field whose answers are known, and check them:

- `examples/50_field_expressions.py`: values, derivatives, and several expressions in one kernel.
- `examples/51_expressions_unstored.py`: an expression in a kernel of your own, in a contour,
  and on pieces, and the time one kernel for four fields saves.

```bash
uv run python packages/tack-expr/examples/50_field_expressions.py --arch cuda
```
