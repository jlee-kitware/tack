# tack-expr

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
imported, so `@tack.kernel` compiles it as any other kernel. Constants are kernel arguments, so
expressions that differ only in their numbers share a kernel; no text the user typed reaches the
source.

## Locations

`at` is `"cells"`, `"points"`, `"faces"`, or a space: `L2` of any order, or `H1` for an expression
of continuous fields. A field with a basis (`H1` of any order, `L2`, `Constant`) is evaluated
through its basis at the output's parametric coordinate. Values on cells, points or faces are read
by entity. Data on other entities needs an explicit operator:

| Operator | Meaning |
| --- | --- |
| `points(T)` | cell data averaged at each point, as `vtkCellDataToPointData`, inside the kernel on every mesh but polyhedra |
| `cells(T)` | point data averaged over each cell, as `vtkPointDataToCellData` |
| `owner(T)`, `neighbor(T)` | in a face expression, the cell value on side 0 or 1 |
| `fsum(F)` | in a cell expression, the outward sum of a face field over the cell's faces |
| `grad(u)`, `div(V)`, `curl(V)`, `qcrit(V)` | derivatives through each cell's basis and geometry |

`grad()` and the others also take an expression, such as `grad(T * P)` or `qcrit(U / (1 + u))`.
The kernel carries each value's derivative alongside it (a gradient for a scalar, a Jacobian for a
vector) and applies the chain rule, so the result is exact wherever the expression is
differentiable, at any order. `test_derivatives.py` checks 16 such expressions against central
differences of the exact fields. `HDiv` and `HCurl` fields, on a Tack with those spaces, are read
through their Piola maps, and `div()` of an `HDiv` field is its outward flux over the cell's
volume.

A `td.PartitionedDataSet` runs piece by piece; with a ghost layer, what each piece owns is the
whole mesh's result.

## Several at once

`fx.compile_many({"vorticity": "mag(curl(V))", "q": "qcrit(V)", ...}, data, at="cells")` compiles
several expressions at one location into one kernel. Each field read, derivative and conversion
is done once for all of them, and calling the result gives `{name: td.Field}`. Four derived fields
of a velocity field on a 192 x 64 x 32 grid take 0.20 ms this way, 0.62 ms as four kernels.

## In other kernels

`fx.function(expression, data)` returns a device class with `value(cells, c, pc)`, and
`fx.for_each(kernel, data, function, *args)` runs a kernel that calls it, so a threshold or a
reduction can use an expression without storing it.

## Checks

- `test_dataset.py`: the calculator syntax against `vtkArrayCalculator`, conversions against VTK's
  filters, tetrahedra, order-2 fields, face operators.
- `test_more.py`: `div`/`curl`/`qcrit` against `td.flow_quantities`, fused conversions, functions
  in kernels, node outputs, polyhedra, float64, invalid values, reuse after updates, partitioned
  data, `HDiv`/`HCurl`.

```bash
~/gitsync/tack/.venv/bin/python test_dataset.py cuda
~/gitsync/tack/.venv/bin/python test_more.py cpu
```

