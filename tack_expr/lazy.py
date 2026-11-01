"""An expression as a field evaluated where an algorithm reads it, never stored.

fx.field(text, data) is a td.Field whose view computes the expression at each point an
algorithm asks for: value(c, pc), corner_value(c, j), and an unbounded value_range for
the algorithms that skip cells by their Bernstein bound. td.contour takes it as it takes
a stored field; fx.contour does so with the subdivision its inputs' order calls for.
"""

import itertools
import re

import numpy as np
import tack
import tack.dataset as td
from tack.dataset.spaces import Space

from . import parser as P

_keys = itertools.count(1)


class ExpressionSpace(Space):
    """The space of an expression field: one notional value per cell, read through the
    generated view. continuous when the expression is pointwise in continuous fields, so
    contour shares its points between cells."""

    on = "cells"
    interpolated = True
    evaluated = True

    @classmethod
    def _params(cls, key=0):
        return {"key": key}

    @property
    def size(self):
        return self.topology.num_cells

    def mixin_for(self, key, shape=None):
        return self._mixin

    def attributes(self, group=None, dtype=None):
        if group is None:
            return {}
        data = self._data
        out = {"dom": data.domain_view("cells", group)}
        for prefix, args, consts in self._parts:
            memo = {}
            for k, (kind, get) in enumerate(args):
                payload = get(data, memo)
                out[f"{prefix}{k}"] = payload.view(group) if kind == "field" else payload
            for k, v in enumerate(consts):
                out[f"{prefix.lower()}c{k}"] = float(P._scalar(data, v))
        return out


def _method(lines, result, prefix, name, signature, head=()):
    swap = re.compile(r"\b(A\d+|c\d+|dom)\b")

    def sub(text):
        def rename(m):
            token = m.group(1)
            if token == "dom":
                return "self.dom"
            return f"self.{prefix}{token[1:]}" if token[0] == "A" else f"self.{prefix.lower()}{token}"
        return swap.sub(rename, text)

    body = list(head) + [sub(line) for line in lines] + [f"return {sub(result)}"]
    return f"\n    @tack.func\n    def {name}({signature}):\n" + "".join(f"        {b}\n" for b in body)


def field(text, data):
    """The expression as a td.Field evaluated wherever it is read: by contour, clip,
    probe and the other algorithms that read a field through its view."""
    c = P._Compiler(text, data, "cells")
    result, rtype = c.parse()
    if rtype != P.S:
        raise P.ExprError("a field to contour or clip is a scalar expression", 0, text)
    parts = [("A", list(c.args), list(c.consts))]
    methods = _method(c.lines, result, "A", "value", "self, c, pc")
    imports = P._imports(c)
    continuous = False
    try:
        k = P._Compiler(text, data, "corners")
        corner, _ = k.parse()
        parts.append(("B", list(k.args), list(k.consts)))
        head = ["pc = self.dom.parametric_point(j)"] if k.uses_pc else []
        methods += _method(k.lines, corner, "B", "corner_value", "self, c, j", head)
        imports = "\n".join(sorted(set(imports.splitlines()) | set(P._imports(k).splitlines()))) + "\n"
        continuous = not c.uses_derivatives and not c.uses_world_gradient
    except (P.ExprError, P._NeedCorners):
        methods += ("\n    @tack.func\n    def corner_value(self, c, j):\n"
                    "        return self.value(c, self.dom.parametric_point(j))\n")
    methods += ("\n    @tack.func\n    def value_range(self, c):\n"
                "        return -3.0e38, 3.0e38\n")
    # Exact crossings read stored fields' entity values through dof; an expression has none
    # and is read through value there, but the method has to exist for the kernel to build.
    methods += ("\n    @tack.func\n    def dof(self, c, k):\n"
                "        return 0.0 * self.dom.parametric_center()[0]\n")
    source = imports + "\n\nclass Mixin:\n" + methods
    space = ExpressionSpace(data, key=next(_keys))
    space._mixin = P._load(source, "Mixin")
    space._data = data
    space._parts = parts
    space.continuous = continuous
    space.source = source
    space.text = text
    real = np.float64 if data.dtype is tack.f64 else np.float32
    return td.Field(space, tack.zeros((data.num_cells,), tack.f64 if real is np.float64 else tack.f32))


def contour(text, data, value, subdivision=None, **kwargs):
    """td.contour of the expression, evaluated where the contour reads it. Inputs above
    order 1 subdivide each cell by their largest order (or subdivision), and each crossing
    is the expression's own root along its sub-edge (exact=True, as td.contour's default),
    or with exact=False the crossing of the lattice values. With no Bernstein bound to skip
    cells by, every cell is refined."""
    f = field(text, data)
    orders = [getattr(data.fields[n].space, "order", 1) for n in sorted(_names(text, data))]
    order = max([1] + [o for o in orders if isinstance(o, int)])
    if subdivision is None and order > 1:
        subdivision = order
    if subdivision is not None and subdivision > 1:
        return td.contour(data, f, value, subdivision=subdivision, **kwargs)
    return td.contour(data, f, value, **kwargs)


def _names(text, data):
    c = P._Compiler(text, data, "cells")
    c.parse()
    return c.fields_read

