# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Fields a dataset derives from itself when it stores none of the name."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset.algorithms import _common


def _box():
    return td.extract_surface(td.uniform_grid((4, 3, 3), spacing=(0.5, 0.5, 0.5)))


def test_a_surface_derives_its_normals(backend):
    box = _box()
    made = td.surface_normals(box)
    for name in ("normals", "cell_normals"):
        np.testing.assert_array_equal(td.named_field(box, name).values.numpy(),
                                      made.fields[name].values.numpy())
    assert "cell_normals" not in box.fields
    assert td.named_field(box, "cell_normals") is td.named_field(box, "cell_normals")
    area = td.integrate(box, ["cell_measures"])
    np.testing.assert_allclose(area.measure, 2 * (1.5 * 1.0 + 1.5 * 1.0 + 1.0 * 1.0), rtol=1e-6)


def test_a_stored_field_comes_first(backend):
    box = _box()
    values = td.surface_normals(box).fields["cell_normals"].values
    box.fields["cell_normals"] = td.Field(td.Values(box, "cells"), values)
    assert td.named_field(box, "cell_normals") is box.fields["cell_normals"]


def test_registered_names_and_misses(backend):
    calls = []

    def compute(data):
        calls.append(1)
        return td.Field(td.Values(data, "cells"), tack.array(np.arange(data.num_cells, dtype=np.float32)))

    td.register_derived_field("cell_index", compute)
    try:
        box = _box()
        assert "cell_index" in td.derived_names()
        assert td.statistics(box, "cell_index").max == box.num_cells - 1
        td.statistics(box, "cell_index")
        assert len(calls) == 1
        with pytest.raises(KeyError):
            td.named_field(box, "nosuch")
    finally:
        del _common._DERIVED["cell_index"]
