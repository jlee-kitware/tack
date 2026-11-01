"""HDiv and HCurl reproduce the fields their spaces hold, through their Piola maps."""

import importlib

import numpy as np
import pytest

import tack
import tack.dataset as td


@tack.func
def _normal_linear(x):
    # Each component depends on its own coordinate alone: in lowest-order HDiv.
    return tack.vec3(1.0 + 2.0 * x[0], 3.0 - x[1], 0.5 + 4.0 * x[2])


@tack.func
def _constant(x):
    return tack.vec3(1.0, -2.0, 0.5)


@tack.kernel
def _face_fluxes(faces, centers, normals, areas, out):
    for f in faces:
        e = faces.entity_id(f)
        out[e] = _normal_linear(centers[e]).dot(normals[e]) * areas[e]


@tack.kernel
def _edge_integrals(edges, out):
    for i in edges:
        x0 = edges.corner_position(i, 0)
        x1 = edges.corner_position(i, 1)
        out[edges.entity_id(i)] = _constant(0.5 * (x0 + x1)).dot(x1 - x0)


@tack.kernel
def _hdiv_at(cells, u, px, py, pz, values, positions):
    for c in cells:
        pc = tack.vec3(px, py, pz)
        values[cells.entity_id(c)] = td.hdiv_value(cells, u, c, pc)
        positions[cells.entity_id(c)] = cells.position(c, pc)


@tack.kernel
def _hcurl_at(cells, u, px, py, pz, values, positions):
    for c in cells:
        pc = tack.vec3(px, py, pz)
        values[cells.entity_id(c)] = td.hcurl_value(cells, u, c, pc)
        positions[cells.entity_id(c)] = cells.position(c, pc)


def _grid():
    return td.uniform_grid((5, 4, 3), origin=(0.3, -0.2, 1.0), spacing=(0.5, 0.3, 0.7))


POINTS = [(0.5, 0.5, 0.5), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.2, 0.7, 0.4)]


@pytest.mark.parametrize("pc", POINTS)
def test_hdiv_reproduces_its_fields(backend, pc):
    data = _grid()
    space = td.HDiv(data)
    assert space.size == data.topology.faces().num_faces
    geometry = td.face_geometry(data)
    fluxes = tack.empty((space.size,), data.dtype)
    td.for_each(_face_fluxes, data, "faces", td.face_centers(data).values,
                geometry.normal.values, geometry.area.values, fluxes)
    u = td.Field(space, fluxes)
    values = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    positions = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    td.for_each(_hdiv_at, data, "cells", u, *pc, values, positions)
    x = positions.numpy()
    expected = np.stack([1 + 2 * x[:, 0], 3 - x[:, 1], 0.5 + 4 * x[:, 2]], axis=1)
    np.testing.assert_allclose(values.numpy(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("pc", POINTS)
def test_hcurl_reproduces_constants(backend, pc):
    data = _grid()
    space = td.HCurl(data)
    assert space.size == data.topology.edges().num_edges
    integrals = tack.empty((space.size,), data.dtype)
    td.for_each(_edge_integrals, data, "edges", integrals)
    u = td.Field(space, integrals)
    values = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    positions = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    td.for_each(_hcurl_at, data, "cells", u, *pc, values, positions)
    expected = np.tile([1.0, -2.0, 0.5], (data.num_cells, 1))
    np.testing.assert_allclose(values.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_world_gradient_is_public():
    module = importlib.import_module("tack.dataset.algorithms.gradients")
    assert td.world_gradient is module._world_gradient


def test_vector_spaces_refuse_other_shapes(backend):
    data = td.uniform_grid((4, 3, 1))           # quads
    with pytest.raises((NotImplementedError, TypeError, ValueError)):
        values = tack.zeros((td.HDiv(data).size,), data.dtype)
        u = td.Field(td.HDiv(data), values)
        out = tack.empty((data.num_cells,), tack.vec3(data.dtype))
        td.for_each(_hdiv_at, data, "cells", u, 0.5, 0.5, 0.5, out, out)

