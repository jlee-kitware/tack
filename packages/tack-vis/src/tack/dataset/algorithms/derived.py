"""The fields a dataset derives from its own geometry when none is stored under their
names (``register_derived_field``): its surface's normals, and each cell's measure
and centroid."""

from tack.dataset.algorithms._common import register_derived_field


def _normals(name, on):
    def compute(data):
        from tack.dataset.algorithms.surface_normals import surface_normals
        from tack.dataset.dataset import Field
        from tack.dataset.spaces import Values

        made = surface_normals(data).fields[name]
        return Field(Values(data, on), made.values, made.unit)
    return compute


def _cell_geometry(part):
    def compute(data):
        from tack.dataset.algorithms.cell_geometry import cell_geometry

        return getattr(cell_geometry(data), part)
    return compute


register_derived_field("normals", _normals("normals", "points"))
register_derived_field("cell_normals", _normals("cell_normals", "cells"))
register_derived_field("cell_measures", _cell_geometry("measure"))
register_derived_field("cell_centroids", _cell_geometry("centroid"))
