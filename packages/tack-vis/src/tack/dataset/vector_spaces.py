"""Vector spaces of the lowest order on hexahedra and voxels: ``HDiv``, one value per face,
and ``HCurl``, one value per edge.

``HDiv`` (Raviart-Thomas) holds each face's integrated flux along the face's stored
normal, out of its side 0 -- the convention of ``Values(data, "faces", oriented=True)``,
``upwind_flux`` and ``net_flux`` -- and interpolates it inside each cell so that the
flux through every face is its value. ``HCurl`` (Nedelec, first kind) holds each
edge's tangential integral from its low point id to its high one, and interpolates
it so that the integral along every edge is its value. Both reproduce constant
fields exactly, and ``HDiv`` the fields whose x-component depends on x alone, and so
on: ``(a + b x, c + d y, e + f z)``.

A field view gives ``parametric_value(c, pc)``, the field pulled back to the
reference cell. ``hdiv_value`` and ``hcurl_value`` map it into the world through the
cell's geometry, as ``world_gradient`` maps ``parametric_gradient``: the contravariant
Piola map ``J v / det J``, which keeps fluxes, and the covariant ``J^-T v``, which
keeps tangential integrals.
"""

import tack
from tack.dataset.spaces import Space

__all__ = ["HCurl", "HDiv", "hcurl_value", "hdiv_value"]


def _check_shape(space, group):
    shape = group.shape
    if not (getattr(shape, "DIMENSION", 0) == 3 and shape.NUM_FACES == 6
            and shape.NUM_EDGES == 12):
        raise NotImplementedError(f"{type(space).__name__} is on hexahedra and voxels, "
                                  f"not {shape.__name__}")


class _HDivField:
    """An HDiv field's value in the reference cell: each face's outward flux times its
    reference function, which has unit flux through that face and none through the others."""

    @tack.func
    def parametric_value(self, c, pc):
        v = tack.vec3(0.0, 0.0, 0.0)
        for k in range(6):
            center = tack.vec3(0.0, 0.0, 0.0)
            for i in range(4):
                center += self.parametric_point(self.face_corner(k, i))
            # The reference outward normal: +-1 along the face's axis, 0 elsewhere.
            n = 2.0 * (center * 0.25) - 1.0
            g = n.dot(pc) + (0.0 if n.sum() > 0.0 else 1.0)
            flux = self.get(self.face_id(c, k)) * (1.0 if self.face_side(c, k) == 0 else -1.0)
            v += n * (g * flux)
        return v


class _HCurlField:
    """An HCurl field's value in the reference cell: each edge's tangential integral times
    its reference function, which has unit integral along that edge and none along the others."""

    @tack.func
    def parametric_value(self, c, pc):
        v = tack.vec3(0.0, 0.0, 0.0)
        for e in range(12):
            p0 = self.parametric_point(self.edge_corner(e, 0))
            d = self.parametric_point(self.edge_corner(e, 1)) - p0
            w = 1.0
            for a in range(3):
                across = p0[a] * pc[a] + (1.0 - p0[a]) * (1.0 - pc[a])
                w = w * (1.0 if d[a] * d[a] > 0.25 else across)
            integral = self.get(self.edge_id(c, e)) * float(self.edge_sign(c, e))
            v += d * (w * integral)
        return v


class HDiv(Space):
    """Lowest-order H(div) on hexahedra and voxels: one value per face, its integrated
    flux along the face's stored normal (out of side 0). Read with ``hdiv_value``."""

    on = "cells"
    mixin = _HDivField

    @classmethod
    def _params(cls):
        return {}

    @property
    def size(self):
        return self.topology.faces().num_faces

    def extra_mixins(self, group):
        from tack.dataset import views

        _check_shape(self, group)
        return (views._FaceIncidence,)

    def attributes(self, group=None, dtype=None):
        if group is None:
            return {}
        faces = self.topology.faces()
        return {"side_face": faces.side_face, "side_slot": faces.side_slot,
                "side_orientation": faces.side_orientation,
                "face_start": faces.group_starts[group.root]}


class HCurl(Space):
    """Lowest-order H(curl) on hexahedra and voxels: one value per edge, its tangential
    integral from the edge's low point id to its high one. Read with ``hcurl_value``."""

    on = "cells"
    mixin = _HCurlField

    @classmethod
    def _params(cls):
        return {}

    @property
    def size(self):
        return self.topology.edges().num_edges

    def extra_mixins(self, group):
        from tack.dataset import views

        _check_shape(self, group)
        return (views._EdgeIncidence,)

    def attributes(self, group=None, dtype=None):
        if group is None:
            return {}
        edges = self.topology.edges()
        return {"side_edge": edges.side_edge, "side_sign": edges.side_sign,
                "edge_start": edges.group_starts[group.root]}


@tack.func
def hdiv_value(cells, u, c, pc):
    """An HDiv field's value in the world: J v / det J, the map that keeps fluxes."""
    J = cells.geometry_jacobian(c, pc)
    return (J @ u.parametric_value(c, pc)) / J.determinant()


@tack.func
def hcurl_value(cells, u, c, pc):
    """An HCurl field's value in the world: J^-T v, the map that keeps tangential integrals."""
    return cells.geometry_jacobian(c, pc).inverse().transpose() @ u.parametric_value(c, pc)

