"""World derivatives of a field read inside another view's method. A view held by a
view is reached through its methods with plain values, so the cell geometry is read
here from the outer view's domain and passed in."""

import tack
from tack.dataset.algorithms.gradients import _in_plane


@tack.func
def held_gradient(jacobian, parametric, dimension):
    g = tack.vec3(0.0, 0.0, 0.0)
    if dimension == 3:
        g = jacobian.inverse().transpose() @ parametric
    elif dimension == 2:
        g = _in_plane(jacobian, parametric)
    return g


@tack.func
def held_hessian(jacobian, parametric, dimension):
    h = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    if dimension == 3:
        inverse = jacobian.inverse()
        h = inverse.transpose() @ parametric @ inverse
    return h
