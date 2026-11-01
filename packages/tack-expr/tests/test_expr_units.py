"""Units, the field dictionary and complex values, handled on the expression's tree before
code generation, so the kernels compute over real fields with the conversions in place.
Units are Tack's: Field.unit, DataSet.unit and tack.dataset.units."""

import pytest
import tack_expr as fx
from _expr_checks import assert_close, values

import tack
import tack.dataset as td
from tack.dataset import units as U


@tack.func
def f_x(x):
    return x[0] + 0.5


@tack.func
def f_y(x):
    return x[1] * x[1] + 0.25


@tack.func
def f_v(x):
    return tack.vec3(x[1] + 1.0, x[0] * x[2], 2.0 - x[0])


@pytest.fixture
def data(backend):
    d = td.uniform_grid((5, 4, 3), origin=(0.2, 0.3, 0.4), spacing=(0.25, 0.2, 0.3))
    h2 = td.H1(d, order=2)
    for name, f in (("p", f_x), ("q", f_y), ("u_real", f_x), ("u_imag", f_y)):
        d.fields[name] = td.interpolate(d, f, h2)
    for name in ("U", "W_re", "W_im"):
        d.fields[name] = td.interpolate(d, f_v, h2, dtype=tack.vec3())
    return d


def with_unit(data, name, unit):
    f = data.fields[name]
    data.fields[name] = td.Field(f.space, f.values, unit)


def test_units_meet_where_they_are_added(data):
    p = values("p", data, at="points")
    q = values("q", data, at="points")
    got = fx.evaluate("p", data, at="points", fields={"p": "kPa"}, units="Pa")
    assert_close(got.values.numpy(), p * 1e3)
    assert got.unit == "Pa"
    got = fx.evaluate("p + 1[atm]", data, at="points", fields={"p": "kPa"}, units="kPa")
    assert_close(got.values.numpy(), p + 101.325)
    got = fx.evaluate("p + q", data, at="points", fields={"p": "kPa", "q": "Pa"})
    assert_close(got.values.numpy(), p + q * 1e-3)
    assert got.unit == "kPa"
    got = fx.evaluate("p * q", data, at="points", fields={"p": "kPa", "q": "mm"})
    assert_close(got.values.numpy(), p * q)
    assert U.same(got.unit, U.canonical("kPa*mm"))
    got = fx.evaluate("p > 0.7", data, at="points", fields={"p": "kPa"})
    assert_close(got.values.numpy(), (p > 0.7).astype(float))


def test_a_fields_own_unit(data):
    with_unit(data, "p", "kPa")
    with_unit(data, "q", "Pa")
    p, q = values("p", data, at="points"), values("q", data, at="points")
    got = fx.evaluate("2 * p - q", data, at="points")
    assert_close(got.values.numpy(), 2 * p - q * 1e-3)
    assert got.unit == "kPa"
    got = (data.fields["p"] * 2 + data.fields["q"]).eval(data, at="points", units="Pa")
    assert_close(got.values.numpy(), p * 2000 + q)
    with pytest.raises(fx.ExprError, match="do not match"):
        fx.evaluate("p + coordsX", data, fields=fx.FieldDictionary(length="m"))


def test_mismatches_and_unknowns(data):
    with pytest.raises(fx.ExprError, match="do not match"):
        fx.evaluate("p + q", data, fields={"p": "Pa", "q": "m"})
    with pytest.raises(fx.ExprError, match="dimensionless"):
        fx.evaluate("exp(p)", data, fields={"p": "Pa"})
    got = fx.evaluate("p + q", data, fields={"p": "Pa"})
    assert got.unit is None
    with pytest.raises(fx.ExprError, match="unknown"):
        fx.evaluate("p + q", data, fields={"p": "Pa"}, units="Pa")
    with pytest.raises(fx.ExprError, match="not m/s"):
        fx.evaluate("p", data, fields={"p": "Pa"}, units="m/s")
    got = fx.evaluate("sin(p)", data, fields={"p": "deg"})
    assert_close(got.values.numpy(), values("sin(p * pi / 180)", data))


def test_derivatives_divide_by_the_geometry_unit(data):
    d = fx.FieldDictionary(length="mm").define("p", "kPa").define("U", "mm/s")
    got = fx.evaluate("grad(p)", data, fields=d, units="Pa/m")
    assert_close(got.values.numpy(), values("grad(p)", data) * 1e6)
    got = fx.evaluate("div(U)", data, fields=d, units="1/s")
    assert_close(got.values.numpy(), values("div(U)", data))
    got = fx.evaluate("laplacian(p * p)", data, fields=d, units="Pa^2/m^2")
    assert_close(got.values.numpy(), values("laplacian(p * p)", data) * 1e12, tol=1e-3)
    got = fx.evaluate("mag(coords)", data, fields=d, units="m")
    assert_close(got.values.numpy(), values("mag(coords)", data) * 1e-3)
    assert fx.evaluate("grad(p)", data, fields={"p": "kPa"}).unit is None


def test_offset_units(data):
    p = values("p", data, at="points")
    d = fx.FieldDictionary(length="m").define("p", "degC").define("q", "K")
    got = fx.evaluate("p", data, at="points", fields=d, units="K")
    assert_close(got.values.numpy(), p + 273.15)
    got = fx.evaluate("p", data, at="points", fields=d, units="degF")
    assert_close(got.values.numpy(), p * 1.8 + 32.0)
    got = fx.evaluate("grad(p)", data, fields=d)
    assert U.same(got.unit, "K/m")
    assert_close(got.values.numpy(), values("grad(p)", data))
    got = fx.evaluate("q - p", data, at="points", fields=d)
    assert_close(got.values.numpy(), values("q", data, at="points") - (p + 273.15))
    assert got.unit == "K"


def test_a_dictionary_maps_names_to_storage(data):
    data.fields["PRESSURE"] = data.fields["p"]
    for k, c in enumerate("XYZ"):
        U_ = data.fields["U"]
        data.fields[f"Velocity{c}"] = td.Field(U_.space, tack.array(U_.values.numpy()[:, k].copy()))
    names = fx.cgns()
    p = values("p", data, at="points")
    got = fx.evaluate("Pressure - 101.325[kPa]", data, at="points", fields=names)
    assert_close(got.values.numpy(), p - 101325.0)
    assert got.unit == "Pa"
    got = fx.evaluate("pressure", data, at="points", fields=names.with_units(Pressure="psi"), units="Pa")
    assert_close(got.values.numpy(), p * 6894.757293168361)
    got = fx.evaluate("mag(Velocity) * Density", data, at="points", fields=names.with_sources(Density="q"))
    assert_close(got.values.numpy(), values("mag(U) * q", data, at="points"))
    assert U.same(got.unit, "kg/(m^2*s)")
    with pytest.raises(fx.ExprError, match="do not match"):
        fx.evaluate("Pressure + VelocityX", data, fields=names)


@pytest.mark.parametrize(("text", "ref"), [
    ("abs(u)", "sqrt(u_real^2 + u_imag^2)"),
    ("phase(u, pi / 3)", "u_real * cos(pi / 3) - u_imag * sin(pi / 3)"),
    ("real(u * (1 + 2j))", "u_real - 2 * u_imag"),
    ("imag(u / (1 + 2j))", "(u_imag - 2 * u_real) / 5"),
    ("imag(conj(u) * u)", "0 * u_real"),
    ("arg(u)", "atan2(u_imag, u_real)"),
    ("real(exp(2j * u))", "exp(-2 * u_imag) * cos(2 * u_real)"),
    ("mag(W)", "sqrt(mag2(W_re) + mag2(W_im))"),
    ("real(dot(W, W))", "dot(W_re, W_re) - dot(W_im, W_im)"),
    ("mag(imag(cross(W, U)))", "mag(cross(W_im, U))"),
    ("real(grad(u))[0] + imag(laplacian(u * u))", "grad(u_real)[0] + laplacian(2 * u_real * u_imag)"),
    ("real(complex(p, q) * u)", "p * u_real - q * u_imag"),
])
def test_complex_values(data, text, ref):
    assert_close(fx.evaluate(text, data).values.numpy(), values(ref, data), tol=1e-3)


def test_complex_results_and_units(data):
    with pytest.raises(fx.ExprError, match="the result is complex"):
        fx.evaluate("u + 1", data)
    with pytest.raises(fx.ExprError, match="real values"):
        fx.evaluate("u > 1", data)
    got = fx.evaluate("abs(u)", data, fields={"u": "mm"}, units="m")
    assert_close(got.values.numpy(), values("sqrt(u_real^2 + u_imag^2)", data) * 1e-3)
    u = data.fields["u_real"] + 1j * data.fields["u_imag"]
    assert_close(fx.phase(u, 0.5).eval(data).values.numpy(), values("u_real * cos(0.5) - u_imag * sin(0.5)", data))
