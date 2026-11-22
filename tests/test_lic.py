"""Автотесты базового LIC. Запуск из корня проекта: python -m pytest tests"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lic import make_field, make_noise, lic, lic_reference, pixel_centers


def test_vectorized_matches_reference():
    f = make_field("superposition", n=201)
    noise = make_noise(40, 40, seed=1)
    a = lic(f, noise, length_px=6, h_px=0.5)
    b = lic_reference(f, noise, length_px=6, h_px=0.5)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    m = ~np.isnan(a)
    assert np.max(np.abs(a[m] - b[m])) < 1e-12


def test_source_and_sink_give_same_image():
    noise = make_noise(128, 128, seed=0)
    a = lic(make_field("source"), noise, length_px=15)
    b = lic(make_field("sink"), noise, length_px=15)
    m = ~np.isnan(a)
    assert np.max(np.abs(a[m] - b[m])) < 1e-9


def test_uniform_flow_is_anisotropic():
    W = 192
    img = lic(make_field("uniform", angle_deg=30.0), make_noise(W, W), length_px=20)
    gy, gx = np.gradient(img)
    a = math.radians(30.0)
    along = np.abs(gx * math.cos(a) - gy * math.sin(a))[25:-25, 25:-25].mean()
    across = np.abs(gx * math.sin(a) + gy * math.cos(a))[25:-25, 25:-25].mean()
    assert along / across < 0.6


def test_singularity_is_masked():
    W = 128
    f = make_field("vortex", r_mask=0.06)
    img = lic(f, make_noise(W, W), length_px=10)
    X, Y = pixel_centers(f, W, W)
    r = np.hypot(X, Y)
    assert np.isnan(img[r < 0.04]).all()          # центр замаскирован
    assert not np.isnan(img[r > 0.2]).any()        # вне маски значения есть


def test_values_stay_in_texture_range():
    W = 96
    noise = make_noise(W, W)
    img = lic(make_field("vortexsource"), noise, length_px=10)
    m = ~np.isnan(img)
    assert img[m].min() >= noise.min() - 1e-12 and img[m].max() <= noise.max() + 1e-12


def test_rk2_keeps_vortex_streamline_on_circle():
    f = make_field("vortex", n=801)
    h = 1.0 * 2.0 / 512
    x = np.array([0.5]); y = np.array([0.0])
    u1, v1, _ = f.direction(x, y)
    for _ in range(40):
        u2, v2, _ = f.direction(x + 0.5 * h * u1, y + 0.5 * h * v1)
        x = x + h * u2; y = y + h * v2
        u1, v1, _ = f.direction(x, y)
    assert abs(np.hypot(x, y)[0] - 0.5) < 1e-5
