"""Проверки корректности LIC. Печатает числа, которые пойдут в текст."""
import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from lic import make_field, make_noise, lic, lic_reference, pixel_centers

res = {}

# 1) Векторизованная версия против эталона на чистом Python
N = 48
f = make_field("superposition", n=201)
noise = make_noise(N, N, seed=1)
a = lic(f, noise, length_px=8, h_px=0.5)
b = lic_reference(f, noise, length_px=8, h_px=0.5)
both = ~np.isnan(a) & ~np.isnan(b)
res["ref_nan_match"] = bool(np.array_equal(np.isnan(a), np.isnan(b)))
res["ref_max_abs_diff"] = float(np.max(np.abs(a[both] - b[both])))
print("эталон: NaN совпали:", res["ref_nan_match"], " max|diff| =", res["ref_max_abs_diff"])

# 2) Равномерный поток: вдоль линий изображение гладкое, поперёк — нет
W = 256
noise = make_noise(W, W, seed=0)
img = lic(make_field("uniform", angle_deg=30.0), noise, length_px=20)
gy, gx = np.gradient(img)                      # gy — по строкам (вниз)
ang = math.radians(30.0)
# направление потока в координатах (x вправо, y вверх); строки идут вниз
d_along = gx * math.cos(ang) - gy * math.sin(ang)
d_across = gx * math.sin(ang) + gy * math.cos(ang)
sl = (slice(30, -30), slice(30, -30))          # без краёв, где линии обрываются
res["uniform_ratio_along_across"] = float(np.mean(np.abs(d_along[sl])) / np.mean(np.abs(d_across[sl])))
print("равномерный поток: |d_along|/|d_across| =", res["uniform_ratio_along_across"])

# 3) Источник и сток: одинаковая текстура -> почти одинаковая картина
i_src = lic(make_field("source"), noise, length_px=20)
i_snk = lic(make_field("sink"), noise, length_px=20)
m = ~np.isnan(i_src) & ~np.isnan(i_snk)
res["source_vs_sink_max_diff"] = float(np.max(np.abs(i_src[m] - i_snk[m])))
res["source_vs_sink_mean_diff"] = float(np.mean(np.abs(i_src[m] - i_snk[m])))
print("источник vs сток: max|diff| = %.3g, mean|diff| = %.3g" %
      (res["source_vs_sink_max_diff"], res["source_vs_sink_mean_diff"]))

# 4) Вихрь: радиальный/тангенциальный градиент (изображение гладкое вдоль окружностей)
img = lic(make_field("vortex"), noise, length_px=20)
X, Y = pixel_centers(make_field("vortex"), W, W)
r = np.hypot(X, Y); gy, gx = np.gradient(img)
gyu = -gy
er_x, er_y = X / r, Y / r
et_x, et_y = -Y / r, X / r
g_rad = np.abs(gx * er_x + gyu * er_y)
g_tan = np.abs(gx * et_x + gyu * et_y)
ring = (r > 0.25) & (r < 0.8)                 # там, где свёртка не упирается в маску/границу
ring &= ~np.isnan(img)
res["vortex_ratio_tan_rad"] = float(np.nanmean(g_tan[ring]) / np.nanmean(g_rad[ring]))
print("вихрь: |g_tan|/|g_rad| =", res["vortex_ratio_tan_rad"])

# 5) Ошибка RK2 на линии тока вихря: дрейф радиуса при пошаговом интегрировании
f = make_field("vortex", n=801)
drift = {}
for h_px in (2.0, 1.0, 0.5, 0.25):
    h = h_px * 2.0 / W
    x = np.array([0.5]); y = np.array([0.0])
    u1, v1, _ = f.direction(x, y)
    L = 40.0 / h_px                           # путь 40 пикселей = 0.3125 ед. длины
    for _ in range(int(L)):
        xm = x + 0.5 * h * u1; ym = y + 0.5 * h * v1
        u2, v2, _ = f.direction(xm, ym)
        x = x + h * u2; y = y + h * v2
        u1, v1, _ = f.direction(x, y)
    drift[h_px] = float(abs(np.hypot(x, y)[0] - 0.5))
res["rk2_radius_drift"] = drift
print("дрейф радиуса (RK2, путь 40 px):", drift)

# 6) Седло: около нуля поле не должно ломать расчёт (нет NaN вне критической точки)
img = lic(make_field("saddle"), noise, length_px=20)
res["saddle_nan_fraction"] = float(np.isnan(img).mean())
print("седло: доля NaN =", res["saddle_nan_fraction"])

json.dump(res, open("checks.json", "w"), ensure_ascii=False, indent=1)
