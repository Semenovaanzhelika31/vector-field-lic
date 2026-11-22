"""Базовый LIC (линейная интегральная свёртка) для двумерных векторных полей.

Что реализовано:
  * поле на регулярной сетке + билинейная интерполяция компонент;
  * нормировка интерполированного вектора (параметризация длиной дуги);
  * локальная линия тока в обе стороны, схема RK2 (метод средней точки);
  * прямоугольное ядро, нормировка по фактически использованным весам;
  * остановка при выходе из области, попадании в маску и почти нулевом векторе.

Единицы: длина свёртки и шаг задаются в пикселях выходного изображения.
Пиксели считаются квадратными.
"""
import math
import numpy as np

EPS_ZERO = 1e-6   # порог "почти нулевого" интерполированного вектора


# --------------------------------------------------------------------------
# Поле на регулярной сетке
# --------------------------------------------------------------------------
class GridField:
    """Компоненты U, V в узлах регулярной сетки (оси: строка = y по возрастанию)."""

    def __init__(self, U, V, valid, bounds):
        self.U = np.where(valid, U, 0.0).astype(float)
        self.V = np.where(valid, V, 0.0).astype(float)
        self.valid = valid.astype(bool)
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.ny, self.nx = U.shape

    def direction(self, x, y):
        """Единичный вектор направления в точках (x, y).

        Возвращает (ux, uy, ok); ok=False вне области, рядом с маской
        (любой из четырёх узлов недопустим) и при |v| < EPS_ZERO.
        """
        inside = (x >= self.xmin) & (x <= self.xmax) & (y >= self.ymin) & (y <= self.ymax)
        fx = np.clip((x - self.xmin) / (self.xmax - self.xmin) * (self.nx - 1), 0, self.nx - 1 - 1e-9)
        fy = np.clip((y - self.ymin) / (self.ymax - self.ymin) * (self.ny - 1), 0, self.ny - 1 - 1e-9)
        i0 = fx.astype(np.intp)
        j0 = fy.astype(np.intp)
        tx = fx - i0
        ty = fy - j0
        w00 = (1 - tx) * (1 - ty)
        w10 = tx * (1 - ty)
        w01 = (1 - tx) * ty
        w11 = tx * ty

        def bil(A):
            return (w00 * A[j0, i0] + w10 * A[j0, i0 + 1] +
                    w01 * A[j0 + 1, i0] + w11 * A[j0 + 1, i0 + 1])

        u = bil(self.U)
        v = bil(self.V)
        ok = inside & (self.valid[j0, i0] & self.valid[j0, i0 + 1] &
                       self.valid[j0 + 1, i0] & self.valid[j0 + 1, i0 + 1])
        mag = np.hypot(u, v)
        ok &= mag > EPS_ZERO
        safe = np.where(ok, mag, 1.0)
        return u / safe, v / safe, ok

    def magnitude_image(self, W, H):
        """|v| (с билинейной интерполяцией) на пикселях W x H, NaN в маске."""
        x, y = pixel_centers(self, W, H)
        u = self.U; v = self.V
        fx = (x - self.xmin) / (self.xmax - self.xmin) * (self.nx - 1)
        fy = (y - self.ymin) / (self.ymax - self.ymin) * (self.ny - 1)
        fx = np.clip(fx, 0, self.nx - 1 - 1e-9); fy = np.clip(fy, 0, self.ny - 1 - 1e-9)
        i0 = fx.astype(int); j0 = fy.astype(int); tx = fx - i0; ty = fy - j0
        def bil(A):
            return ((1-tx)*(1-ty)*A[j0, i0] + tx*(1-ty)*A[j0, i0+1] +
                    (1-tx)*ty*A[j0+1, i0] + tx*ty*A[j0+1, i0+1])
        m = np.hypot(bil(u), bil(v))
        ok = (self.valid[j0, i0] & self.valid[j0, i0+1] & self.valid[j0+1, i0] & self.valid[j0+1, i0+1])
        return np.where(ok, m, np.nan)


def pixel_centers(field, W, H):
    dx = (field.xmax - field.xmin) / W
    dy = (field.ymax - field.ymin) / H
    xs = field.xmin + (np.arange(W) + 0.5) * dx
    ys = field.ymax - (np.arange(H) + 0.5) * dy      # строка 0 — сверху
    return np.meshgrid(xs, ys)


# --------------------------------------------------------------------------
# LIC (векторизовано по всем пикселям)
# --------------------------------------------------------------------------
def _sample_texture(tex, field, x, y, W, H):
    dx = (field.xmax - field.xmin) / W
    dy = (field.ymax - field.ymin) / H
    fx = np.clip((x - field.xmin) / dx - 0.5, 0, W - 1 - 1e-9)
    fy = np.clip((field.ymax - y) / dy - 0.5, 0, H - 1 - 1e-9)
    i0 = fx.astype(np.intp); j0 = fy.astype(np.intp)
    tx = fx - i0; ty = fy - j0
    return ((1 - tx) * (1 - ty) * tex[j0, i0] + tx * (1 - ty) * tex[j0, i0 + 1] +
            (1 - tx) * ty * tex[j0 + 1, i0] + tx * ty * tex[j0 + 1, i0 + 1])


def lic(field, noise, length_px=20.0, h_px=0.5, return_stats=False):
    """LIC с прямоугольным ядром и RK2.

    noise      — шумовая текстура (H, W), её размер задаёт размер изображения;
    length_px  — длина свёртки в каждую сторону (пиксели);
    h_px       — шаг интегрирования (пиксели).
    Возвращает изображение (H, W); NaN в пикселях, где поле недопустимо.
    """
    H, W = noise.shape
    dx = (field.xmax - field.xmin) / W
    dy = (field.ymax - field.ymin) / H
    assert abs(dx - dy) < 1e-12 * max(dx, dy), "нужны квадратные пиксели"
    h = h_px * dx                                  # шаг в единицах области
    n_steps = int(round(length_px / h_px))

    cx, cy = pixel_centers(field, W, H)
    cx = cx.ravel(); cy = cy.ravel()
    t = noise.astype(float)
    _, _, center_ok = field.direction(cx, cy)

    acc = _sample_texture(t, field, cx, cy, W, H) * center_ok
    wsum = center_ok.astype(float)
    steps_used = np.zeros_like(cx)

    for s in (+1.0, -1.0):
        x = cx.copy(); y = cy.copy()
        u1, v1, _ = field.direction(x, y)
        alive = center_ok.copy()
        for _ in range(n_steps):
            xm = x + 0.5 * h * s * u1
            ym = y + 0.5 * h * s * v1
            u2, v2, okm = field.direction(xm, ym)
            xn = x + h * s * u2
            yn = y + h * s * v2
            u1n, v1n, okn = field.direction(xn, yn)
            alive &= okm & okn
            if not alive.any():
                break
            acc += np.where(alive, _sample_texture(t, field, xn, yn, W, H), 0.0)
            wsum += alive
            steps_used += alive
            x, y, u1, v1 = xn, yn, u1n, v1n

    out = np.where(center_ok, acc / np.where(wsum > 0, wsum, 1.0), np.nan).reshape(H, W)
    if return_stats:
        return out, steps_used.reshape(H, W)
    return out


# --------------------------------------------------------------------------
# Эталон на чистом Python (для проверки векторизованной версии)
# --------------------------------------------------------------------------
def lic_reference(field, noise, length_px=20.0, h_px=0.5):
    H, W = noise.shape
    dx = (field.xmax - field.xmin) / W
    h = h_px * dx
    n_steps = int(round(length_px / h_px))

    def direction(x, y):
        if not (field.xmin <= x <= field.xmax and field.ymin <= y <= field.ymax):
            return 0.0, 0.0, False
        fx = min(max((x - field.xmin) / (field.xmax - field.xmin) * (field.nx - 1), 0), field.nx - 1 - 1e-9)
        fy = min(max((y - field.ymin) / (field.ymax - field.ymin) * (field.ny - 1), 0), field.ny - 1 - 1e-9)
        i = int(fx); j = int(fy); tx = fx - i; ty = fy - j
        if not (field.valid[j, i] and field.valid[j, i+1] and field.valid[j+1, i] and field.valid[j+1, i+1]):
            return 0.0, 0.0, False
        def b(A):
            return ((1-tx)*(1-ty)*A[j, i] + tx*(1-ty)*A[j, i+1] + (1-tx)*ty*A[j+1, i] + tx*ty*A[j+1, i+1])
        u, v = b(field.U), b(field.V)
        m = math.hypot(u, v)
        if m <= EPS_ZERO:
            return 0.0, 0.0, False
        return u / m, v / m, True

    def tex(x, y):
        fx = min(max((x - field.xmin) / dx - 0.5, 0), W - 1 - 1e-9)
        fy = min(max((field.ymax - y) / dx - 0.5, 0), H - 1 - 1e-9)
        i = int(fx); j = int(fy); tx = fx - i; ty = fy - j
        return ((1-tx)*(1-ty)*noise[j, i] + tx*(1-ty)*noise[j, i+1] +
                (1-tx)*ty*noise[j+1, i] + tx*ty*noise[j+1, i+1])

    out = np.full((H, W), np.nan)
    for r in range(H):
        for c in range(W):
            x0 = field.xmin + (c + 0.5) * dx
            y0 = field.ymax - (r + 0.5) * dx
            _, _, ok = direction(x0, y0)
            if not ok:
                continue
            acc = tex(x0, y0); w = 1.0
            for s in (1.0, -1.0):
                x, y = x0, y0
                u1, v1, _ = direction(x, y)
                for _ in range(n_steps):
                    u2, v2, okm = direction(x + 0.5*h*s*u1, y + 0.5*h*s*v1)
                    if not okm:
                        break
                    xn = x + h*s*u2; yn = y + h*s*v2
                    u1n, v1n, okn = direction(xn, yn)
                    if not okn:
                        break
                    acc += tex(xn, yn); w += 1.0
                    x, y, u1, v1 = xn, yn, u1n, v1n
            out[r, c] = acc / w
    return out


# --------------------------------------------------------------------------
# Тестовые поля
# --------------------------------------------------------------------------
def _point(kind, X, Y, x0=0.0, y0=0.0, k=1.0):
    dxx = X - x0; dyy = Y - y0
    r2 = dxx**2 + dyy**2
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "source":
            return k * dxx / r2, k * dyy / r2
        if kind == "vortex":
            return -k * dyy / r2, k * dxx / r2
        if kind == "dipole":
            return k * (dyy**2 - dxx**2) / r2**2, k * (-2 * dxx * dyy) / r2**2
    raise ValueError(kind)


def make_field(name, n=401, bounds=(-1.0, 1.0, -1.0, 1.0), r_mask=0.06, angle_deg=30.0):
    """Тестовые поля. Сингулярные центры маскируются кругом радиуса r_mask."""
    xmin, xmax, ymin, ymax = bounds
    xs = np.linspace(xmin, xmax, n); ys = np.linspace(ymin, ymax, n)
    X, Y = np.meshgrid(xs, ys)
    valid = np.ones_like(X, dtype=bool)
    a = math.radians(angle_deg)

    if name == "uniform":
        U = np.full_like(X, math.cos(a)); V = np.full_like(X, math.sin(a))
    elif name in ("source", "sink"):
        U, V = _point("source", X, Y, k=1.0 if name == "source" else -1.0)
        valid &= np.hypot(X, Y) > r_mask
    elif name == "vortex":
        U, V = _point("vortex", X, Y)
        valid &= np.hypot(X, Y) > r_mask
    elif name == "dipole":
        U, V = _point("dipole", X, Y)
        valid &= np.hypot(X, Y) > r_mask
    elif name == "saddle":
        U, V = X.copy(), -Y.copy()                 # нулевая критическая точка, не сингулярность
    elif name == "vortexsource":                   # источник + вихрь: логарифмические спирали
        u1, v1 = _point("source", X, Y, k=1.0)
        u2, v2 = _point("vortex", X, Y, k=1.0)
        U = u1 + u2; V = v1 + v2
        valid &= np.hypot(X, Y) > r_mask
    elif name == "superposition":
        u1, v1 = _point("source", X, Y, -0.5, 0.0, 1.0)
        u2, v2 = _point("source", X, Y, 0.5, 0.0, -1.0)
        U = 0.5 + u1 + u2; V = v1 + v2
        valid &= (np.hypot(X + 0.5, Y) > r_mask) & (np.hypot(X - 0.5, Y) > r_mask)
    else:
        raise ValueError(name)
    U = np.where(valid, U, 0.0); V = np.where(valid, V, 0.0)
    return GridField(U, V, valid, bounds)


def make_noise(W, H, seed=0):
    return np.random.default_rng(seed).random((H, W))
