"""Отрисовка режимов «стрелки / линии тока / LIC» (без зависимости от Streamlit)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lic import make_field, make_noise, lic

FIELDS = {"uniform": "Равномерный поток", "source": "Источник", "sink": "Сток",
          "vortex": "Вихрь", "vortexsource": "Вихреисточник", "dipole": "Диполь", "saddle": "Седло (тест)",
          "superposition": "Суперпозиция"}


def render(mode, field_name, size=256, length_px=20, h_px=0.5, arrows_overlay=True, seed=0):
    """Возвращает (figure, время LIC в секундах или None)."""
    f = make_field(field_name)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(f.xmin, f.xmax); ax.set_ylim(f.ymin, f.ymax); ax.set_aspect("equal")
    elapsed = None

    def draw_arrows(color, n):
        xs = np.linspace(f.xmin + 0.05, f.xmax - 0.05, n); ys = np.linspace(f.ymin + 0.05, f.ymax - 0.05, n)
        X, Y = np.meshgrid(xs, ys)
        u, v, ok = f.direction(X.ravel(), Y.ravel())
        ax.quiver(X.ravel()[ok], Y.ravel()[ok], u[ok], v[ok], color=color, scale=n * 1.4, width=0.004)

    if mode == "Стрелки":
        draw_arrows("0.2", 21)
    elif mode == "Линии тока":
        xs = np.linspace(f.xmin, f.xmax, f.nx); ys = np.linspace(f.ymin, f.ymax, f.ny)
        ax.streamplot(xs, ys, np.where(f.valid, f.U, np.nan), np.where(f.valid, f.V, np.nan),
                      color="0.2", density=1.4, linewidth=0.7)
    elif mode == "LIC":
        import time
        noise = make_noise(size, size, seed=seed)
        t0 = time.perf_counter()
        img = lic(f, noise, length_px=length_px, h_px=h_px)
        elapsed = time.perf_counter() - t0
        cmap = plt.get_cmap("gray").copy(); cmap.set_bad("#d9534f")
        ax.imshow(np.ma.masked_invalid(img), cmap=cmap, extent=(f.xmin, f.xmax, f.ymin, f.ymax), vmin=0.2, vmax=0.8)
        if arrows_overlay:
            draw_arrows("#1f77b4", 11)
    else:
        raise ValueError(mode)
    ax.set_title(f"{FIELDS[field_name]} — {mode}")
    fig.tight_layout()
    return fig, elapsed
