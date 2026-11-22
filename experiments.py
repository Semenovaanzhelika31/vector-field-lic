"""Проверочные изображения и первичные замеры времени."""
import json, os, platform, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lic import make_field, make_noise, lic, lic_reference, pixel_centers

os.makedirs("figures", exist_ok=True)
TITLES = {"uniform": "Равномерный поток (30°)", "source": "Источник", "sink": "Сток",
          "vortex": "Потенциальный вихрь", "vortexsource": "Вихреисточник", "dipole": "Диполь",
          "saddle": "Седловое поле (u=x, v=−y)", "superposition": "Суперпозиция: равномерный поток + источник + сток"}
W = 512
noise = make_noise(W, W, seed=0)


def arrows_axes(ax, f, title, density=1.0):
    n = 21
    xs = np.linspace(f.xmin, f.xmax, n); ys = np.linspace(f.ymin, f.ymax, n)
    X, Y = np.meshgrid(xs, ys)
    u, v, ok = f.direction(X.ravel(), Y.ravel())
    ax.quiver(X.ravel()[ok], Y.ravel()[ok], u[ok], v[ok], color="0.2", scale=30, width=0.003)
    ax.set_title(title, fontsize=10); ax.set_aspect("equal"); ax.set_xlim(f.xmin, f.xmax); ax.set_ylim(f.ymin, f.ymax)


def stream_axes(ax, f, title):
    xs = np.linspace(f.xmin, f.xmax, f.nx); ys = np.linspace(f.ymin, f.ymax, f.ny)
    U = np.where(f.valid, f.U, np.nan); V = np.where(f.valid, f.V, np.nan)
    ax.streamplot(xs, ys, U, V, color="0.2", density=1.4, linewidth=0.7, arrowsize=0.8)
    ax.set_title(title, fontsize=10); ax.set_aspect("equal"); ax.set_xlim(f.xmin, f.xmax); ax.set_ylim(f.ymin, f.ymax)


def lic_axes(ax, f, img, title, with_arrows=False):
    cmap = plt.get_cmap("gray").copy(); cmap.set_bad("#d9534f")
    ax.imshow(np.ma.masked_invalid(img), cmap=cmap, extent=(f.xmin, f.xmax, f.ymin, f.ymax), vmin=0.2, vmax=0.8)
    if with_arrows:
        n = 11
        xs = np.linspace(f.xmin + 0.1, f.xmax - 0.1, n); ys = np.linspace(f.ymin + 0.1, f.ymax - 0.1, n)
        X, Y = np.meshgrid(xs, ys)
        u, v, ok = f.direction(X.ravel(), Y.ravel())
        ax.quiver(X.ravel()[ok], Y.ravel()[ok], u[ok], v[ok], color="#1f77b4", scale=22, width=0.004)
    ax.set_title(title, fontsize=10); ax.set_aspect("equal")


imgs = {}
for name in TITLES:
    f = make_field(name)
    img = lic(f, noise, length_px=20)
    imgs[name] = img
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
    arrows_axes(axs[0], f, "Стрелки"); stream_axes(axs[1], f, "Линии тока (Matplotlib)")
    lic_axes(axs[2], f, img, "LIC, L = 20 px")
    fig.suptitle(TITLES[name], fontsize=12); fig.tight_layout()
    fig.savefig(f"figures/fig_{name}.png", dpi=130); plt.close(fig)

# источник и сток: LIC одинаков, направление передаёт только стрелочное наложение
fig, axs = plt.subplots(1, 2, figsize=(9, 4.6))
for ax, name in zip(axs, ("source", "sink")):
    f = make_field(name); lic_axes(ax, f, imgs[name], TITLES[name] + ": LIC + стрелки", with_arrows=True)
fig.tight_layout(); fig.savefig("figures/fig_source_vs_sink.png", dpi=130); plt.close(fig)

# влияние длины свёртки
f = make_field("superposition")
fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, L in zip(axs, (5, 20, 60)):
    lic_axes(ax, f, lic(f, noise, length_px=L), f"L = {L} px")
fig.tight_layout(); fig.savefig("figures/fig_length.png", dpi=130); plt.close(fig)

# --------------------------------------------------------------------------
# Замеры времени (медиана из 3 запусков; только вычислительное ядро lic())
# --------------------------------------------------------------------------
def timeit(fn, repeats=3):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))

f = make_field("superposition")
timing = {"env": {"python": platform.python_version(), "numpy": np.__version__,
                  "cpu_count": os.cpu_count(), "machine": platform.machine()}}
timing["by_size_L20"] = {}
for n in (256, 512, 1024):
    nz = make_noise(n, n, seed=0)
    timing["by_size_L20"][n] = timeit(lambda: lic(f, nz, length_px=20), repeats=3 if n < 1024 else 1)
    print("size", n, timing["by_size_L20"][n], flush=True)
timing["by_L_512"] = {}
nz = make_noise(512, 512, seed=0)
for L in (10, 20, 40):
    timing["by_L_512"][L] = timeit(lambda: lic(f, nz, length_px=L))
    print("L", L, timing["by_L_512"][L], flush=True)
timing["by_h_512_L20"] = {}
for h in (2.0, 1.0, 0.5, 0.25):
    timing["by_h_512_L20"][h] = timeit(lambda: lic(f, nz, length_px=20, h_px=h))
    print("h", h, timing["by_h_512_L20"][h], flush=True)
# чистый Python против NumPy-версии, 64x64, L=10
n = 64; nz = make_noise(n, n, seed=0); fs = make_field("superposition", n=201)
timing["python_vs_numpy_64_L10"] = {
    "python": timeit(lambda: lic_reference(fs, nz, length_px=10), repeats=1),
    "numpy": timeit(lambda: lic(fs, nz, length_px=10), repeats=3)}
print(timing["python_vs_numpy_64_L10"])
json.dump(timing, open("timing.json", "w"), ensure_ascii=False, indent=1)
