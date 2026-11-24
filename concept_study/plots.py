"""Plotting helpers (matplotlib), one function per figure type.

Colours: fixed categorical order blue / orange / aqua (last_3, all_history, third series),
one-hue blue ramp for magnitudes, orange-grey-blue diverging map for signed values.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
SERIES = (BLUE, ORANGE, AQUA)
VARIANT_COLOURS = {"last_3": BLUE, "all_history": ORANGE}
DIVERGING = LinearSegmentedColormap.from_list("orange_grey_blue", [ORANGE, "#e8e6e1", BLUE])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_ramp", ["#f2f6fc", BLUE, "#0d3b73"])


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 140, "font.size": 9, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
        "lines.linewidth": 1.6, "lines.markersize": 4, "legend.frameon": False,
    })


def grid(n: int, columns: int = 3, width: float = 4.2, height: float = 3.2, **kwargs):
    rows = max(1, int(np.ceil(n / columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows), squeeze=False, **kwargs)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat[:n]


def heatmap(ax, matrix, row_labels, column_labels, title="", diverging=False, limit=None,
            annotate=True, fmt="{:.2f}", colorbar=True):
    """Matrix heatmap; NaN cells are left blank (undefined stays visible as empty)."""
    matrix = np.asarray(matrix, dtype=float)
    if diverging:
        if limit is None:
            limit = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
        image = ax.imshow(matrix, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    else:
        image = ax.imshow(matrix, cmap=SEQUENTIAL, aspect="auto", vmax=limit)
    ax.set_xticks(range(len(column_labels)), column_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.grid(False)
    ax.set_title(title)
    if annotate and matrix.size <= 400:
        for (i, j), value in np.ndenumerate(matrix):
            if np.isfinite(value):
                ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=7, color="#0b0b0b")
    if colorbar:
        plt.colorbar(image, ax=ax, fraction=0.04)
    return image


def identity(ax, published, recomputed, title, label="published"):
    """Recomputed versus published values on the identity line, with the largest absolute difference."""
    published, recomputed = np.asarray(published, float), np.asarray(recomputed, float)
    ok = np.isfinite(published) & np.isfinite(recomputed)
    ax.scatter(published[ok], recomputed[ok], s=10, color=BLUE, alpha=0.6, edgecolor="none")
    if ok.any():
        low, high = np.min(published[ok]), np.max(published[ok])
        ax.plot([low, high], [low, high], color=GREY, lw=1)
        worst = np.max(np.abs(published[ok] - recomputed[ok]))
        ax.set_title(f"{title}\n{ok.sum()} values, max |difference| {worst:.2g}")
    ax.set_xlabel(label)
    ax.set_ylabel("recomputed")


def forest(ax, labels, value, low, high, colours=None, title="", xlabel="", zero=True):
    """Point estimates with 95% intervals, one row per label (top to bottom in the given order)."""
    y = np.arange(len(labels))[::-1]
    colours = colours if colours is not None else [BLUE] * len(labels)
    for k in range(len(labels)):
        ax.plot([low[k], high[k]], [y[k], y[k]], color=colours[k], lw=1.4)
        ax.plot(value[k], y[k], "o", color=colours[k], ms=4)
    if zero:
        ax.axvline(0, color=GREY, lw=1)
    ax.set_yticks(y, labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)


def band(ax, x, value, low, high, colour=BLUE, label=None):
    """Line with a shaded 95% interval."""
    ax.plot(x, value, color=colour, label=label, marker="o", ms=3)
    ax.fill_between(x, low, high, color=colour, alpha=0.15, linewidth=0)


def spaghetti(ax, x, per_fit, colour=BLUE, label=None, median=True):
    """One thin line per CAV fit plus the median over fits."""
    per_fit = np.asarray(per_fit, float)
    for column in per_fit.T:
        ax.plot(x, column, color=colour, lw=0.5, alpha=0.25)
    if median:
        ax.plot(x, np.nanmedian(per_fit, axis=1), color=colour, lw=2, label=label)


def strip(ax, groups: dict, title="", ylabel="", zero=True):
    """Distribution per group: jittered points plus median line."""
    rng = np.random.default_rng(0)
    for k, (name, values) in enumerate(groups.items()):
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        colour = SERIES[k % len(SERIES)]
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(values)), values, s=9, alpha=0.5, color=colour, edgecolor="none")
        if len(values):
            ax.plot([k - 0.28, k + 0.28], [np.median(values)] * 2, color="#0b0b0b", lw=2)
    ax.set_xticks(range(len(groups)), [f"{name}\n(n={np.isfinite(np.asarray(v, float)).sum()})" for name, v in groups.items()])
    if zero:
        ax.axhline(0, color=GREY, lw=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def stacked_split(ax, labels, returning, composition, title=""):
    """Bars of the change split into returning-patient part and composition part (undefined stays empty)."""
    x = np.arange(len(labels))
    returning, composition = np.asarray(returning, float), np.asarray(composition, float)
    ax.bar(x, returning, color=BLUE, width=0.7, label="same patients (returning)")
    bottom = np.where(np.sign(composition) == np.sign(returning), returning, 0)
    ax.bar(x, composition, bottom=bottom, color=ORANGE, width=0.7, label="different patients (composition)")
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xticks(x, labels, rotation=90, fontsize=6)
    ax.set_title(title)
    ax.legend(loc="best")
