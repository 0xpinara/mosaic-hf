"""
benchmark.py - makes the tables and figures used in the report

    python benchmark.py

1. Timing: the vectorized grid code vs. the nested-loop version for 8x8 to
   128x128 grids (same palette and tiles for both, and it checks that the two
   mosaics are identical), plus the whole make_mosaic() call and the metrics.
2. Similarity: MSE and SSIM for every tile set and grid size.
3. Figures: all tile sets on the astronaut, and a mosaic of every other example.

For 1 and 2 the test images are center-cropped to squares, so grid N gives
N x N cells, and Photo tiles never use patches of the image being rebuilt.
Everything is saved in results/.
"""

import csv
import glob
import os
import platform
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mosaic as M

GRIDS = [8, 16, 32, 64, 128]
K = 16
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(name):
    return cv2.cvtColor(cv2.imread(os.path.join(M.EXAMPLES, name)), cv2.COLOR_BGR2RGB)


def square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    return img[top:top + s, left:left + s]


def median_time(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def timing(images):
    rows = []
    for n in GRIDS:
        vec, loops, full, metrics, same = [], [], [], [], True
        for img in images.values():
            im, cell = M.preprocess(img, n)
            pal = M.palette(M.cell_colors(im, cell), K)
            tiles = M.make_tiles("LEGO bricks", pal, cell)
            a = M.mosaic_vectorized(im, cell, pal, tiles)
            b = M.mosaic_loops(im, cell, pal, tiles)
            same = same and np.array_equal(a, b)

            vec.append(median_time(lambda: M.mosaic_vectorized(im, cell, pal, tiles), 7))
            loops.append(median_time(lambda: M.mosaic_loops(im, cell, pal, tiles), 3))
            full.append(median_time(lambda: M.make_mosaic(img, n, "LEGO bricks", K), 3))
            original, _, mosaic = M.make_mosaic(img, n, "LEGO bricks", K)
            metrics.append(median_time(lambda: (M.mse(original, mosaic), M.ssim(original, mosaic)), 3))

        v, l = 1000 * float(np.mean(vec)), 1000 * float(np.mean(loops))
        rows.append({
            "grid": f"{n}x{n}", "cells": n * n, "cell_px": M.SIZE // n,
            "vectorized_ms": round(v, 1), "loops_ms": round(l, 1), "speedup": round(l / v, 1),
            "full_pipeline_ms": round(1000 * float(np.mean(full)), 1),
            "metrics_ms": round(1000 * float(np.mean(metrics)), 1),
            "identical_output": bool(same),
        })
        print(rows[-1])
    return rows


def similarity(images):
    rows = []
    for tile_set in M.TILE_SETS:
        for n in [16, 32, 64, 128]:
            scores = []
            for name, img in images.items():
                original, _, mosaic = M.make_mosaic(img, n, tile_set, K, exclude=name)
                scores.append((M.mse(original, mosaic), M.ssim(original, mosaic)))
            mse, ssim = np.mean(scores, axis=0)
            rows.append({"tile_set": tile_set, "grid": f"{n}x{n}",
                         "mse": round(float(mse), 1), "ssim": round(float(ssim), 3)})
            print(rows[-1])
    return rows


def save_csv(rows, name):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot_timing(rows):
    cells = [r["cells"] for r in rows]
    plt.figure(figsize=(5, 3.2))
    plt.loglog(cells, [r["loops_ms"] for r in rows], "o-", label="nested loops")
    plt.loglog(cells, [r["vectorized_ms"] for r in rows], "s-", label="vectorized (NumPy)")
    plt.xticks(cells, [r["grid"] for r in rows])
    ticks = [5, 10, 20, 50, 100, 200, 500, 1000]
    lo, hi = min(r["vectorized_ms"] for r in rows), max(r["loops_ms"] for r in rows)
    ticks = [t for t in ticks if lo / 2 <= t <= hi * 2]
    plt.yticks(ticks, [str(t) for t in ticks])
    plt.minorticks_off()
    plt.xlabel("grid size")
    plt.ylabel("time (ms), log scale")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "timing.png"), dpi=200)
    plt.close()


def save_panels(panels, nrows, ncols, figsize, name, width_ratios=None, fontsize=9):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             gridspec_kw={"width_ratios": width_ratios} if width_ratios else None)
    for ax, (im, title) in zip(np.ravel(axes), panels):
        ax.imshow(im)
        ax.set_title(title, fontsize=fontsize)
        ax.axis("off")
    plt.tight_layout(pad=0.4, w_pad=0.3, h_pad=1.0)
    plt.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def figure_tile_sets():
    """Figure 1: original, segmented image and all five tile sets (32x32, k = 16)."""
    img = load("astronaut.jpg")
    runs = {t: M.make_mosaic(img, 32, t, K, exclude="astronaut.jpg") for t in M.TILE_SETS}
    original, segmented, _ = runs["LEGO bricks"]
    photo = runs["Photo tiles"][2]
    cell = M.SIZE // 32
    closeup = photo[3 * cell:11 * cell, 12 * cell:20 * cell]   # 8x8 tiles around the face
    save_panels([
        (original, "(a) original, resized"),
        (segmented, f"(b) segmented, k = {K}"),
        (runs["Flat colors"][2], "(c) Flat colors"),
        (runs["LEGO bricks"][2], "(d) LEGO bricks"),
        (runs["Dots"][2], "(e) Dots"),
        (runs["ASCII"][2], "(f) ASCII"),
        (photo, "(g) Photo tiles"),
        (closeup, "(h) Photo tiles, close-up"),
    ], 2, 4, (8, 4.6), "figure_tile_sets.png", fontsize=11)


def figure_examples():
    """Figure 2: the other six test images with the settings of the app's examples."""
    settings = [("cat", 96, "Photo tiles"), ("coffee", 40, "Dots"), ("rocket", 48, "Photo tiles"),
                ("hubble", 64, "Flat colors"), ("halo", 48, "Dots"), ("stained_glass", 64, "LEGO bricks")]
    panels = []
    for name, n, tile_set in settings:
        mosaic = M.make_mosaic(load(f"{name}.jpg"), n, tile_set, K, exclude=f"{name}.jpg")[2]
        title = name.replace("_", " ").replace("hubble", "Hubble")
        panels.append((mosaic, f"{title}\n{tile_set}, {n}"))
    ratios = [m.shape[1] / m.shape[0] for m, _ in panels]
    save_panels(panels, 1, 6, (8, 8 / sum(ratios) + 0.5), "figure_examples.png", ratios)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print(f"Python {platform.python_version()}, NumPy {np.__version__}, OpenCV {cv2.__version__}, "
          f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs")
    names = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(M.EXAMPLES, "*.jpg")))]
    images = {name: square(load(name)) for name in names}
    print(f"{len(images)} test images, k = {K}\n")
    M.make_mosaic(images["astronaut.jpg"], 32, "LEGO bricks", K)   # warm-up

    t = timing(images)
    save_csv(t, "timing.csv")
    plot_timing(t)
    print()
    save_csv(similarity(images), "similarity.csv")
    figure_tile_sets()
    figure_examples()
    print("\nsaved timing.csv, timing.png, similarity.csv, figure_tile_sets.png, "
          "figure_examples.png in results/")
