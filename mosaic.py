"""
mosaic.py - image mosaic generator (CS5130 Lab 1)

  1. preprocess()   resize and center-crop so the grid fits the image exactly
  2. cell_colors()  average color of every cell (one reshape + mean, no loops)
  3. palette()      k-means color quantization -> k color categories
     classify()     closest category for every cell, measured in CIELAB
  4. make_tiles()   one tile per category, then the mosaic is tiles[labels]
  5. mse(), ssim()  compare the mosaic with the original

mosaic_loops() does steps 2-4 with nested loops. It is only used in
benchmark.py to compare its speed with the vectorized version.
"""

import glob
import os
from functools import lru_cache

import cv2
import numpy as np
from skimage.metrics import structural_similarity

SIZE = 1024   # long side of the working image (a bit less when the grid doesn't divide 1024)
TILE_SETS = ["LEGO bricks", "Flat colors", "Dots", "ASCII", "Photo tiles"]
EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")


# step 1

def preprocess(img, grid):
    """grid is the number of cells along the long side. Each cell is SIZE // grid
    pixels, the long side is resized to exactly grid cells and the short side is
    center-cropped to a whole number of cells."""
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[:, :, :3]

    cell = SIZE // grid
    h, w = img.shape[:2]
    scale = grid * cell / max(h, w)
    size = (max(cell, round(w * scale)), max(cell, round(h * scale)))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    h, w = img.shape[:2]
    top, left = (h % cell) // 2, (w % cell) // 2
    return img[top:top + h - h % cell, left:left + w - w % cell], cell


# step 2

def cell_colors(img, cell):
    """Mean RGB color of every cell -> (rows, cols, 3).
    After the reshape the pixels of one cell are on axes 1 and 3."""
    rows, cols = img.shape[0] // cell, img.shape[1] // cell
    return img.reshape(rows, cell, cols, cell, 3).mean(axis=(1, 3))


# step 3

def to_lab(rgb):
    """RGB (0-255, any shape ending in 3) -> CIELAB, where distances are closer
    to how different two colors look."""
    rgb = np.asarray(rgb, dtype=np.float32)
    return cv2.cvtColor(rgb.reshape(-1, 1, 3) / 255, cv2.COLOR_RGB2LAB).reshape(rgb.shape)


def palette(colors, k, seed=0):
    """k-means on the cell colors (in CIELAB). Returns the k category colors in RGB."""
    data = colors.reshape(-1, 3).astype(np.float32)
    if len(data) > 4096:
        # for big grids k-means is fitted on a random sample of the cells to stay fast
        data = data[np.random.default_rng(seed).choice(len(data), 4096, replace=False)]
    k = min(k, len(np.unique(data, axis=0)))   # not more clusters than different colors

    cv2.setRNGSeed(seed)   # same palette every run
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, _, centers = cv2.kmeans(to_lab(data), k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    rgb = cv2.cvtColor(centers.reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)
    return np.clip(rgb * 255, 0, 255)


def classify(colors, pal):
    """Index of the closest palette color for every cell -> (rows, cols).
    (cells, 1, 3) minus (1, k, 3) gives every distance at once."""
    a = to_lab(colors.reshape(-1, 3))
    b = to_lab(pal)
    dist = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    return dist.argmin(axis=1).reshape(colors.shape[:2])


# step 4

S = 128   # tile patterns are drawn at 128x128 and shrunk to the cell size


def shrink(pattern, cell):
    return cv2.resize(pattern, (cell, cell), interpolation=cv2.INTER_AREA)


@lru_cache
def lego_pattern(cell):
    """A 1x1 brick seen from above: bright top/left edges, dark bottom/right
    edges and a round stud with a shadow and a shiny spot.
    The tile is color * shade + light."""
    shade = np.ones((S, S), np.float32)
    shade[:7] = shade[:, :7] = 1.15
    shade[-7:] = shade[:, -7:] = 0.7
    cv2.circle(shade, (70, 71), 36, 0.62, -1)   # shadow under the stud
    cv2.circle(shade, (64, 64), 36, 1.06, -1)   # the stud
    light = np.zeros((S, S), np.float32)
    light[:7] = light[:, :7] = 12
    spot = np.zeros((S, S), np.float32)
    cv2.circle(spot, (51, 51), 13, 55, -1)       # shiny spot on the stud
    light += cv2.GaussianBlur(spot, (0, 0), 5)
    return shrink(shade, cell), shrink(light, cell)


@lru_cache
def dot_pattern(cell):
    """A round bead on a dark background (like an LED board) with a highlight."""
    mask = np.zeros((S, S), np.float32)
    cv2.circle(mask, (64, 64), 56, 1.0, -1)
    spot = np.zeros((S, S), np.float32)
    cv2.circle(spot, (49, 49), 14, 70, -1)
    spot = cv2.GaussianBlur(spot, (0, 0), 7) * mask
    return shrink(mask, cell), shrink(spot, cell)


CHARS = " .:-=+*o#%@"


@lru_cache
def ascii_masks(cell):
    """One mask per character, sorted from least to most ink."""
    masks = []
    for ch in CHARS:
        m = np.zeros((S, S), np.float32)
        (w, h), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX, 3.8, 10)
        cv2.putText(m, ch, ((S - w) // 2, (S + h) // 2), cv2.FONT_HERSHEY_SIMPLEX, 3.8, 1.0, 10)
        masks.append(shrink(m, cell))
    masks = np.array(masks)
    return masks[np.argsort(masks.mean(axis=(1, 2)))]


def make_tiles(tile_set, pal, cell):
    """One tile per palette color -> uint8 (k, cell, cell, 3). The colors are
    broadcast against the pattern, so all k tiles are made at once."""
    col = pal[:, None, None, :].astype(np.float32)   # (k, 1, 1, 3)

    if tile_set == "Flat colors":
        tiles = np.broadcast_to(col, (len(pal), cell, cell, 3))
    elif tile_set == "LEGO bricks":
        shade, light = lego_pattern(cell)
        tiles = col * shade[:, :, None] + light[:, :, None]
    elif tile_set == "Dots":
        mask, spot = dot_pattern(cell)
        m = mask[:, :, None]
        tiles = col * m + 25 * (1 - m) + spot[:, :, None]
    elif tile_set == "ASCII":
        # intensity thresholding: the brightness range is cut into 11 levels,
        # and a brighter color gets a character with more ink
        brightness = pal @ np.array([0.299, 0.587, 0.114])
        level = (brightness * len(CHARS) / 256).astype(int)
        m = ascii_masks(cell)[level][..., None]
        tiles = np.minimum(col * 1.3, 255) * m + 0.12 * col * (1 - m)
    else:
        raise ValueError(f"unknown tile set: {tile_set}")
    return np.clip(tiles, 0, 255).round().astype(np.uint8)


@lru_cache
def photo_tiles(cell):
    """The Photo tiles set: every photo in examples/ is resized so its short side
    is 512 px and cut into 64x64 patches (592 tiles from the 7 photos). Returns the
    tiles resized to the cell size, the mean color of each tile and the name of
    the photo each tile was cut from."""
    patches, names = [], []
    for path in sorted(glob.glob(os.path.join(EXAMPLES, "*.jpg"))):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        s = 512 / min(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        r, c = img.shape[0] // 64, img.shape[1] // 64
        img = img[:r * 64, :c * 64]
        patches.append(img.reshape(r, 64, c, 64, 3).swapaxes(1, 2).reshape(-1, 64, 64, 3))
        names += [os.path.basename(path)] * (r * c)
    interp = cv2.INTER_AREA if cell < 64 else cv2.INTER_CUBIC
    tiles = np.array([cv2.resize(p, (cell, cell), interpolation=interp)
                      for p in np.concatenate(patches)])
    return tiles, tiles.mean(axis=(1, 2)), np.array(names)


def thumbnail(img):
    return cv2.resize(np.ascontiguousarray(img), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)


@lru_cache
def example_thumbnails():
    """32x32 thumbnails of the example photos."""
    return {os.path.basename(p): thumbnail(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
            for p in sorted(glob.glob(os.path.join(EXAMPLES, "*.jpg")))}


def find_example(img):
    """Name of the example photo that img is a copy of (renamed, re-saved or
    resized), or None. Different photos differ by 30+ on average, copies by < 15."""
    t = thumbnail(img)
    diff = {name: np.abs(t - th).mean() for name, th in example_thumbnails().items()}
    name = min(diff, key=diff.get)
    return name if diff[name] < 15 else None


def assemble(tiles, labels):
    """tiles[labels] is (rows, cols, cell, cell, 3). Swapping axes 1 and 2 and
    reshaping turns it into one (rows*cell, cols*cell, 3) image."""
    rows, cols = labels.shape
    cell = tiles.shape[1]
    return tiles[labels].transpose(0, 2, 1, 3, 4).reshape(rows * cell, cols * cell, 3)


# step 5

def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b) ** 2))


def ssim(a, b):
    # mean SSIM of the three color channels, 7x7 window
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


# everything together

def make_mosaic(image, grid=32, tile_set="LEGO bricks", k=16, exclude=None):
    """Returns the resized original, the segmented image and the mosaic.
    exclude is the file name of the input if it is one of the example photos,
    so that Photo tiles never rebuild a photo out of pieces of itself."""
    img, cell = preprocess(image, grid)
    colors = cell_colors(img, cell)

    if tile_set == "Photo tiles":
        tiles, pal, source = photo_tiles(cell)
        if exclude:
            tiles, pal = tiles[source != exclude], pal[source != exclude]
        labels = classify(colors, pal)
        mosaic = assemble(tiles, labels).astype(np.float32)
        # move each patch 40% of the way toward the color of the cell it replaces
        shift = 0.4 * (colors - pal[labels])
        mosaic += shift.repeat(cell, axis=0).repeat(cell, axis=1)
        mosaic = np.clip(mosaic, 0, 255).round().astype(np.uint8)
    else:
        pal = palette(colors, k)
        labels = classify(colors, pal)
        mosaic = assemble(make_tiles(tile_set, pal, cell), labels)

    # segmented image: every cell in the color of its category, with grid lines
    seg = np.clip(pal[labels], 0, 255).round().astype(np.uint8)
    seg = seg.repeat(cell, axis=0).repeat(cell, axis=1)
    seg[::cell] = seg[:, ::cell] = 40
    seg[-1] = seg[:, -1] = 40
    return img, seg, mosaic


def mosaic_vectorized(img, cell, pal, tiles):
    return assemble(tiles, classify(cell_colors(img, cell), pal))


def mosaic_loops(img, cell, pal, tiles):
    """Steps 2-4 again, but one cell at a time."""
    pal_lab = to_lab(pal)
    out = np.empty_like(img)
    for y in range(0, img.shape[0], cell):
        for x in range(0, img.shape[1], cell):
            color = to_lab(img[y:y + cell, x:x + cell].mean(axis=(0, 1)))
            best = np.argmin(((pal_lab - color) ** 2).sum(axis=1))
            out[y:y + cell, x:x + cell] = tiles[best]
    return out
