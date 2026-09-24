---
title: Image Mosaic Generator
emoji: 🧩
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# Image Mosaic Generator (CS5130 Lab 1)

A Gradio app that rebuilds a picture out of small tiles. You upload an image, pick a grid size, a tile set and the number of colors, and it shows the resized original, the segmented image and the mosaic, together with the MSE, the SSIM and how long it took.

**Live demo:** https://huggingface.co/spaces/piax0x/mosaic-hf  
**Report:** `report/Lab1_Report_Pinar_Aksoy.pdf`

## How it works

1. **Preprocessing:** the grid size N is the number of tiles along the long side. Each cell is 1024 // N pixels (32 px for N = 32), the long side is resized to exactly N cells and the short side is center-cropped to a whole number of cells.
2. **Grid:** the image is reshaped to `(rows, cell, cols, cell, 3)` and averaged over axes 1 and 3, which gives the mean color of every cell in one NumPy call.
3. **Classification:** k-means (OpenCV) on the cell colors gives k color categories, and every cell goes to the closest one (distances in CIELAB, all at once with broadcasting). The segmented image shows this step.
4. **Tiles:** one tile is made per category and the mosaic is `tiles[labels]` followed by a transpose and a reshape, so there is no loop over the cells.
5. **Metrics:** MSE and SSIM between the resized original and the mosaic.

## Tile sets

- **LEGO bricks:** a 1x1 brick seen from above (beveled edges, a stud with a shadow and a highlight) in the category color.
- **Flat colors:** plain squares. This is the baseline and gets the best scores.
- **Dots:** round beads on a dark background, like an LED board.
- **ASCII:** intensity thresholding picks one of 11 characters (brighter cell, more ink), drawn in the category color on a near-black background.
- **Photo tiles:** 592 patches of 64x64 px cut from the images in `examples/`. Each cell gets the patch with the closest mean color, shifted 40% toward the cell's color. When the input is one of the example photos (recognized by its file name or a small thumbnail, so renamed copies count too), its own patches are left out, so it is not rebuilt from pieces of itself.

## Files

| File | What it does |
|---|---|
| `app.py` | Gradio interface |
| `mosaic.py` | all the image processing, plus a nested-loop version for the speed test |
| `benchmark.py` | timing (vectorized vs. loops), MSE/SSIM for every tile set, and the two report figures |
| `examples/` | the 7 test images (also the source of the photo tiles) |
| `results/` | output of `benchmark.py` |
| `report/` | the report (LaTeX source and PDF) |

## Running it

```bash
pip install -r requirements.txt
python app.py           # or: gradio app.py (reloads on changes)
```

Then open http://localhost:7860.

`python benchmark.py` recreates everything in `results/` (a few minutes). The report compiles with `pdflatex Lab1_Report_Pinar_Aksoy.tex` (or `tectonic`) inside `report/`.

## Image credits

Five images come from the scikit-image sample data: astronaut (NASA, public domain), cat "Chelsea" (Stefan van der Walt, CC0), coffee (Rachel Michetti, CC0), rocket (SpaceX, public domain) and the Hubble eXtreme Deep Field (NASA, public domain). halo.jpg: [source]. stained_glass.jpg: [source].
