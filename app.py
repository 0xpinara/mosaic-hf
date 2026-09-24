"""
app.py - Gradio interface for the image mosaic generator (CS5130 Lab 1)

    gradio app.py      (reloads when the code changes)
    python app.py      then open http://localhost:7860
"""

import os
import time

import gradio as gr
import numpy as np
import spaces
from PIL import Image, ImageOps

from mosaic import SIZE, TILE_SETS, find_example, make_mosaic, mse, ssim

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = [
    [os.path.join(HERE, "examples", name), grid, tile_set, 16]
    for name, grid, tile_set in [
        ("astronaut.jpg", 32, "LEGO bricks"),
        ("astronaut.jpg", 48, "ASCII"),
        ("cat.jpg", 96, "Photo tiles"),
        ("coffee.jpg", 40, "Dots"),
        ("rocket.jpg", 48, "Photo tiles"),
        ("hubble.jpg", 64, "Flat colors"),
        ("halo.jpg", 48, "Dots"),
        ("stained_glass.jpg", 64, "LEGO bricks"),
    ]
]
START = "Upload an image or click one of the examples below."


def load_image(path):
    """Any image file -> uint8 RGB array: phone photos turned the right way up,
    16-bit images scaled to 0-255 and transparent parts put on white."""
    im = ImageOps.exif_transpose(Image.open(path))
    if im.mode.startswith("I"):
        a = np.asarray(im, dtype=np.float32)
        im = Image.fromarray(np.uint8(255 * a / max(a.max(), 1)))
    if im.mode in ("RGBA", "LA", "P"):
        im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im.convert("RGBA"))
    return np.asarray(im.convert("RGB"))


# the Space runs on ZeroGPU, which only starts apps that have a @spaces.GPU function;
# the mosaic itself is CPU-only (no effect when running locally)
@spaces.GPU(duration=10)
def run(path, grid, tile_set, k):
    if path is None:
        return None, None, None, START
    grid, k = int(grid), int(k)

    try:
        image = load_image(path)
        # if the input is one of the example photos, Photo tiles must not use its own patches
        example = find_example(image) or os.path.basename(path)
        t0 = time.perf_counter()
        original, segmented, mosaic = make_mosaic(image, grid, tile_set, k, exclude=example)
    except Exception as e:
        raise gr.Error(f"Could not make a mosaic from this image: {e}")
    t1 = time.perf_counter()
    err, sim = mse(original, mosaic), ssim(original, mosaic)
    t2 = time.perf_counter()

    cell = SIZE // grid
    rows, cols = original.shape[0] // cell, original.shape[1] // cell
    info = (
        "| Grid | Cell size | MSE | SSIM | Mosaic time | Metric time |\n"
        "|---|---|---|---|---|---|\n"
        f"| {cols} x {rows} | {cell} px | {err:.1f} | {sim:.3f} "
        f"| {(t1 - t0) * 1000:.0f} ms | {(t2 - t1) * 1000:.0f} ms |\n\n"
        "Both scores compare the mosaic with the resized original. "
        "Lower MSE is better; SSIM goes up to 1 (identical images)."
    )
    return original, segmented, mosaic, info


with gr.Blocks(title="Image Mosaic Generator") as demo:
    gr.Markdown(
        "# Image Mosaic Generator\n"
        "Upload a picture and it gets rebuilt out of small tiles. The image is split into a grid, "
        "every cell is put into one of *k* color categories (k-means), and each cell is replaced "
        "by the tile for its category."
    )
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(EXAMPLES[0][0], type="filepath", image_mode=None,
                             label="Input image", height=300)
            grid = gr.Slider(8, 128, value=32, step=8, label="Grid size",
                             info="Number of tiles along the longer side")
            tile_set = gr.Dropdown(TILE_SETS, value="LEGO bricks", label="Tile set",
                                   info="Photo tiles are small photos; the others use the k colors")
            k = gr.Slider(2, 64, value=16, step=1, label="Number of colors (k)",
                          info="Color categories found by k-means (not used by Photo tiles)")
            button = gr.Button("Make mosaic", variant="primary")
        with gr.Column(scale=2):
            with gr.Row():
                original = gr.Image(label="Original (resized and cropped)", height=300, format="png")
                segmented = gr.Image(label="Segmented (color category of each cell)", height=300,
                                     format="png")
            mosaic = gr.Image(label="Mosaic", height=520, format="png")
            info = gr.Markdown(START)

    inputs = [image, grid, tile_set, k]
    outputs = [original, segmented, mosaic, info]
    gr.Examples(EXAMPLES, inputs=inputs, outputs=outputs, fn=run,
                run_on_click=True, cache_examples=False)

    # update after anything the user changes (sliders when they are let go); a change
    # made while a mosaic is being computed runs again afterwards with the new settings
    gr.on([button.click, image.input, tile_set.select, grid.release, k.release],
          run, inputs, outputs, trigger_mode="always_last")
    demo.load(run, inputs, outputs)   # show the first example's mosaic right away


if __name__ == "__main__":
    demo.launch(allowed_paths=[os.path.join(HERE, "examples")])
