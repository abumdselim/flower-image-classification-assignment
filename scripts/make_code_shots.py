"""
make_code_shots.py — render every source file as a syntax-highlighted PNG
screenshot for the report's "Implementation" section.

Long files are split into page-sized chunks (~42 lines) with continuous line
numbers, and all chunks of a file share the same width for a clean look.
"""
import os
import sys

from PIL import Image
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import ImageFormatter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROJECT_DIR  # noqa: E402

OUT_DIR = os.path.join(PROJECT_DIR, "report", "code_shots")
CHUNK = 42          # lines per screenshot chunk
FONT_SIZE = 15

# (display order, source file)
FILES = [
    (1, "config.py"),
    (2, "data.py"),
    (3, "models.py"),
    (4, "extract_features.py"),
    (5, "train_cv.py"),
    (6, "evaluate.py"),
    (7, "main.py"),
    (8, "prepare_dataset.py"),
    (9, "scripts/collect_dataset.py"),
]

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def render_chunk(code: str, line_start: int) -> Image.Image:
    fmt = ImageFormatter(
        style="one-dark",
        font_name=FONT,
        font_size=FONT_SIZE,
        line_numbers=True,
        line_number_start=line_start,
        line_number_bg="#282c34",
        line_number_fg="#7f8790",
        line_pad=5,
    )
    png = highlight(code, PythonLexer(), fmt)
    import io
    return Image.open(io.BytesIO(png)).convert("RGB")


def pad_to_width(img: Image.Image, width: int) -> Image.Image:
    bg = img.getpixel((2, 2))
    if img.width >= width:
        return img
    canvas = Image.new("RGB", (width, img.height), bg)
    canvas.paste(img, (0, 0))
    return canvas


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in os.listdir(OUT_DIR):
        os.remove(os.path.join(OUT_DIR, f))

    for order, rel in FILES:
        src = os.path.join(PROJECT_DIR, rel)
        with open(src, "r", encoding="utf-8") as fh:
            lines = fh.read().rstrip("\n").split("\n")

        chunks = [lines[i:i + CHUNK] for i in range(0, len(lines), CHUNK)]
        stem = os.path.basename(rel)
        prefix = f"{order:02d}_{stem}"

        imgs, start = [], 1
        for ch in chunks:
            imgs.append(render_chunk("\n".join(ch), start))
            start += len(ch)

        width = max(im.width for im in imgs)
        total_h = 0
        for ci, im in enumerate(imgs):
            im = pad_to_width(im, width)
            imgs[ci] = im
            if len(imgs) == 1:
                out = f"{prefix}.png"
            else:
                out = f"{prefix}_p{ci + 1}.png"
            im.save(os.path.join(OUT_DIR, out))
            total_h += im.height
        tag = f" ({len(imgs)} chunks)" if len(imgs) > 1 else ""
        print(f"[shot] {rel:<28s} {len(lines):4d} lines{tag}")
    print(f"All code screenshots -> {OUT_DIR}")


if __name__ == "__main__":
    main()
