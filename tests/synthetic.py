"""Synthetic page images with PAGE XML and manifests for engine smoke benchmarks."""

import json
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


def write_manifest(root, samples):
    manifest = root / "manifest.json"
    data = {"format": "ocrbench.dataset-manifest", "version": 1, "name": "t", "samples": samples}
    manifest.write_text(json.dumps(data), encoding="utf-8")
    return manifest


PAGE = [
    ["The Daily Chronicle"],
    ["The quick brown fox jumps over", "the lazy dog near the river."],
    ["A second paragraph and more text", "with numbers 1923 and 2026."],
]
PARAGRAPH = [["Typewriters were common in", "offices for many decades."]]


def render(path, paragraphs, size):
    """Synthetic image of black text; returns the PAGE regions as [[(text, box), ...], ...]."""
    image = Image.new("L", size, 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=28)
    regions, y = [], 30
    for lines in paragraphs:
        region = []
        for text in lines:
            draw.text((40, y), text, font=font, fill=0)
            region.append((text, draw.textbbox((40, y), text, font=font)))
            y += 40
        regions.append(region)
        y += 40
    image.save(path)
    return regions


def page_xml(path, regions, size):
    def points(box):
        x0, y0, x1, y1 = box
        return f"{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"

    body = []
    for r, lines in enumerate(regions):
        box = (
            min(b[0] for _, b in lines),
            min(b[1] for _, b in lines),
            max(b[2] for _, b in lines),
            max(b[3] for _, b in lines),
        )
        body.append(f'<TextRegion id="r{r}"><Coords points="{points(box)}"/>')
        for i, (text, line_box) in enumerate(lines):
            body.append(
                f'<TextLine id="r{r}l{i}"><Coords points="{points(line_box)}"/>'
                f"<TextEquiv><Unicode>{escape(text)}</Unicode></TextEquiv></TextLine>"
            )
        body.append("</TextRegion>")
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        f'<Page imageFilename="{path.stem}.png" imageWidth="{size[0]}" imageHeight="{size[1]}">'
        + "".join(body)
        + "</Page></PcGts>",
        encoding="utf-8",
    )


def smoke_dataset(root):
    root.mkdir()
    samples = []
    for sid, unit, paragraphs, size in (
        ("page-1", "page", PAGE, (900, 420)),
        ("para-1", "paragraph", PARAGRAPH, (520, 120)),
        ("blank-1", "page", [], (300, 200)),
    ):
        regions = render(root / f"{sid}.png", paragraphs, size)
        page_xml(root / f"{sid}.xml", regions, size)
        samples.append({"id": sid, "image": f"{sid}.png", "gt": f"{sid}.xml", "unit": unit})
    return write_manifest(root, samples)
