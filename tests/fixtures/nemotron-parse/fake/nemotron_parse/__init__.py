"""Fake nemotron_parse: small stand-ins for the functions of the nemotron-parse 3.5.0.post7 package
that the adapter uses, written for the test (../README.md)."""

import re
from dataclasses import dataclass

BLOCK = re.compile(
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>(.*?)<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>"
    r"<class_([^>]+)>",
    re.S,
)


@dataclass(frozen=True)
class PostprocessOptions:
    infer_headers: bool = True
    render_charts: bool = True
    split_chart_value_percent: bool = True
    relocate_chart_captions: bool = True
    normalize_text: bool = True
    deduplicate: bool = True


def postprocess_document(raw, *, image_size=None, options=None):
    """The blocks of a raw output (an incomplete last block is left out) with their literal text
    as rendered content and boxes scaled to the image and clipped; Markdown joins the blocks."""
    width, height = image_size
    blocks = []
    for match in BLOCK.finditer(raw or ""):
        x1, y1, body, x2, y2, cls = match.groups()
        box = [float(x1), float(y1), float(x2), float(y2)]
        x0, x1 = sorted((box[0], box[2]))
        y0, y1 = sorted((box[1], box[3]))
        pixels = [min(width, max(0.0, x0 * width)), min(height, max(0.0, y0 * height))]
        pixels += [min(width, max(0.0, x1 * width)), min(height, max(0.0, y1 * height))]
        blocks.append(
            {"class": cls, "bbox": box, "text": body, "content": body, "bbox_pixels": pixels}
        )
    return {
        "raw_out": raw,
        "markdown": "\n\n".join(b["content"] for b in blocks if b["content"].strip()),
        "blocks": blocks,
        "warnings": [],
        "coordinate_frame": "content_normalized",
    }
