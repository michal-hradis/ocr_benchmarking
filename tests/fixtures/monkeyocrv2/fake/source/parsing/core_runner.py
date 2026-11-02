"""Stand-ins for the functions of upstream's parsing/core_runner.py that the MonkeyOCRv2 adapter
uses, written for the offline test (../../README.md), not copied from upstream. The prompts are
upstream's strings. The tolerant parser, the formula and table formatting and the repetition check
are simplified; the GPU smoke test runs the real ones."""

import ast
import base64
import io
import re

ALL_PROMPT = {
    "Caption": "Please output the text content from the image.",
    "List-item": "Please output the text content from the image.",
    "Page-footer": "Please output the text content from the image.",
    "Page-header": "Please output the text content from the image.",
    "Section-header": "Please output the text content from the image.",
    "Text": "Please output the text content from the image.",
    "Title": "Please output the text content from the image.",
    "Formula": "Please write out the expression of the formula in the image using LaTeX format.",
    "Table": "Please extract the table from the image and represent it in OTSL format.",
    "LAYOUT": (
        "Please output the categories and coordinates of the document elements in reading order."
    ),
}


def image_to_png_data_uri(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def load_image(image, max_pixels=None, min_pixels=None):
    """Proportional resizing to at least min_pixels and at most max_pixels."""
    image = image.convert("RGB")
    area = image.width * image.height
    for limit, grow in ((min_pixels, True), (max_pixels, False)):
        if limit and (area < limit if grow else area > limit):
            scale = (limit / area) ** 0.5
            image = image.resize((int(image.width * scale), int(image.height * scale)))
            area = image.width * image.height
    return image


def _normalize_model_item(item, include_content):
    if not isinstance(item, dict) or "bbox" not in item or "label" not in item:
        return None
    bbox = item["bbox"]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        return {"bbox": [float(v) for v in bbox], "label": str(item["label"])}
    except (TypeError, ValueError):
        return None


def _parse_tolerant_items(text, normalize_item):
    """The whole output as a list literal, else every complete {...} item in it."""
    text = (text or "").strip()
    try:
        value = ast.literal_eval(text)
        if isinstance(value, list):
            return [item for raw in value if (item := normalize_item(raw)) is not None]
    except (SyntaxError, ValueError):
        pass
    items = []
    for block in re.findall(r"\{[^{}]*\}", text):
        try:
            item = normalize_item(ast.literal_eval(block))
        except (SyntaxError, ValueError):
            continue
        if item is not None:
            items.append(item)
    return items


def _map_bbox_to_image(bbox, width, height):
    x1, x2 = sorted(v / 1000 * width for v in (bbox[0], bbox[2]))
    y1, y2 = sorted(v / 1000 * height for v in (bbox[1], bbox[3]))
    x1 = max(0, min(round(x1), width - 1))
    y1 = max(0, min(round(y1), height - 1))
    return [x1, y1, max(x1 + 1, min(round(x2), width)), max(y1 + 1, min(round(y2), height))]


def _build_page_tasks(page_idx, image, layouts):
    return [
        {
            "image": image.crop(tuple(item["bbox"])),
            "bbox": item["bbox"],
            "label": item["label"],
            "question": ALL_PROMPT.get(item["label"], ""),
            "need_infer": item["label"] in ALL_PROMPT,
            "block_idx": index,
        }
        for index, item in enumerate(layouts)
    ]


def _format_block_fields(task, raw, doc_name, picture_count, use_base64, image_dir):
    content, fields = (raw or "").strip(), {}
    if task["label"] == "Formula":
        content = "$$\n" + content.strip("$").strip() + "\n$$\n"
    elif task["label"] == "Table":
        fields["otsl"] = content
        rows = [r for r in content.split("<nl>") if r]
        cells = ["".join(f"<td>{c}</td>" for c in re.split(r"<fcel>|<ecel>", r)[1:]) for r in rows]
        content = "<table>" + "".join(f"<tr>{c}</tr>" for c in cells) + "</table>"
    elif task["label"] == "Title":
        content = "# " + content
    elif task["label"] == "Section-header":
        content = "## " + content
    fields["content"] = content
    return fields


def _should_retry_repeat_output(raw):
    """Whether the output ends with a unit of up to 20 characters repeated 10 times."""
    return any(raw and raw.endswith(raw[-n:] * 10) for n in range(1, 21) if len(raw) >= 10 * n)


def result2md(names, results, save_dir=None, keep_header_footer=False):
    blocks = [
        (item.get("content") or "").strip()
        for item in results[0]
        if keep_header_footer or item.get("label") not in {"Page-header", "Page-footer"}
    ]
    blocks = [block for block in blocks if block]
    return ["\n\n".join(blocks) + ("\n" if blocks else "")]
