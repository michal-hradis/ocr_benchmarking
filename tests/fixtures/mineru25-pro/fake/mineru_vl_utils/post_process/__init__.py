"""Fake post_process: json2md as in mineru-vl-utils 2.0.7, which joins the content of every block
with blank lines and appends a text block with merge_prev to the previous text block."""

import re


def json2md(blocks):
    parts, last_text = [], -1
    for block in blocks:
        content = block["content"]
        if not content:
            continue
        if block.get("merge_prev", False) and last_text >= 0:
            separator = "" if re.search(r"[一-鿿㐀-䶿]", content) else " "
            parts[last_text] += separator + content
        else:
            parts.append(content)
            if block.get("type", "text") == "text":
                last_text = len(parts) - 1
    return "\n\n".join(parts)
