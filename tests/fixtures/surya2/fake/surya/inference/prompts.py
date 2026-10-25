SKIP_OCR_LABELS = {"Figure", "Image", "Diagram", "Blank-Page"}
HIGH_ACCURACY_BBOX_PROMPT = (
    "OCR this image to HTML. Each block is a div with data-label and data-bbox "
    "(x0 y0 x1 y1, normalized 0-1000)."
)
