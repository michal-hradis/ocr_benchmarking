def _crop_block(image, polygon, pad=4):
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    x0, y0 = max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad)
    x1, y1 = min(image.size[0], int(max(xs)) + pad), min(image.size[1], int(max(ys)) + pad)
    return image.crop((0, 0, 1, 1)) if x1 <= x0 or y1 <= y0 else image.crop((x0, y0, x1, y1))


def _detect_repeat_loop(text, base_max_repeats=4, window_size=500, scaling_factor=3.0):
    for length in range(1, window_size // 2 + 1):
        candidate, repeats, position = text[-length:], 0, len(text) - length
        while position >= 0 and text[position : position + length] == candidate:
            repeats, position = repeats + 1, position - length
        if text and repeats > int(base_max_repeats * (1 + scaling_factor / length)):
            return True
    return False
