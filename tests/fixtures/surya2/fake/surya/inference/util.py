from PIL import Image


def scale_to_fit(img, max_size=(3072, 2048), min_size=(1792, 28), grid_size=28):
    """Scaled into the pixel range and rounded to whole 28-pixel units, as Surya's; units are
    removed while the rounded size is over the limit."""
    width, height = img.size
    pixels, limit, scale = width * height, max_size[0] * max_size[1], 1.0
    if pixels > limit:
        scale = (limit / pixels) ** 0.5
    elif pixels < min_size[0] * min_size[1]:
        scale = (min_size[0] * min_size[1] / pixels) ** 0.5
    columns = max(1, round(width * scale / grid_size))
    rows = max(1, round(height * scale / grid_size))
    while columns * rows * grid_size * grid_size > limit and columns * rows > 1:
        if columns / rows > width / height:
            columns -= 1
        else:
            rows -= 1
    size = (columns * grid_size, rows * grid_size)
    return img if size == img.size else img.resize(size, Image.Resampling.LANCZOS)
