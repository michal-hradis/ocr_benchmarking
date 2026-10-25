from PIL import ImageStat


def is_blank_region(image):
    """Mostly near-white (every channel >= 245 in 99 % of pixels) or nearly one color."""
    if image.width * image.height == 0:
        return False
    gray = image.convert("L")
    white = ImageStat.Stat(gray.point(lambda v: 255 if v >= 245 else 0)).mean[0] / 255
    return white >= 0.99 or ImageStat.Stat(gray).stddev[0] < 8
