"""Fake PAGE XML serializer: records the lines and image size it was given."""


def serialize(results, image_size, template):
    assert template == "pagexml"
    texts = "|".join(line.prediction for line in results.lines)
    return (
        f'<PcGts size="{image_size[0]}x{image_size[1]}" name="{results.imagename}">{texts}</PcGts>'
    )
