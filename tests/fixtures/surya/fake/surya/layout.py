from surya import Result, native


class LayoutPredictor:
    def __init__(self, foundation):
        assert foundation.checkpoint.endswith("layout/2025_09_23")

    def __call__(self, images):
        return [Result(native(image)["layout"]) for image in images]
