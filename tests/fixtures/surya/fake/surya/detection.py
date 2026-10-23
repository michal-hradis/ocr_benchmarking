from surya import Predictor, Result, native


class DetectionPredictor(Predictor):
    def __call__(self, images):
        return [Result(native(image)["detection"]) for image in images]
