"""Fake inference configurations that keep their keyword arguments."""


class _Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class SegmentationInferenceConfig(_Config):
    pass


class RecognitionInferenceConfig(_Config):
    pass
