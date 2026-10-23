class PreProcessor:
    """Records the arguments the adapter builds docTR's detection pre-processor with."""

    def __init__(self, output_size, batch_size, mean, std, preserve_aspect_ratio, symmetric_pad):
        assert (batch_size, preserve_aspect_ratio, symmetric_pad) == (2, True, True)
        assert (mean, std) == ((0.8, 0.8, 0.8), (0.3, 0.3, 0.3))
        self.batch_size = batch_size
        self.resize = type("Resize", (), {})()
        self.resize.size = output_size
        self.resize.preserve_aspect_ratio = preserve_aspect_ratio
        self.resize.symmetric_pad = symmetric_pad
