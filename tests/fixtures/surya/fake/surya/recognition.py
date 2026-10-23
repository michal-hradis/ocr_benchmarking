from surya import Result, native


class RecognitionPredictor:
    def __init__(self, foundation):
        assert foundation.checkpoint.endswith("text_recognition/2025_09_23")

    def __call__(self, images, task_names, polygons, recognition_batch_size, math_mode):
        assert task_names == ["ocr_with_boxes"] * len(images)
        assert recognition_batch_size is None or recognition_batch_size >= 1
        assert isinstance(math_mode, bool)
        results = [native(image) for image in images]
        for result, given in zip(results, polygons, strict=True):
            assert given == [b["polygon"] for b in result["detection"]["bboxes"]]
        return [Result(result["recognition"]) for result in results]
