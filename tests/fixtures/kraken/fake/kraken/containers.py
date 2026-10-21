"""The fields of Kraken's result containers, as dataclasses like the real ones."""

from dataclasses import dataclass


@dataclass
class BaselineLine:
    id: str
    text: str | None = None
    base_dir: str | None = None
    imagename: str | None = None
    tags: dict | None = None
    split: str | None = None
    regions: list | None = None
    language: list | None = None
    type: str = "baselines"
    baseline: list | None = None
    boundary: list | None = None


@dataclass
class Region:
    id: str
    boundary: list
    imagename: str | None = None
    tags: dict | None = None
    language: list | None = None


@dataclass
class Segmentation:
    type: str
    imagename: str | None
    text_direction: str
    script_detection: bool
    lines: list | None = None
    regions: dict | None = None
    line_orders: list | None = None
    language: list | None = None


class BaselineOCRRecord(BaselineLine):
    """A recognized line: the line's dataclass fields plus prediction and confidences."""

    def __init__(self, prediction, confidences, line):
        super().__init__(**line)
        self.prediction = prediction
        self.confidences = confidences
