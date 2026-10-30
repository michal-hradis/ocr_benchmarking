"""Fake docling.datamodel.base_models."""

from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO


class InputFormat(StrEnum):
    IMAGE = "image"


class ConversionStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"


@dataclass
class DocumentStream:
    name: str
    stream: BytesIO
