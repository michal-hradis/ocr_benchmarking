"""Fake nemotron_parse.inference: the prompt, GenerationSettings, GenerationError and a
DocumentParser that calls the injected backend for the full page and, in the double mode, once per
Table block, whose body it replaces with the crop's (../README.md)."""

from dataclasses import dataclass

from . import BLOCK, PostprocessOptions, postprocess_document

PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 0.2
    repetition_penalty: float = 1.05
    max_tokens: int = 8744
    top_k: int = 50
    top_p: float = 1.0
    seed: int = 0


class GenerationError(RuntimeError):
    def __init__(self, stage, response):
        self.stage, self.response = stage, response
        super().__init__(f"{stage} generation failed or is incomplete")


class DocumentParser:
    def __init__(self, *, backend, fullpage, crop, merge_policy, options=None):
        assert merge_policy == "guarded"
        self.backend = backend
        self.settings = {"fullpage": fullpage, "crop": crop}
        self.options = options or PostprocessOptions()

    def _generate(self, image, stage):
        response = dict(
            self.backend(image, prompt=PROMPT, stage=stage, settings=self.settings[stage])
        )
        if (
            response.get("finish_reason") not in (None, "stop", "eos")
            or not response["raw_out"].strip()
        ):
            raise GenerationError(stage, response)
        return response

    def parse(self, image, *, mode="single"):
        raw = self._generate(image, "fullpage")["raw_out"]
        if mode == "double":

            def refine(match):
                if match[6] != "Table":
                    return match[0]
                w, h = image.size
                box = (
                    float(match[1]) * w,
                    float(match[2]) * h,
                    float(match[4]) * w,
                    float(match[5]) * h,
                )
                crop = self._generate(image.crop(tuple(map(int, box))), "crop")["raw_out"]
                body = BLOCK.search(crop)[3]
                return match[0].replace(match[3], body)

            raw = BLOCK.sub(refine, raw)
        document = postprocess_document(raw, image_size=image.size, options=self.options)
        document["stages"] = {}
        return document
