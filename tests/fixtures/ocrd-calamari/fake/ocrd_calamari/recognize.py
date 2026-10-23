from pathlib import Path


class CalamariRecognize:
    @staticmethod
    def step(parameter):
        assert parameter["device"] == -1
        assert list(Path(parameter["checkpoint_dir"]).glob("*.ckpt.json"))
        return "recognize"
