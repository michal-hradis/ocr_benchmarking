"""Fake OCR-D core: a workspace that lists its files and a run_processor that writes PAGE."""

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

FIXTURES = Path(__file__).resolve().parents[2]
OUTPUTS = {900: "page.xml", 1500: "newspaper.xml"}  # image width: committed pipeline output


class Workspace:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.files = []
        self.saved = False

    def add_file(self, file_grp, *, file_id, mimetype, page_id, local_filename):
        assert (self.directory / local_filename).is_file()
        entry = SimpleNamespace(
            fileGrp=file_grp,
            ID=file_id,
            mimetype=mimetype,
            pageId=page_id,
            local_filename=local_filename,
        )
        self.files.append(entry)
        return entry

    def save_mets(self):
        (self.directory / "mets.xml").write_text("<mets/>", encoding="utf-8")
        self.saved = True

    def find_files(self, *, file_grp, page_id):
        return (f for f in self.files if f.fileGrp == file_grp and f.pageId == page_id)


class Resolver:
    def workspace_from_nothing(self, directory):
        assert not any(Path(directory).iterdir()), "the workspace directory must be new"
        return Workspace(directory)


def run_processor(
    processor, *, workspace, input_file_grp, output_file_grp, parameter, instance_caching
):
    assert instance_caching
    assert workspace.saved
    assert len(list(workspace.find_files(file_grp=input_file_grp, page_id="PHYS_0001"))) == 1
    image = next(f for f in workspace.files if f.mimetype.startswith("image/"))
    with Image.open(workspace.directory / image.local_filename) as opened:
        width = opened.size[0]
    step = processor.step(parameter)
    if width == 666 and step == "segment":
        raise RuntimeError("fake segmentation error")
    if width == 555 and step == "recognize":
        return
    xml = (FIXTURES / OUTPUTS.get(width, "blank.xml")).read_text(encoding="utf-8")
    name = f"{output_file_grp}/{output_file_grp}_PHYS_0001.xml"
    (workspace.directory / output_file_grp).mkdir()
    (workspace.directory / name).write_text(xml, encoding="utf-8")
    workspace.add_file(
        output_file_grp,
        file_id=f"{output_file_grp}_PHYS_0001",
        mimetype="application/vnd.prima.page+xml",
        page_id="PHYS_0001",
        local_filename=name,
    )
