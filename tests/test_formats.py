import json
from pathlib import Path

from app.align import Segment
from app.formats import write_json, write_srt, write_txt


def test_write_formats(tmp_path: Path):
    segs = [Segment(start=0.0, end=1.0, speaker="Speaker 1", text="hi", words=[])]
    txt = tmp_path / "out.txt"
    srt = tmp_path / "out.srt"
    js = tmp_path / "out.json"
    write_txt(segs, txt)
    write_srt(segs, srt)
    write_json(segs, js)
    assert "Speaker 1" in txt.read_text()
    assert "00:00:00,000 --> 00:00:01,000" in srt.read_text()
    data = json.loads(js.read_text())
    assert data["segments"][0]["speaker"] == "Speaker 1"
