from app.align import assign_speakers, Segment
from app.asr import ASRSegment, Word
from app.diarization import DiarizationSegment


def test_assign_speakers_simple():
    asr = [ASRSegment(start=0.0, end=1.0, text="hello", words=[])]
    diar = [DiarizationSegment(start=0.0, end=1.0, speaker="SPEAKER_00")]
    result = assign_speakers(asr, diar)
    assert result[0].speaker == "Speaker 1"
