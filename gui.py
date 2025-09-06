from __future__ import annotations

"""
Whisper–Pyannote GUI
--------------------
A desktop GUI for your transcription + diarization app with drag‑and‑drop,
mode presets, and tunable advanced parameters. Built with PyQt6 and designed
to call your existing modules:

  app.asr         -> load_model, transcribe
  app.diarization -> load_pipeline, diarize
  app.align       -> assign_speakers
  app.config      -> load_config

Run:
  pip install PyQt6
  python gui.py

Notes:
- Creates an output folder per input file (next to input, or under a chosen root).
- Threaded processing with progress + log console.
- Three modes: Two Speakers, Bounded, Max Accuracy.
- Advanced knobs surfaced with sensible defaults.
- Requires your existing .env (HF token) for diarization.
"""

import sys
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QLabel, QLineEdit,
    QFileDialog, QComboBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QPlainTextEdit, QProgressBar, QMessageBox
)

# Import your app modules (assumes this gui.py is in repo root)
from app.asr import load_model, transcribe, ASRSegment
from app.diarization import load_pipeline, diarize, DiarizationSegment
from app.align import assign_speakers, Segment
from app.config import load_config, ConfigError

try:
    import torch
except Exception:
    torch = None  # Graceful handling if torch isn't importable


# --------------------------- Data structures --------------------------- #

@dataclass
class GUISettings:
    mode: str  # 'TwoSpeakers' | 'Bounded' | 'MaxAccuracy'
    model_name: str
    language: str
    enable_vad: bool
    vad_min_sil_ms: int
    hf_token_file: Path
    num_speakers: Optional[int]
    min_speakers: Optional[int]
    max_speakers: Optional[int]
    # Advanced assignment knobs
    boundary_eps: float
    nearest_eps: float
    fill_unknown_window: float
    snap_window: float
    min_snap_shift: float
    # Output root
    output_mode: str  # 'next_to_input' | 'custom_root'
    output_root: Optional[Path]


# --------------------------- Worker thread ---------------------------- #

class TranscribeWorker(QThread):
    log = pyqtSignal(str)
    file_started = pyqtSignal(str, int, int)  # path, idx, total
    file_done = pyqtSignal(str, bool)         # path, success
    overall_done = pyqtSignal()

    def __init__(self, files: List[Path], settings: GUISettings, parent=None):
        super().__init__(parent)
        self.files = files
        self.s = settings
        self._stop = False

    def stop(self):
        self._stop = True

    def _write_outputs(self, segments: List[Segment], out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        # TXT
        txt = out_dir / "transcript.txt"
        with txt.open("w", encoding="utf-8") as f:
            for s in segments:
                f.write(f"[{s.speaker}] {s.text}\n")
        # SRT
        def srt_time(t: float) -> str:
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = t % 60
            return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
        srt = out_dir / "subtitles.srt"
        with srt.open("w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                f.write(f"{i}\n{srt_time(seg.start)} --> {srt_time(seg.end)}\n[{seg.speaker}] {seg.text}\n\n")
        # JSON (rich)
        import json
        payload = [
            {
                "start": seg.start,
                "end": seg.end,
                "speaker": seg.speaker,
                "text": seg.text,
                "words": [
                    {"start": w.start, "end": w.end, "word": w.word, "prob": getattr(w, "prob", None)}
                    for w in seg.words
                ],
            }
            for seg in segments
        ]
        (out_dir / "segments.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(self):
        try:
            total = len(self.files)
            # Device & model cache dir
            device = "cuda" if (torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available()) else "cpu"
            models_dir = Path(".models")

            # Model selection (MaxAccuracy overrides)
            model_name = self.s.model_name
            if self.s.mode == "MaxAccuracy":
                model_name = "large-v2"

            self.log.emit(f"Loading ASR model: {model_name} on {device}…")
            asr_model = load_model(model_name, device, models_dir)

            # Diarization pipeline (if token present)
            hf_token = None
            try:
                cfg = load_config(self.s.hf_token_file)
                hf_token = cfg.hf_token if cfg else None
            except ConfigError as e:
                self.log.emit(f"Config error: {e}")
            pipeline = None
            if hf_token:
                self.log.emit("Loading diarization pipeline…")
                pipeline = load_pipeline(hf_token, models_dir)
            else:
                self.log.emit("No HF token found; diarization will be skipped (speakers = Unknown).")

            for idx, in_path in enumerate(self.files, start=1):
                if self._stop:
                    break
                self.file_started.emit(str(in_path), idx, total)
                try:
                    in_path = in_path.resolve()
                    # Output folder rules
                    if self.s.output_mode == "custom_root" and self.s.output_root:
                        base = in_path.stem
                        out_dir = self.s.output_root / base
                    else:
                        out_dir = in_path.parent / in_path.stem

                    self.log.emit(f"Transcribing: {in_path}")
                    # Transcription
                    asr_segments: List[ASRSegment] = transcribe(
                        asr_model,
                        in_path,
                        None if self.s.language.lower() == "auto" else self.s.language,
                        enable_vad=self.s.enable_vad,
                        vad_min_silence_ms=self.s.vad_min_sil_ms,
                    )

                    # Diarization
                    diar_segments: List[DiarizationSegment] = []
                    if pipeline is not None:
                        kwargs = {}
                        if self.s.mode == "TwoSpeakers":
                            kwargs["num_speakers"] = 2
                        elif self.s.mode == "Bounded":
                            if self.s.min_speakers:
                                kwargs["min_speakers"] = self.s.min_speakers
                            if self.s.max_speakers:
                                kwargs["max_speakers"] = self.s.max_speakers
                        elif self.s.mode == "MaxAccuracy":
                            # Use explicit num if provided; otherwise bounded
                            if self.s.num_speakers and self.s.num_speakers > 0:
                                kwargs["num_speakers"] = self.s.num_speakers
                            else:
                                if self.s.min_speakers:
                                    kwargs["min_speakers"] = self.s.min_speakers
                                if self.s.max_speakers:
                                    kwargs["max_speakers"] = self.s.max_speakers
                        self.log.emit("Running diarization…")
                        diar_segments = diarize(pipeline, in_path, **kwargs)
                    else:
                        self.log.emit("(Skipping diarization: no token)")

                    # Word‑level assignment (with robust unknown handling)
                    self.log.emit("Assigning speakers at word level…")
                    final_segments: List[Segment] = assign_speakers(
                        asr_segments,
                        diar_segments,
                        snap_window=self.s.snap_window,
                        min_snap_shift=self.s.min_snap_shift,
                        boundary_eps=self.s.boundary_eps,
                        nearest_eps=self.s.nearest_eps,
                        fill_unknown_window=self.s.fill_unknown_window,
                    )

                    # Write outputs
                    self._write_outputs(final_segments, out_dir)
                    self.log.emit(f"✅ Done: {in_path} → {out_dir}")
                    self.file_done.emit(str(in_path), True)

                except Exception as e:
                    self.log.emit("\n" + traceback.format_exc())
                    self.file_done.emit(str(in_path), False)

        finally:
            self.overall_done.emit()


# --------------------------- Main Window ------------------------------ #

class DropList(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(self.SelectionMode.ExtendedSelection)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dropEvent(self, e: QDropEvent):
        if e.mimeData().hasUrls():
            for url in e.mimeData().urls():
                p = Path(url.toLocalFile())
                if p.is_file():
                    self.addItem(str(p))
            e.acceptProposedAction()
        else:
            super().dropEvent(e)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper–Pyannote GUI")
        self.resize(1060, 720)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # File list + controls
        files_box = QGroupBox("Files (drag & drop or Add…) ")
        files_layout = QVBoxLayout(files_box)
        self.list = DropList()
        files_btns = QHBoxLayout()
        btn_add = QPushButton("Add files…")
        btn_remove = QPushButton("Remove selected")
        btn_clear = QPushButton("Clear")
        files_btns.addWidget(btn_add)
        files_btns.addWidget(btn_remove)
        files_btns.addWidget(btn_clear)
        files_layout.addWidget(self.list)
        files_layout.addLayout(files_btns)

        # Mode + basic params
        mode_box = QGroupBox("Mode & Basics")
        form = QFormLayout(mode_box)

        self.mode = QComboBox()
        self.mode.addItems(["TwoSpeakers", "Bounded", "MaxAccuracy"])
        self.model = QComboBox()
        self.model.addItems(["tiny", "base", "small", "medium.en", "medium", "large-v2"])  # GUI choice; MaxAccuracy overrides
        self.lang = QLineEdit("auto")

        self.vad_enable = QCheckBox("Enable VAD (recommended)")
        self.vad_enable.setChecked(True)
        self.vad_sil = QSpinBox(); self.vad_sil.setRange(0, 4000); self.vad_sil.setValue(250)

        self.hf_path = QLineEdit(".env")
        hf_browse = QPushButton("…")

        # Speaker constraints
        self.num_spk = QSpinBox(); self.num_spk.setRange(0, 32); self.num_spk.setValue(2)
        self.min_spk = QSpinBox(); self.min_spk.setRange(0, 32); self.min_spk.setValue(2)
        self.max_spk = QSpinBox(); self.max_spk.setRange(0, 32); self.max_spk.setValue(3)

        # Output options
        out_box = QGroupBox("Output")
        out_form = QFormLayout(out_box)
        self.out_next = QCheckBox("Create folder next to each input file")
        self.out_next.setChecked(True)
        self.out_root = QLineEdit("")
        out_browse = QPushButton("…")

        # Advanced knobs
        adv_box = QGroupBox("Advanced (speaker assignment & boundary tuning)")
        adv_form = QFormLayout(adv_box)
        self.boundary_eps = QDoubleSpinBox(); self.boundary_eps.setRange(0.0, 2.0); self.boundary_eps.setSingleStep(0.01); self.boundary_eps.setDecimals(3); self.boundary_eps.setValue(0.12)
        self.nearest_eps = QDoubleSpinBox(); self.nearest_eps.setRange(0.0, 2.0); self.nearest_eps.setSingleStep(0.01); self.nearest_eps.setDecimals(3); self.nearest_eps.setValue(0.35)
        self.fill_unknown = QDoubleSpinBox(); self.fill_unknown.setRange(0.0, 2.0); self.fill_unknown.setSingleStep(0.01); self.fill_unknown.setDecimals(3); self.fill_unknown.setValue(0.40)
        self.snap_window = QDoubleSpinBox(); self.snap_window.setRange(0.0, 2.0); self.snap_window.setSingleStep(0.01); self.snap_window.setDecimals(3); self.snap_window.setValue(0.60)
        self.min_snap = QDoubleSpinBox(); self.min_snap.setRange(0.0, 2.0); self.min_snap.setSingleStep(0.01); self.min_snap.setDecimals(3); self.min_snap.setValue(0.08)

        # Buttons + progress + log
        ctrl = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.progress = QProgressBar(); self.progress.setRange(0, 100)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True)

        # Assemble forms
        form.addRow("Mode", self.mode)
        form.addRow("Whisper model", self.model)
        form.addRow("Language (auto or code)", self.lang)
        form.addRow(self.vad_enable)
        form.addRow("VAD min silence (ms)", self.vad_sil)
        form.addRow(QLabel("HuggingFace .env (HF_TOKEN)"))
        row_hf = QHBoxLayout(); row_hf.addWidget(self.hf_path); row_hf.addWidget(hf_browse)
        form.addRow(row_hf)

        out_form.addRow(self.out_next)
        row_out = QHBoxLayout(); row_out.addWidget(self.out_root); row_out.addWidget(out_browse)
        out_form.addRow("Custom output root", row_out)

        adv_form.addRow("Exact speakers (TwoSpeakers/MaxAcc)", self.num_spk)
        adv_form.addRow("Min speakers (Bounded)", self.min_spk)
        adv_form.addRow("Max speakers (Bounded)", self.max_spk)
        adv_form.addRow("Boundary tolerance (s)", self.boundary_eps)
        adv_form.addRow("Nearest diar fallback (s)", self.nearest_eps)
        adv_form.addRow("Fill-unknown window (s)", self.fill_unknown)
        adv_form.addRow("Snap window (s)", self.snap_window)
        adv_form.addRow("Min snap shift (s)", self.min_snap)

        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)
        ctrl.addStretch(1)
        ctrl.addWidget(self.progress)

        # Tooltips (suggested defaults rationale)
        self.mode.setToolTip("TwoSpeakers = force 2; Bounded = min/max; MaxAccuracy = Whisper large-v2 + optional speaker hints")
        self.boundary_eps.setToolTip("Treat words within this distance of a diar boundary as inside it (default 0.12s)")
        self.nearest_eps.setToolTip("If no segment contains the word midpoint, snap to nearest diar segment within this window (default 0.35s)")
        self.fill_unknown.setToolTip("If a word remains unknown, adopt nearest neighbor speaker within this window (default 0.40s)")
        self.snap_window.setToolTip("When splitting runs, prefer diar boundaries within this window (default 0.60s)")
        self.min_snap.setToolTip("Merge micro-runs shorter than this (default 0.08s)")

        # Layout
        layout.addWidget(files_box)
        layout.addWidget(mode_box)
        layout.addWidget(out_box)
        layout.addWidget(adv_box)
        layout.addLayout(ctrl)
        layout.addWidget(QLabel("Log:"))
        layout.addWidget(self.log)

        # Connections
        btn_add.clicked.connect(self._on_add_files)
        btn_remove.clicked.connect(self._on_remove)
        btn_clear.clicked.connect(self.list.clear)
        hf_browse.clicked.connect(self._on_pick_hf)
        out_browse.clicked.connect(self._on_pick_out)
        self.mode.currentTextChanged.connect(self._on_mode_changed)
        self.out_next.toggled.connect(self._on_out_toggle)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)

        # Defaults behavior
        self._on_mode_changed(self.mode.currentText())
        self._on_out_toggle(self.out_next.isChecked())

    # --------- UI helpers ---------

    def _append_log(self, text: str):
        self.log.appendPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _on_add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Add audio/video files")
        for f in files:
            self.list.addItem(f)

    def _on_remove(self):
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))

    def _on_pick_hf(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select .env with HF_TOKEN", filter=".env files (*.env);;All files (*)")
        if p:
            self.hf_path.setText(p)

    def _on_pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output root directory")
        if d:
            self.out_root.setText(d)
            if d:
                self.out_next.setChecked(False)

    def _on_mode_changed(self, mode: str):
        # Enable/disable controls based on mode
        if mode == "TwoSpeakers":
            self.num_spk.setEnabled(True)
            self.num_spk.setValue(2)
            self.min_spk.setEnabled(False)
            self.max_spk.setEnabled(False)
            # Model user choice honored
            self.model.setEnabled(True)
        elif mode == "Bounded":
            self.num_spk.setEnabled(False)
            self.min_spk.setEnabled(True)
            self.max_spk.setEnabled(True)
            self.model.setEnabled(True)
        else:  # MaxAccuracy
            self.num_spk.setEnabled(True)   # optional hint
            self.min_spk.setEnabled(True)
            self.max_spk.setEnabled(True)
            # Model will be overridden to large-v2; keep UI enabled but show hint
            self.model.setCurrentText("large-v2")

    def _on_out_toggle(self, checked: bool):
        self.out_root.setEnabled(not checked)

    def _collect_settings(self) -> Optional[GUISettings]:
        items = [self.list.item(i).text() for i in range(self.list.count())]
        files = [Path(p) for p in items]
        if not files:
            QMessageBox.warning(self, "No files", "Please add at least one file.")
            return None

        mode = self.mode.currentText()
        model_name = self.model.currentText()
        language = self.lang.text().strip() or "auto"
        enable_vad = self.vad_enable.isChecked()
        vad_min_sil = int(self.vad_sil.value())
        hf_file = Path(self.hf_path.text().strip()) if self.hf_path.text().strip() else Path(".env")

        num = int(self.num_spk.value()) if self.num_spk.isEnabled() else None
        mn = int(self.min_spk.value()) if self.min_spk.isEnabled() else None
        mx = int(self.max_spk.value()) if self.max_spk.isEnabled() else None

        boundary_eps = float(self.boundary_eps.value())
        nearest_eps = float(self.nearest_eps.value())
        fill_unknown = float(self.fill_unknown.value())
        snap_window = float(self.snap_window.value())
        min_snap = float(self.min_snap.value())

        if self.out_next.isChecked():
            output_mode = "next_to_input"; output_root = None
        else:
            oroot = self.out_root.text().strip()
            if not oroot:
                QMessageBox.warning(self, "Output", "Choose a custom output root or enable 'next to input'.")
                return None
            output_mode = "custom_root"; output_root = Path(oroot)

        self._files = files  # store for worker creation
        return GUISettings(
            mode=mode,
            model_name=model_name,
            language=language,
            enable_vad=enable_vad,
            vad_min_sil_ms=vad_min_sil,
            hf_token_file=hf_file,
            num_speakers=num,
            min_speakers=mn,
            max_speakers=mx,
            boundary_eps=boundary_eps,
            nearest_eps=nearest_eps,
            fill_unknown_window=fill_unknown,
            snap_window=snap_window,
            min_snap_shift=min_snap,
            output_mode=output_mode,
            output_root=output_root,
        )

    def _on_start(self):
        s = self._collect_settings()
        if s is None:
            return
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setRange(0, 0)  # indeterminate during load
        self.log.clear()

        self.worker = TranscribeWorker(self._files, s)
        self.worker.log.connect(self._append_log)
        self.worker.file_started.connect(self._on_file_started)
        self.worker.file_done.connect(self._on_file_done)
        self.worker.overall_done.connect(self._on_all_done)
        self.worker.start()

    def _on_stop(self):
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.stop()
            self._append_log("\n⏹️  Stop requested. Will finish current file then stop.")

    def _on_file_started(self, path: str, idx: int, total: int):
        self.progress.setRange(0, total)
        self.progress.setValue(idx - 1)
        self._append_log(f"\n--- [{idx}/{total}] {path} ---")

    def _on_file_done(self, path: str, ok: bool):
        self._append_log("OK" if ok else "FAILED")
        self.progress.setValue(self.progress.value() + 1)

    def _on_all_done(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self._append_log("\n✅ All done.")


# --------------------------- Entrypoint ------------------------------- #

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
