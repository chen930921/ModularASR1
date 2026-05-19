"""
Core processor that bridges VAD and ASR.
"""

from typing import Optional

import numpy as np

from asr.interface import ASRInterface, TranscriptionResult
from db.writer import DBWriter
from vad.segmenter import VADSegmenter


class ASRProcessor:
    def __init__(
        self,
        asr_engine: ASRInterface,
        vad_segmenter: Optional[VADSegmenter] = None,
        db_writer: Optional[DBWriter] = None,
        sample_rate: int = 16000,
    ):
        self.asr_engine = asr_engine
        self.sample_rate = sample_rate
        self.vad_segmenter = vad_segmenter or VADSegmenter(sample_rate=sample_rate)
        self.db_writer = db_writer

    def process_audio(self, audio: np.ndarray) -> TranscriptionResult:
        segments = self.vad_segmenter.segment(audio, max_segment_seconds=59.0)
        if not segments:
            return TranscriptionResult(text="", language="zh", confidence=0.0, segments=[])

        merged_text = []
        details = []
        confidences = []
        total_ms = 0.0
        language = "zh"

        for seg in segments:
            result = self.asr_engine.transcribe(seg.audio, sample_rate=self.sample_rate)
            if result.text.strip():
                merged_text.append(result.text.strip())
            confidences.append(result.confidence)
            total_ms += result.processing_time_ms
            language = result.language or language
            details.append(
                {
                    "start": seg.start_sec,
                    "end": seg.end_sec,
                    "text": result.text,
                    "confidence": result.confidence,
                }
            )

        final_text = " ".join(merged_text).strip()
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        merged_result = TranscriptionResult(
            text=final_text,
            language=language,
            confidence=avg_conf,
            segments=details,
            processing_time_ms=total_ms,
        )

        if self.db_writer is not None and final_text:
            self.db_writer.write_transcript_sync(text=final_text, confidence=avg_conf, source="asr_processor")

        return merged_result
