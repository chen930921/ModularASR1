"""
Simple energy-based VAD segmenter.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class AudioSegment:
    audio: np.ndarray
    start_sec: float
    end_sec: float


class VADSegmenter:
    def __init__(self, sample_rate: int = 16000, silence_threshold: float = 300.0):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold

    def _is_speech(self, chunk: np.ndarray) -> bool:
        return float(np.abs(chunk).mean()) >= self.silence_threshold

    def segment(
        self,
        audio: np.ndarray,
        max_segment_seconds: float = 59.0,
        frame_ms: int = 30,
        min_silence_ms: int = 500,
    ) -> List[AudioSegment]:
        if len(audio) == 0:
            return []

        frame_size = max(1, int(self.sample_rate * frame_ms / 1000))
        silence_frames_needed = max(1, int(min_silence_ms / frame_ms))
        max_samples = int(max_segment_seconds * self.sample_rate)

        segments: List[AudioSegment] = []
        current_start = 0
        i = 0
        silent_count = 0

        while i < len(audio):
            frame = audio[i : i + frame_size]
            has_speech = self._is_speech(frame)
            if has_speech:
                silent_count = 0
            else:
                silent_count += 1

            current_len = i - current_start
            reached_max_len = current_len >= max_samples
            reached_silence_cut = silent_count >= silence_frames_needed and current_len > 0

            if reached_max_len or reached_silence_cut:
                end = i
                chunk = audio[current_start:end]
                if len(chunk) > 0:
                    segments.append(
                        AudioSegment(
                            audio=chunk,
                            start_sec=current_start / self.sample_rate,
                            end_sec=end / self.sample_rate,
                        )
                    )
                current_start = i
                silent_count = 0

            i += frame_size

        if current_start < len(audio):
            segments.append(
                AudioSegment(
                    audio=audio[current_start:],
                    start_sec=current_start / self.sample_rate,
                    end_sec=len(audio) / self.sample_rate,
                )
            )

        # Hard safety split: keep every segment strictly below 60 seconds.
        final_segments: List[AudioSegment] = []
        max_samples_hard = int(59.0 * self.sample_rate)
        for seg in segments:
            if len(seg.audio) <= max_samples_hard:
                final_segments.append(seg)
                continue
            offset = 0
            while offset < len(seg.audio):
                part = seg.audio[offset : offset + max_samples_hard]
                start_sec = seg.start_sec + (offset / self.sample_rate)
                end_sec = start_sec + (len(part) / self.sample_rate)
                final_segments.append(AudioSegment(audio=part, start_sec=start_sec, end_sec=end_sec))
                offset += max_samples_hard
        return final_segments
