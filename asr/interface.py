"""
Abstract ASR interface and result object.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class TranscriptionResult:
    text: str
    language: str
    confidence: float
    segments: Optional[List[dict]] = None
    processing_time_ms: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.text or self.text.strip() == ""


class ASRInterface(ABC):
    def __init__(self, language: str = "zh"):
        self.language = language
        self._is_loaded = False

    @abstractmethod
    def load_model(self) -> bool:
        pass

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        pass

    @abstractmethod
    def transcribe_stream(self, audio_generator, sample_rate: int = 16000) -> TranscriptionResult:
        pass

    @abstractmethod
    def unload_model(self) -> None:
        pass

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
