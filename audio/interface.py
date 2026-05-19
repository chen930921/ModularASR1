"""
Abstract audio interface.
"""

from abc import ABC, abstractmethod
from typing import Generator, Optional

import numpy as np


class AudioInterface(ABC):
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._is_initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        pass

    @abstractmethod
    def start_stream(self) -> bool:
        pass

    @abstractmethod
    def stop_stream(self) -> None:
        pass

    @abstractmethod
    def read_chunk(self, chunk_size: int) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def read_until_silence(
        self, max_duration_seconds: float, silence_threshold: int, silence_duration_ms: int
    ) -> np.ndarray:
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass

    def stream_chunks(self, chunk_duration_ms: int) -> Generator[np.ndarray, None, None]:
        chunk_size = int(self.sample_rate * chunk_duration_ms / 1000)
        while True:
            chunk = self.read_chunk(chunk_size)
            if chunk is not None:
                yield chunk
