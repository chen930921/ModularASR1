"""
Windows Microphone Implementation.
"""
import time
from typing import Optional
import numpy as np

try:
    import pyaudio
except ImportError:
    raise ImportError("PyAudio not found. Install with: pip install pyaudio")

from audio.interface import AudioInterface

AUDIO_CONFIG = {
    "sample_rate": 16000,
    "channels": 1,
    "chunk_duration_ms": 100,
}

RECORDING_CONFIG = {
    "max_duration_seconds": 60,      # 每段最長 60 秒，超過自動切
    "silence_threshold": 300,
    "silence_duration_ms": 1500,     # 停頓 1.5 秒後送出
}


class WindowsMicrophone(AudioInterface):
    def __init__(
        self,
        sample_rate: int = AUDIO_CONFIG["sample_rate"],
        channels: int = AUDIO_CONFIG["channels"],
        device_index: Optional[int] = None
    ):
        super().__init__(sample_rate, channels)
        self.device_index = device_index
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._format = pyaudio.paInt16
        self._chunk_size = int(sample_rate * AUDIO_CONFIG["chunk_duration_ms"] / 1000)

    def initialize(self) -> bool:
        try:
            self._pyaudio = pyaudio.PyAudio()
            if self.device_index is None:
                self._select_default_device()
            self._is_initialized = True
            print(f"[Audio] Initialized (device: {self.device_index})")
            return True
        except Exception as e:
            print(f"[Audio] Initialization failed: {e}")
            return False

    def _select_default_device(self) -> None:
        try:
            info = self._pyaudio.get_default_input_device_info()
            self.device_index = info["index"]
            print(f"[Audio] Using: {info['name']}")
        except IOError:
            self.device_index = None

    def list_devices(self) -> list:
        if not self._pyaudio:
            self._pyaudio = pyaudio.PyAudio()
        devices = []
        for i in range(self._pyaudio.get_device_count()):
            info = self._pyaudio.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append({
                    "index": i,
                    "name": info["name"],
                    "channels": info["maxInputChannels"],
                    "sample_rate": int(info["defaultSampleRate"])
                })
        return devices

    def start_stream(self) -> bool:
        if not self._is_initialized:
            return False
        try:
            self._stream = self._pyaudio.open(
                format=self._format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self._chunk_size
            )
            print("[Audio] Stream started")
            return True
        except Exception as e:
            print(f"[Audio] Failed to start stream: {e}")
            return False

    def stop_stream(self) -> None:
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                print(f"[Audio] Error stopping stream: {e}")
            finally:
                self._stream = None

    def read_chunk(self, chunk_size: Optional[int] = None) -> Optional[np.ndarray]:
        if not self._stream:
            return None
        chunk_size = chunk_size or self._chunk_size
        try:
            data = self._stream.read(chunk_size, exception_on_overflow=False)
            return np.frombuffer(data, dtype=np.int16)
        except Exception as e:
            print(f"[Audio] Read error: {e}")
            return None

    def read_until_silence(
        self,
        max_duration_seconds: float = RECORDING_CONFIG["max_duration_seconds"],
        silence_threshold: int = RECORDING_CONFIG["silence_threshold"],
        silence_duration_ms: int = RECORDING_CONFIG["silence_duration_ms"],
    ) -> np.ndarray:
        """
        錄音直到靜音 1.5 秒或超過 max_duration_seconds。
        等到有聲音才開始計算靜音，避免一開始就立刻結束。
        """
        frames = []
        silent_chunks = 0
        chunks_for_silence = int(silence_duration_ms / AUDIO_CONFIG["chunk_duration_ms"])
        max_chunks = int(max_duration_seconds * 1000 / AUDIO_CONFIG["chunk_duration_ms"])
        has_speech = False

        for _ in range(max_chunks):
            chunk = self.read_chunk()
            if chunk is None:
                continue
            frames.append(chunk)
            energy = float(np.abs(chunk).mean())

            if energy >= silence_threshold:
                has_speech = True
                silent_chunks = 0
            else:
                if has_speech:
                    silent_chunks += 1

            if has_speech and silent_chunks >= chunks_for_silence:
                break

        if not frames:
            return np.array([], dtype=np.int16)
        return np.concatenate(frames)

    def stream(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_stream()

    def cleanup(self) -> None:
        self.stop_stream()
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
        self._is_initialized = False
        print("[Audio] Cleaned up")


AudioRecorder = WindowsMicrophone

def create_audio_source(backend: str = "windows", **kwargs) -> WindowsMicrophone:
    if backend == "windows":
        return WindowsMicrophone(**kwargs)
    raise ValueError(f"Unknown audio backend: {backend}")