"""
Whisper ASR implementation using faster-whisper.
"""
import time
from typing import Optional
import numpy as np
from faster_whisper import WhisperModel
from asr.interface import ASRInterface, TranscriptionResult

ASR_CONFIG = {
    "model_size": "medium",
    "language": None,          # None = 自動偵測，支援中英混合
    "task": "transcribe",
    "device": "cuda",
    "compute_type": "float16",
    "beam_size": 5,
    "vad_filter": True,
    "vad_parameters": {
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 400,
    },
}

class WhisperASR(ASRInterface):
    def __init__(
        self,
        model_size: str = ASR_CONFIG["model_size"],
        language: Optional[str] = ASR_CONFIG["language"],
        device: str = ASR_CONFIG["device"],
        compute_type: str = ASR_CONFIG["compute_type"],
    ):
        super().__init__(language or "zh")
        self.model_size = model_size
        self.language = language      # None = 自動偵測
        self.device = device
        self.compute_type = compute_type
        self._model: Optional[WhisperModel] = None
        self.beam_size = ASR_CONFIG["beam_size"]
        self.vad_filter = ASR_CONFIG["vad_filter"]
        self.vad_parameters = ASR_CONFIG["vad_parameters"]

    def load_model(self) -> bool:
        try:
            print(f"[ASR] Loading Whisper {self.model_size} on {self.device}...")
            started = time.time()
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=None,
                local_files_only=False,
            )
            print(f"[ASR] Model loaded in {time.time() - started:.2f}s")
            self._is_loaded = True
            return True
        except Exception as exc:
            print(f"[ASR] Failed to load model: {exc}")
            if self.device == "cuda":
                print("[ASR] Falling back to CPU int8...")
                self.device = "cpu"
                self.compute_type = "int8"
                return self.load_model()
            return False

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        return audio

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        if not self._is_loaded or self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if len(audio) == 0:
            return TranscriptionResult(text="", language="zh", confidence=0.0)

        started = time.time()
        audio_float = self._preprocess_audio(audio)

        try:
            segments, info = self._model.transcribe(
                audio_float,
                language=self.language,        # None = 自動偵測
                task=ASR_CONFIG["task"],
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                vad_parameters=self.vad_parameters,
                condition_on_previous_text=True,   # 長錄音保持上下文
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )

            segment_list = []
            full_text_parts = []
            total_confidence = 0.0

            for seg in segments:
                segment_list.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "avg_logprob": seg.avg_logprob,
                })
                full_text_parts.append(seg.text)
                total_confidence += np.exp(seg.avg_logprob)

            text = "".join(full_text_parts).strip()
            avg_conf = total_confidence / len(segment_list) if segment_list else 0.0
            elapsed_ms = (time.time() - started) * 1000

            return TranscriptionResult(
                text=text,
                language=info.language,
                confidence=avg_conf,
                segments=segment_list,
                processing_time_ms=elapsed_ms,
            )
        except Exception as exc:
            print(f"[ASR] Transcription error: {exc}")
            return TranscriptionResult(
                text="",
                language="zh",
                confidence=0.0,
                processing_time_ms=(time.time() - started) * 1000,
            )

    def transcribe_stream(self, audio_generator, sample_rate: int = 16000) -> TranscriptionResult:
        chunks = list(audio_generator)
        if not chunks:
            return TranscriptionResult(text="", language="zh", confidence=0.0)
        return self.transcribe(np.concatenate(chunks), sample_rate=sample_rate)

    def unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._is_loaded = False
            print("[ASR] Model unloaded")

def create_asr_engine(engine: str = "whisper", **kwargs) -> ASRInterface:
    if engine != "whisper":
        raise ValueError(f"Unknown ASR engine: {engine}")
    return WhisperASR(**kwargs)

WhisperASREngine = WhisperASR