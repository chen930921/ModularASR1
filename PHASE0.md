# PHASE0

原始 `PHASE0.pdf` 已嘗試讀取，但未能提取出可用文字內容（疑似掃描版或無文字層）。

以下為本次重構執行目標（依你提供的最終指令）：

## ACTION 1-7 執行清單

1. 建立模組化目錄：`asr/`, `db/`, `vad/`, `audio/` 並補齊 `__init__.py`。
2. 將 `asr_whisper_engine.py.py` 邏輯遷移到 `asr/whisper_engine.py`。
3. 建立 `db/writer.py` 作為 MongoDB 寫入層。
4. 建立 `vad/segmenter.py` 作為 VAD 切段層。
5. 將 `audio_windows.py` 移植為 `audio/capture.py`。
6. 建立 `asr/processor.py` 串接 VAD + ASR，強制切段小於 60 秒。
7. 改寫 `main.py` 為 Standalone 模式；更新依賴與環境範例設定。
