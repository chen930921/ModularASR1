import argparse
import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

try:
    from audio.capture import AudioRecorder
    from asr.processor import ASRProcessor
    from asr.whisper_engine import WhisperASREngine
    from vad.segmenter import VADSegmenter
    from db.writer import DBWriter
except ImportError as e:
    print(f"匯入失敗: {e}")
    sys.exit(1)

load_dotenv()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", type=str,
                        default=datetime.now().strftime("session_%Y%m%d_%H%M"))
    args = parser.parse_args()

    print(f"系統啟動中... Session: {args.session_id}")

    engine = WhisperASREngine()
    if not engine.load_model():
        print("模型載入失敗")
        return

    asr_processor = ASRProcessor(asr_engine=engine)

    # 修改：db_name 改成 lecture_ai
    db = DBWriter(mongodb_uri=os.getenv("MONGODB_URI"), db_name="lecture_ai")

    recorder = AudioRecorder()
    if not recorder.initialize() or not recorder.start_stream():
        print("錄音設備啟動失敗")
        return

    total_segments = 0
    session_start = datetime.now()

    try:
        print(f"\n系統就緒，開始說話... (Ctrl+C 停止)")
        print(f"Session ID: {args.session_id}\n")

        while True:
            print("[等待說話...]")
            audio_chunk = recorder.read_until_silence()

            if audio_chunk is None or len(audio_chunk) == 0:
                continue

            duration_sec = len(audio_chunk) / recorder.sample_rate
            if duration_sec < 0.5:
                continue

            print(f"[錄到 {duration_sec:.1f}s，辨識中...]")
            result = asr_processor.process_audio(audio_chunk)

            if result.text and result.text.strip():
                total_segments += 1
                elapsed = (datetime.now() - session_start).seconds
                print(f"[{elapsed//60:02d}:{elapsed%60:02d}] {result.text}")
                print(f"         語言:{result.language} 信心:{result.confidence:.2f}\n")

                start_sec = elapsed - duration_sec
                await db.write_transcript(
                    text=result.text,
                    confidence=result.confidence,
                    session_id=args.session_id,
                    start_sec=max(0, start_sec),
                    end_sec=elapsed,
                    language=result.language or "zh",
                )
            else:
                print("[靜音或無法辨識，略過]\n")

    except KeyboardInterrupt:
        elapsed = (datetime.now() - session_start).seconds
        print(f"\n停止錄音。共 {total_segments} 段，{elapsed//60} 分 {elapsed%60} 秒")
    finally:
        recorder.cleanup()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())