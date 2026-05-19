"""
獨立測試腳本：FFmpeg 管道直投模式 + 完美對接 DBWriter 上傳 MongoDB Atlas 雲端
升級版：支援動態輸入與滑鼠拖曳影片檔案（方案 B）
"""
import asyncio
import time
import subprocess
import numpy as np
from pathlib import Path
import imageio_ffmpeg

# 1. 匯入你專案現有的 ASR 引擎
from asr.whisper_engine import WhisperASR

# 2. 匯入你專案的 DBWriter 類別
from db.writer import DBWriter

def extract_audio_with_ffmpeg(video_path: str, target_sr: int = 16000) -> np.ndarray:
    """ 直接使用 subprocess 呼叫 imageio-ffmpeg 內建的 ffmpeg 執行檔抽取純音訊 """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(target_sr),
        "-f", "s16le", "-"
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg 執行失敗:\n{stderr.decode('utf-8', errors='ignore')}")
    return np.frombuffer(stdout, dtype=np.int16)

async def run_file_pipeline():
    TARGET_SAMPLE_RATE = 16000
    CLOUD_MONGODB_URI = "mongodb+srv://jeremy930921_db_user:Jeremy930921@cluster0.agokgyz.mongodb.net/?appName=Cluster0"
    
    print("==================================================")
    print("🚀 啟動獨立影片檔案處理 (動態輸入/拖曳模式)")
    print("==================================================")
    
    # 讓使用者在終端機輸入檔名，或直接拖曳檔案進來
    user_input = input("👉 請輸入影片檔案名稱（如 3mins.mp4），或直接將影片檔案「拖曳」進此視窗：").strip()
    
    # 修正 Windows 系統拖曳檔案時會自動帶有雙引號或單引號的 Bug
    video_raw_path = user_input.replace('"', '').replace("'", "")
    video_path = Path(video_raw_path)
    
    if not video_path.exists():
        print(f"❌ 錯誤：找不到影片檔案【{video_raw_path}】，請確認路徑或檔名是否正確。")
        return

    VIDEO_FILE = video_path.name

    # 1. 抽取音訊
    print(f"\n[1/4] 正在從 {VIDEO_FILE} 中直接解碼音訊流...")
    try:
        data = extract_audio_with_ffmpeg(str(video_path), target_sr=TARGET_SAMPLE_RATE)
        print(f" 成功抽取音訊！數據長度: {len(data)}")
    except Exception as e:
        print(f"❌ 解析影片音訊失敗: {e}")
        return

    # 2. 載入 Whisper 模型
    print("\n[2/4] 正在載入 Whisper ASR 模型 (medium)...")
    asr_engine = WhisperASR()
    if not asr_engine.load_model():
        print("❌ 錯誤：Whisper 模型載入失敗。")
        return

    # 3. 執行辨識
    print("\n[3/4] 送入 Whisper 進行語音辨識中...")
    start_time = time.time()
    result = asr_engine.transcribe(data, sample_rate=TARGET_SAMPLE_RATE)
    
    print("\n----------------【 辨識結果 】----------------")
    print(f" 偵測語言: {result.language} | 信心度: {result.confidence:.2%}")
    print(f" 辨識耗時: {time.time() - start_time:.2f} 秒")
    print("----------------------------------------------")

    if not result.text.strip():
        print("💡 提示：辨識文字為空，不執行上傳。")
        asr_engine.unload_model()
        return

    # 4. 上傳 MongoDB Atlas
    print("\n[4/4] 正在將結果寫入 MongoDB Atlas 雲端...")
    try:
        db_writer = DBWriter(mongodb_uri=CLOUD_MONGODB_URI, db_name="lecture_ai")
        inserted_id = await db_writer.write_transcript(
            text=result.text,
            confidence=result.confidence,
            session_id=f"file_session_{int(time.time())}",
            start_sec=0.0,
            end_sec=time.time() - start_time,
            language=result.language,
            source=f"video_file:{VIDEO_FILE}"
        )
        print(f" 🎉 成功！資料已寫入雲端！ID: {inserted_id}")
        await db_writer.close()
    except Exception as e:
        print(f"❌ 寫入 MongoDB 雲端失敗: {e}")
    finally:
        asr_engine.unload_model()
        print("\n任務處理完畢。")

if __name__ == "__main__":
    try:
        asyncio.run(run_file_pipeline())
    except KeyboardInterrupt:
        print("\n[手動中斷] 程式結束。")