"""
Embedder — 把 transcripts 裡還沒有 embedding 的文字轉成向量存回去。
跑完錄音後執行一次，或在背景定時跑。
"""
import os
import time
from typing import List
from google import genai
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"
BATCH_SIZE = 100

def get_embedding(client: genai.Client, text: str) -> List[float]:
    text = text.replace("\n", " ").strip()
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
    )
    return result.embeddings[0].values


def embed_all_pending(mongodb_uri: str, db_name: str = "lecture_ai"):
    """把所有 embedding=null 的文件補上向量。"""
    genai_client = genai.Client(
        api_key=os.getenv("GOOGLE_API_KEY"),
        http_options={"api_version": "v1beta"}
    )
    mongo = MongoClient(mongodb_uri)
    col = mongo[db_name]["transcripts"]

    pending = list(col.find({"embedding": None, "text": {"$ne": ""}}))
    print(f"[Embedder] {len(pending)} 筆待處理")

    for i, doc in enumerate(pending):
        try:
            vec = get_embedding(genai_client, doc["text"])
            col.update_one({"_id": doc["_id"]}, {"$set": {"embedding": vec}})
            if (i + 1) % 10 == 0:
                print(f"[Embedder] {i+1}/{len(pending)} 完成")
            time.sleep(0.05)
        except Exception as e:
            print(f"[Embedder] 失敗 {doc['_id']}: {e}")

    print("[Embedder] 全部完成")
    mongo.close()


if __name__ == "__main__":
    embed_all_pending(os.getenv("MONGODB_URI"))