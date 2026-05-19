"""
MongoDB writer — stores transcripts with full metadata for RAG.
"""
import asyncio
from datetime import datetime, timezone
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient


class DBWriter:
    def __init__(self, mongodb_uri: str, db_name: str = "lecture_ai"):
        self.mongodb_uri = mongodb_uri
        self.db_name = db_name
        self._async_client: Optional[AsyncIOMotorClient] = None
        self._sync_client: Optional[MongoClient] = None

    def _get_sync_col(self):
        if self._sync_client is None:
            self._sync_client = MongoClient(self.mongodb_uri)
        return self._sync_client[self.db_name]["transcripts"]

    def _get_async_col(self):
        if self._async_client is None:
            self._async_client = AsyncIOMotorClient(self.mongodb_uri)
        return self._async_client[self.db_name]["transcripts"]

    def write_transcript_sync(
        self,
        text: str,
        confidence: float,
        session_id: str,
        start_sec: float = 0.0,
        end_sec: float = 0.0,
        language: str = "zh",
        source: str = "asr_processor",
    ) -> str:
        doc = {
            "session_id": session_id,
            "text": text,
            "confidence": round(confidence, 4),
            "language": language,
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "duration_sec": round(end_sec - start_sec, 2),
            "timestamp": datetime.now(timezone.utc),
            "source": source,
            "embedding": None,   # 之後由 embedder 填入
        }
        result = self._get_sync_col().insert_one(doc)
        return str(result.inserted_id)

    async def write_transcript(
        self,
        text: str,
        confidence: float,
        session_id: str,
        start_sec: float = 0.0,
        end_sec: float = 0.0,
        language: str = "zh",
        source: str = "asr_processor",
    ) -> str:
        doc = {
            "session_id": session_id,
            "text": text,
            "confidence": round(confidence, 4),
            "language": language,
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "duration_sec": round(end_sec - start_sec, 2),
            "timestamp": datetime.now(timezone.utc),
            "source": source,
            "embedding": None,
        }
        col = self._get_async_col()
        result = await col.insert_one(doc)
        return str(result.inserted_id)

    async def close(self):
        if self._async_client:
            self._async_client.close()
        if self._sync_client:
            self._sync_client.close()