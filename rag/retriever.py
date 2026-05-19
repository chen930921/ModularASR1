import os
from google import genai
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

class Retriever:
    def __init__(self, mongodb_uri, db_name):
        self.mongo_client = MongoClient(mongodb_uri)
        self.db = self.mongo_client[db_name]
        self.collection = self.db["transcripts"]
        self.genai_client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY"),
            http_options={"api_version": "v1beta"}
        )

    def _get_embedding(self, text):
        result = self.genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
        )
        return result.embeddings[0].values

    def search(self, query, session_id=None, limit=5):
        query_vector = self._get_embedding(query)

        if session_id:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "transcript_vector_index",
                        "path": "embedding",
                        "queryVector": query_vector,
                        "numCandidates": 100,
                        "limit": limit,
                        "filter": {"session_id": {"$eq": session_id}}
                    }
                }
            ]
        else:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "transcript_vector_index",
                        "path": "embedding",
                        "queryVector": query_vector,
                        "numCandidates": 100,
                        "limit": limit
                    }
                }
            ]

        return list(self.collection.aggregate(pipeline))

    def format_context(self, docs) -> str:
        parts = []
        for doc in docs:
            start = doc.get("start_sec", 0)
            m, s = int(start // 60), int(start % 60)
            parts.append(f"[{m:02d}:{s:02d}] {doc.get('text', '')}")
        return "\n".join(parts)