import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from rag.retriever import Retriever
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options={"api_version": "v1beta"}
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = Retriever(mongodb_uri=os.getenv("MONGODB_URI"), db_name="lecture_ai")

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        docs = retriever.search(req.question, session_id=req.session_id)
        context = retriever.format_context(docs)
        prompt = f"請根據以下逐字稿回答問題：\n\n內容：{context}\n\n問題：{req.question}"

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        sources = [
            {
                "session_id": doc.get("session_id", ""),
                "start_sec": doc.get("start_sec", 0),
                "text_preview": doc.get("text", "")[:60],
            }
            for doc in docs
        ]
        return {"answer": response.text, "sources": sources}

    except Exception as e:
        print("🔥 錯誤詳情：")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def get_sessions():
    try:
        sessions = retriever.collection.distinct("session_id")
        return {"sessions": sorted(sessions, reverse=True)}
    except Exception as e:
        print("🔥 錯誤詳情：")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transcript/{session_id}")
async def get_transcript(session_id: str):
    try:
        docs = list(
            retriever.collection.find(
                {"session_id": session_id},
                {"_id": 0, "text": 1, "start_sec": 1, "end_sec": 1,
                 "confidence": 1, "language": 1}
            ).sort("start_sec", 1)
        )
        return {"segments": docs}
    except Exception as e:
        print("🔥 錯誤詳情：")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))