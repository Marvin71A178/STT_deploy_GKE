import os , sys

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from Music.api.demos.predict import _do_predictions , load_model
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from starlette.concurrency import run_in_threadpool
import uvicorn



app = FastAPI()
origins = [
    "http://localhost:3000",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class process_Mood_pd(BaseModel):
    TestData: str

class process_Music_pd(BaseModel):
    texts: str
    duration: float
class process_Novel_class(BaseModel):
    url: str


@app.get("/")
async def root():
    return {"message": "Welcome to the AudioCraft API. Use /mood_analyze/ to perform mood analysis and /music_generate/ to generate music based on mood."}

@app.post("/music_generate/")
async def perform_music_pd(request: process_Music_pd):
    try:
        result = await run_in_threadpool(
            _do_predictions ,
            texts = [request.texts],
            duration = request.duration,
        )
        return FileResponse(result[0] , filename="generated_sound.wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    load_model()
    uvicorn.run(app , host = '0.0.0.0' , port = 801)
    