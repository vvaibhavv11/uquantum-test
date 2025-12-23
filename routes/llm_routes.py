# from fastapi import APIRouter
# from services.llm_service import llm_service
# from fastapi import Request
# import asyncio

# router = APIRouter()

# @router.post("/chat")
# async def chat(request: Request):
#     body = await request.json()
#     prompt = body.get("prompt", "")
#     model = body.get("model", "llama3-8b-8192")
#     result = await llm_service.generate_completion(prompt, model)
#     return result
from fastapi import APIRouter
from pydantic import BaseModel
from services.llm_service import llm_service

router = APIRouter()

class ChatRequest(BaseModel):
    messages: list
    model: str = "llama-3.1-8b-instant"
    mode: str = "ask"
    api_keys: dict = None

@router.options("/chat")
async def chat_options():
    return {"status": "ok"}

@router.post("/chat")
async def chat(request: ChatRequest):
    result = await llm_service.generate_completion(
        request.messages, 
        request.model, 
        request.mode,
        request.api_keys
    )
    return result


