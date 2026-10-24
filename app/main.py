import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from llama_cpp import Llama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NeuroGraph API",
    description="OpenAI-compatible API for the CedzLabs NeuroGraph Cognitive Router",
    version="1.0.0"
)

# Global model instance
llm = None

# OpenAI-compatible Request Schema
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "neurograph-v1"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = 512

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Dict[str, int]

@app.on_event("startup")
async def load_model():
    global llm
    model_path = os.getenv("MODEL_PATH", "./models/neurograph-q4_k_m.gguf")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        raise FileNotFoundError(f"Model not found at {model_path}")

    logger.info(f"Loading NeuroGraph model from {model_path}...")
    # Optimized for CPU inference
    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        n_threads=int(os.getenv("CPU_THREADS", "2")),
        n_gpu_layers=0, # Force CPU
        verbose=False,
        chat_format="chatml" # Matches our |system| |user| |assistant| format
    )
    logger.info("✅ NeuroGraph model loaded successfully!")

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Extract user messages
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Inject the mandatory NeuroGraph system prompt if not present
        if not any(msg["role"] == "system" for msg in messages):
            system_prompt = (
                "You are a cognitive router designed by CedzLabs. "
                "Output ONLY valid JSON knowledge graphs with 'nodes', 'edges', 'confidence', and 'source'. "
                "Use <NEEDS_MCP:tool_name> for live data and <NEEDS_ESCALATION> for complex tasks."
            )
            messages.insert(0, {"role": "system", "content": system_prompt})

        # Generate response
        response = llm.create_chat_completion(
            messages=messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=["|endoftext|"]
        )

        # Format to match OpenAI schema exactly
        choice = response["choices"][0]
        return ChatCompletionResponse(
            id=f"chatcmpl-{os.urandom(4).hex()}",
            object="chat.completion",
            created=response.get("created", 0),
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role=choice["message"]["role"],
                        content=choice["message"]["content"]
                    ),
                    finish_reason=choice.get("finish_reason", "stop")
                )
            ],
            usage=response.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        )

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during inference")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": llm is not None}