import os
import logging
import time
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from llama_cpp import Llama

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NeuroGraph API",
    description="OpenAI-compatible API for the CedzLabs NeuroGraph Cognitive Router",
    version="1.0.0"
)

# Security setup
security = HTTPBearer()
API_KEY = os.getenv("API_KEY", "default-fallback-key")

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key. Use 'Bearer <YOUR_API_KEY>'")
    return credentials.credentials

llm = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "neurograph-v1"
    temperature: float = 0.1  # Lowered to 0.1 to minimize repetition in tiny models
    top_p: float = 0.9
    max_tokens: Optional[int] = 250

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
    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        n_threads=int(os.getenv("CPU_THREADS", "2")),
        n_gpu_layers=0,
        verbose=False
    )
    logger.info("✅ NeuroGraph model loaded successfully!")

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key) # Enforce API key
):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        if not any(msg["role"] == "system" for msg in messages):
            system_prompt = (
                "You are a cognitive router designed by CedzLabs. "
                "Output ONLY valid JSON knowledge graphs with 'nodes', 'edges', 'confidence', and 'source'. "
                "Use <NEEDS_MCP:tool_name> for live data and <NEEDS_ESCALATION> for complex tasks."
            )
            messages.insert(0, {"role": "system", "content": system_prompt})

        # Manual prompt formatting to match training
        prompt = ""
        for msg in messages:
            prompt += f"|{msg['role']}| {msg['content']} "
        prompt += "|assistant| "

        # Generate with strict repetition penalty
        output = llm(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repeat_penalty=1.5,  # Critical for tiny models
            stop=["|endoftext|", "\n\n", "territ"] # Aggressive stopping to prevent loops
        )

        generated_text = output["choices"][0]["text"].strip()
        prompt_tokens = output["usage"]["prompt_tokens"]
        completion_tokens = output["usage"]["completion_tokens"]

        return ChatCompletionResponse(
            id=f"chatcmpl-{os.urandom(4).hex()}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=generated_text),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        )

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during inference")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": llm is not None}