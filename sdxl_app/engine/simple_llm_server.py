import argparse
import logging
import time
from typing import List, Optional, Union, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Simple LLM Server (OpenAI Compatible)")

# 全局模型和分词器
model = None
tokenizer = None
MODEL_NAME = ""

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-simple"
    object: str = "chat.completion"
    created: int = int(time.time())
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Dict[str, int]

def load_model(model_path: str, device: str = "cuda"):
    global model, tokenizer, MODEL_NAME
    logger.info(f"Loading model from {model_path}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map=device, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        MODEL_NAME = model_path
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    global model, tokenizer
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 构造 prompt
    messages = [msg.model_dump() for msg in req.messages]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 生成
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        do_sample=True if req.temperature > 0 else False
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return ChatCompletionResponse(
        model=MODEL_NAME,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop"
            )
        ],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0} # 简化处理
    )

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user",
            }
        ]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--port", type=int, default=8001, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    args = parser.parse_args()

    load_model(args.model)
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
