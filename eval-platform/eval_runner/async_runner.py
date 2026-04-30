import asyncio
import litellm
import json 
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv("/Users/kritikachoudhary/Desktop/multi-agent-RAG-system/.env")
async def run_single(model:str, prompt: str)-> dict:
    response = await litellm.acompletion(
        model=model,
        messages=[{"role":"user","content":prompt}])
    print("\n\n",response)
    return {
        "model": model,
        "prompt": prompt,
        "response": response.choices[0].message.content
    }

async def run_all(models: list, prompts: list) -> list:
    tasks = [ run_single(m,p) for m in models for p in prompts]
    return await asyncio.gather(*tasks)

if __name__ == '__main__':
    models = ["groq/llama-3.3-70b-versatile", "groq/llama-3.1-8b-instant"]
    prompts = ["What is the capital of France?", "What is 2+2?"]
    res = asyncio.run(run_all(models, prompts))
    print(json.dumps(res, indent=2))