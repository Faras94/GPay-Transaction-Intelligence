"""
RAG LLM Module

Handles external LLM API calls.
"""

import requests
from .rag_config import OPENROUTER_API_KEY, LLM_MODEL


def call_llm(prompt: str) -> str:
    """
    Call OpenRouter API with the given prompt.
    
    Args:
        prompt: System + user prompt
        
    Returns:
        str: LLM response text
    """
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not configured."
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error calling LLM: {str(e)}"
