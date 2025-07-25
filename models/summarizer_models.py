import os
import requests
from dotenv import load_dotenv

load_dotenv()

# API keys
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
OR_KEY = os.getenv("OPENROUTER_API_KEY")

# HuggingFace models and config
HF_MODELS = {
    "bart_cnn": "facebook/bart-large-cnn"
}
HF_CHUNK_LIMITS = {
    "facebook/bart-large-cnn": 350
}
HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# OpenRouter models and config
OR_MODELS = {
    "llama3": "meta-llama/llama-3-8b-instruct"
}
OR_CHUNK_LIMITS = {
    "meta-llama/llama-3-8b-instruct": 2500
}
OR_HEADERS = {
    "Authorization": f"Bearer {OR_KEY}",
    "Content-Type": "application/json"
}


def call_hf_model(model_id, text):
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    word_count = len(text.split())
    min_length = max(30, int(word_count * 0.2))
    max_length = max(60, int(word_count * 0.4))

    payload = {
        "inputs": text,
        "parameters": {
            "min_length": min_length,
            "max_length": max_length,
            "do_sample": False
        }
    }

    try:
        response = requests.post(url, headers=HF_HEADERS, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and "summary_text" in result[0]:
            return result[0]["summary_text"]
        return "[HF Error] Unexpected response format"
    except Exception as e:
        return f"[HF Error: {model_id}] {str(e)}"


def call_or_model(model_id, text):
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Summarize the following text accurately and concisely."},
            {"role": "user", "content": text}
        ],
        "max_tokens": 400,
        "temperature": 0.4
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=OR_HEADERS,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[OpenRouter Error: {model_id}] {str(e)}"


def call_openrouter(prompt, model_id, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Answer the question clearly."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[OpenRouter Error: {model_id}] {str(e)}"
