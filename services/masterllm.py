# services/masterllm.py
import json
import textwrap
import requests
import os
import re

from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") or "IK7RgLmrXAFDGbDJBO76SoLSHux1UeNL"
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL   = "mistral-small"

# Steps we support
ALLOWED_STEPS = {"text", "table", "describe", "summarize", "ner", "classify", "translate"}

def build_prompt(instruction: str) -> str:
    return f"""You are a document‐processing assistant.
Return exactly **one** JSON object and nothing else—no markdown, no explanation, no extra keys.
Use **only** the steps the user asks for in the instruction.  
Do not add any steps not mentioned.
Valid steps (dash‑separated): {', '.join(sorted(ALLOWED_STEPS))}
Output schema:
{{
  "pipeline": "<dash‑separated‑steps>",
  "tools": {{ /* tool names or null */ }},
  "start_page": <int>,
  "end_page": <int>,
  "target_lang": <string or null>
}}
Instruction:
\"\"\"{instruction.strip()}\"\"\"
"""

def extract_json_block(text: str) -> dict:
    # grab everything between the first { and last }
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        return {"error": "no JSON braces found", "raw": text}
    snippet = text[start:end+1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError as e:
        # attempt to fix common "tools": {null} → "tools": {}
        cleaned = re.sub(r'"tools"\s*:\s*\{null\}', '"tools": {}', snippet)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"error": f"json decode error: {e}", "raw": snippet}

def validate_pipeline(cfg: dict) -> dict:
    pipe = cfg.get("pipeline")
    if isinstance(pipe, list):
        pipe = "-".join(pipe)
        cfg["pipeline"] = pipe
    if not isinstance(pipe, str):
        return {"error": "pipeline must be a string"}
    steps = pipe.split("-")
    bad   = [s for s in steps if s not in ALLOWED_STEPS]
    if bad:
        return {"error": f"invalid steps: {bad}"}
    if "translate" in steps and not cfg.get("target_lang"):
        return {"error": "target_lang required for translate"}
    return {"ok": True}

def generate_pipeline(instruction: str) -> dict:
    prompt = build_prompt(instruction)
    res = requests.post(
        MISTRAL_ENDPOINT,
        headers={
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MISTRAL_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 256
        }
    )
    res.raise_for_status()
    content = res.json()["choices"][0]["message"]["content"]

    parsed = extract_json_block(content)
    if "error" in parsed:
        raise RuntimeError(f"PARSE_ERROR: {parsed['error']}\nRAW_OUTPUT:\n{parsed.get('raw', content)}")

    # Normalize pipeline separators (commas, spaces → dashes)
    raw_pipe = parsed.get("pipeline", "")
    steps = [s.strip() for s in re.split(r"[,\s]+", raw_pipe) if s.strip()]
    parsed["pipeline"] = "-".join(steps)

    check = validate_pipeline(parsed)
    if "error" in check:
        raise RuntimeError(f"PARSE_ERROR: {check['error']}\nRAW_OUTPUT:\n{content}")

    return parsed



