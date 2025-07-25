import os
import requests
from dotenv import load_dotenv

from utilities.ner_extractor_utils import compute_caption_metrics, chunk_text, load_prompt

# Load environment variables
load_dotenv()

class OpenRouterLLM:
    def __init__(self, api_key=None, model_id=None):
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key or os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "UniversalNERExtractor"
        }
        self.model = model_id or "mistralai/mixtral-8x7b-instruct"

    def extract_fields(self, text, max_chars=80000):
        chunks = chunk_text(text, max_chars)
        all_results = []

        for i, chunk in enumerate(chunks):
            prompt_template = load_prompt("ner_prompt.txt")
            prompt = prompt_template.replace("{chunk}", chunk.strip())

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful document parser."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }

            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"].strip()
                metrics = compute_caption_metrics(content)

                all_results.append(
                    f"--- Chunk {i+1} ---\n"
                    f"{content}\n\n"
                    f"[Chunk {i+1} Metrics]\n{metrics}"
                )
            except requests.exceptions.HTTPError as e:
                print(f"⚠️ Chunk {i+1} HTTP error:", e.response.status_code)
                print("⚠️ Response content:", e.response.text)
                all_results.append(f"[NER Error Chunk {i+1}] {e.response.text}")
            except Exception as e:
                print(f"⚠️ Chunk {i+1} request exception:", str(e))
                all_results.append(f"[NER Error Chunk {i+1}] {str(e)}")

        return "\n\n".join(all_results)
