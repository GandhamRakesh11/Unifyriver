import os
import google.generativeai as genai
from PIL import Image
import fitz  # PyMuPDF
from typing import Union
from io import BytesIO

from dotenv import load_dotenv
load_dotenv()


# Configure Gemini API Key (you can load from env or .env)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load Gemini Flash Model
model = genai.GenerativeModel("gemini-2.5-flash")

# Detection prompt
PROMPT = (
    "You are an expert document image analyst. Analyze the given image and identify all visible official stamps, seals, or logos. "
    "Return your analysis strictly in JSON format. Each detected item should be an object in a JSON array with the following fields:\n"
    "- 'position': a brief description of its location, like 'top-left corner', 'bottom center', etc.\n"
    "- 'appearance': a short description of visual features, like 'round blue stamp with Ashoka Chakra'.\n"
    "- 'text': readable text inside or near the stamp/logo/seal (if any). Use an empty string if no text is readable.\n"
    "- 'confidence': your confidence in the detection, as a float between 0 and 1 (e.g., 0.92).\n"
    "- 'bbox': an estimated bounding box of the detected region in the format [x, y, width, height] relative to the image.\n\n"
    "If no stamp, seal, or logo is found, return this exact JSON: {\"status\": \"No stamps or logos found\"}.\n\n"
    "⚠️ Respond only with valid JSON. Do not include explanations or extra text outside the JSON structure."
)

def analyze_stamp(image: Image.Image) -> str:
    try:
        response = model.generate_content([PROMPT, image])
        return response.text.strip()
    except Exception as e:
        return f'{{"error": "Gemini inference failed", "details": "{str(e)}"}}'

def process_pdf_stamp_detection(pdf_path: str) -> dict:
    try:
        doc = fitz.open(pdf_path)
        result = {}
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            result[f"page_{i}"] = analyze_stamp(img)
        doc.close()
        return result
    except Exception as e:
        return {"error": f"PDF processing failed: {str(e)}"}

def process_image_stamp_detection(image_bytes: bytes) -> str:
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        return analyze_stamp(img)
    except Exception as e:
        return f'{{"error": "Image processing failed", "details": "{str(e)}"}}'
