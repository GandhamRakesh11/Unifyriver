import os
import tempfile
import json
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient
from docx2pdf import convert as docx2pdf_convert

from dotenv import load_dotenv

load_dotenv()

YOLO_MODEL_URL = "https://huggingface.co/Tech4Humans/yolov8s-signature-detector/resolve/main/yolov8s.pt"
YOLO_MODEL_DIR = "/tmp/yolov8s-signature-detector"
YOLO_MODEL_PATH = os.path.join(YOLO_MODEL_DIR, "yolov8s.pt")
ROBOPFLOW_API_KEY = os.getenv("ROBOPFLOW_API_KEY", "Wn6ry91gNXquiPV1WI2B")
ROBOPFLOW_MODEL_ID = "signature-verification-wdakn/1"

os.makedirs(YOLO_MODEL_DIR, exist_ok=True)


def download_yolo_model_if_needed():
    if not os.path.exists(YOLO_MODEL_PATH):
        response = requests.get(YOLO_MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(YOLO_MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
        else:
            raise RuntimeError(f"Failed to download YOLO model. Status code: {response.status_code}")


def convert_docx_to_pdf(docx_path: str) -> str:
    temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    docx2pdf_convert(docx_path, temp_pdf.name)
    return temp_pdf.name


def extract_images_from_pdf(pdf_path):
    pages = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append((page_num + 1, img))
    return pages


def detect_signatures(pil_image, model):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        pil_image.save(f.name)
        results = model(f.name, device="cpu")[0]

    crops, boxes = [], []
    for box in results.boxes:
        xyxy = box.xyxy[0].tolist()
        crop = pil_image.crop((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
        crops.append(crop)
        boxes.append([round(x, 2) for x in xyxy])
    return crops, boxes


def verify_signature(image_path, client):
    try:
        response = client.infer(image_path, model_id=ROBOPFLOW_MODEL_ID)
        predictions = response.get("predictions", [])
        if not predictions:
            return {"result": "No Prediction", "confidence": 0}

        top_pred = max(predictions, key=lambda x: x["confidence"])
        return {
            "result": top_pred.get("class", "Unknown"),
            "confidence": round(top_pred.get("confidence", 0.0) * 100, 2)
        }
    except Exception as e:
        return {"result": "Error", "confidence": 0}


def process_signature_verification(file_path: str) -> dict:
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext == ".docx":
        file_path = Path(convert_docx_to_pdf(str(file_path)))
        ext = ".pdf"

    if ext not in [".pdf", ".jpg", ".jpeg", ".png"]:
        raise ValueError("Unsupported file type")

    pages = (
        [(1, Image.open(file_path))] if ext in [".jpg", ".jpeg", ".png"]
        else extract_images_from_pdf(str(file_path))
    )

    download_yolo_model_if_needed()
    model = YOLO(YOLO_MODEL_PATH)
    client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=ROBOPFLOW_API_KEY)

    results = {"document": file_path.name, "pages": []}

    for page_num, image in pages:
        crops, boxes = detect_signatures(image, model)
        page_result = {"page_number": page_num, "signatures": []}

        for idx, (crop, box) in enumerate(zip(crops, boxes), 1):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                crop.save(f.name)
                prediction = verify_signature(f.name, client)

            page_result["signatures"].append({
                "index": idx,
                "bounding_box": box,
                "verification": prediction
            })

        results["pages"].append(page_result)

    return results
