import os
import logging
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import tempfile
import cv2
import re

# OCR
from paddleocr import PaddleOCR

# Mistral OCR (optional)
try:
    from doctr.models import ocr_predictor
    from doctr.io import DocumentFile
    mistral_ocr = ocr_predictor(pretrained=True)
    use_mistral_ocr = True
except ImportError:
    mistral_ocr = None
    use_mistral_ocr = False

# Ensure OCR environment paths
os.environ.setdefault("HOME", "/app")
os.environ.setdefault("PADDLEOCR_HOME", "/app/.paddleocr")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def auto_rotate_image(pil_img):
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    coords = np.column_stack(np.where(img_cv > 0))
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = img_cv.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img_cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_GRAY2RGB))

def extract_images_with_fitz(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images

def extract_text_from_file(file, start_page=None, end_page=None, filename=None):
    ext = os.path.splitext(filename or "")[-1].lower()
    result = []

    if ext == ".pdf":
        doc = fitz.open(file.name)
        images = extract_images_with_fitz(file.name)
        total_pages = len(doc)
        start = max(start_page or 1, 1)
        end = min(end_page or total_pages, total_pages)

        for i, page in enumerate(doc):
            page_num = i + 1
            if not (start <= page_num <= end):
                continue

            text = page.get_text()
            if text.strip():
                result.append(f"Page {page_num} (Extracted):\n{clean_text(text)}")
            else:
                if i < len(images):
                    img = auto_rotate_image(images[i])
                    img_np = np.array(img)
                    try:
                        ocr_result = ocr.ocr(img_np, cls=True)
                        ocr_text = "\n".join([line[1][0] for line in ocr_result[0]]) if ocr_result else ""
                        if not ocr_text and use_mistral_ocr:
                            doc_img = DocumentFile.from_images(img)
                            ocr_text = mistral_ocr(doc_img).render()
                    except Exception as e:
                        logger.warning(f"OCR error on page {page_num}: {e}")
                        ocr_text = "[OCR Error]"
                    result.append(f"Page {page_num} (OCR):\n{clean_text(ocr_text) or '[No OCR Text]'}")
                else:
                    result.append(f"Page {page_num}: [No text or image]")

        doc.close()
        return "\n\n".join(result)

    elif ext == ".docx":
        from docx.api import Document
        doc = Document(file.name)
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        page_texts = []
        page_size = 500
        for i in range(0, len(paras), page_size):
            page_texts.append("\n".join(paras[i:i + page_size]))

        selected_pages = page_texts
        if start_page and end_page:
            selected_pages = page_texts[start_page - 1:end_page]
        return clean_text("\n\n".join(selected_pages))

    elif ext == ".csv":
        import pandas as pd
        return pd.read_csv(file.name).to_string(index=False)

    elif ext in [".xls", ".xlsx"]:
        import pandas as pd
        xl = pd.ExcelFile(file.name)
        return "\n\n".join([
            f"Sheet: {s}\n{xl.parse(s).to_string(index=False)}"
            for s in xl.sheet_names
        ])

    return "Unsupported file type"
