import os
import tempfile
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List

# === Service Imports ===
from services.extract_text import extract_text_from_file, extract_images_with_fitz
from services.extract_table import extract_tables_from_file
from services.describe_image import extract_captions_from_file
from services.summarizer import Summarizer
from services.document_classifier import DocumentClassifier
from services.ner_extractor import OpenRouterLLM
from services.translator import translate_text
from services.vector_store import get_entry, upsert_entry
from services.masterllm import generate_pipeline
from services.signature_verification import process_signature_verification


from services.s3_utils import upload_to_s3

# === Init ===
api = FastAPI()

summarizer     = Summarizer()
classifier     = DocumentClassifier()
ner_extractor  = OpenRouterLLM()


from pydantic import BaseModel
from fastapi import Body, Depends

class InputParams(BaseModel):
    filename: str
    start_page: int | None = None
    end_page: int | None = None

class TranslateParams(InputParams):
    target_lang: str

def get_input_params(body: InputParams = Body(...)) -> InputParams:
    return body


@api.post("/api/orchestrate")
async def orchestrate_pipeline(
    files: List[UploadFile] = File(...),
    instruction: str = Form(...),
    filename: str = Form(...)
):
    # 1️⃣ Generate pipeline from instruction
    try:
        pipeline_cfg = generate_pipeline(instruction)
    except RuntimeError as err:
        msg = str(err)
        if msg.startswith("PARSE_ERROR"):
            parts = msg.split("RAW:", 1)
            err_msg = parts[0].replace("PARSE_ERROR:", "").strip()
            raw    = parts[1] if len(parts) > 1 else ""
            return JSONResponse(
                status_code=500,
                content={"error": err_msg, "raw_model_output": raw.strip()}
            )
        return JSONResponse(status_code=500, content={"error": msg})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    # 2️⃣ Unpack pipeline config
    pipeline_string = pipeline_cfg.get("pipeline", "").lower()
    start_page      = pipeline_cfg.get("start_page")
    end_page        = pipeline_cfg.get("end_page")
    target_lang     = pipeline_cfg.get("target_lang")

    if not pipeline_string or not filename:
        return JSONResponse(status_code=400, content={"error": "pipeline and filename are required"})

    steps = pipeline_string.split("-")
    print(f"🛠 Pipeline chosen: {pipeline_string}")

    # 3️⃣ Save uploaded files to disk
    temp_files = {}
    for f in files:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(f.file.read())
        tmp.flush()
        temp_files[f.filename] = tmp.name
    
        # ✅ Upload to S3 after saving locally
        upload_to_s3(tmp.name, f"documents/{f.filename}")


    if filename not in temp_files:
        return JSONResponse(status_code=400, content={"error": f"'{filename}' not uploaded"})

    result = {}
    cache  = get_entry(filename) or {}

    # 4️⃣ Execute tasks exactly as returned by the LLM
    with open(temp_files[filename], "rb") as fh:
        for task in steps:
            if task == "text":
                if "text" not in cache:
                    try:
                        cache["text"] = extract_text_from_file(fh, start_page, end_page, filename)
                    except Exception:
                        cache["text"] = ""
                result["text"] = cache["text"]

            elif task == "table":
                if "tables" not in cache:
                    try:
                        cache["tables"] = extract_tables_from_file(fh, start_page, end_page, filename)
                    except Exception:
                        cache["tables"] = []
                result["tables"] = cache["tables"]

            elif task == "describe":
                if "captions" not in cache:
                    try:
                        imgs = extract_images_with_fitz(fh.name)
                        cache["captions"] = extract_captions_from_file(imgs, start_page, end_page)
                    except Exception:
                        cache["captions"] = []
                result["captions"] = cache["captions"]

            elif task == "summarize":
                if "summary" not in cache:
                    try:
                        text = cache.get("text", "")
                        combined = f"--- Extracted Text ---\n{text}\n\n"
                        if "table" in steps and cache.get("tables"):
                            combined += f"--- Tables ---\n{cache['tables']}\n\n"
                        if "describe" in steps and cache.get("captions"):
                            combined += f"--- Captions ---\n{cache['captions']}\n\n"
                        res = summarizer.summarize(combined.strip())
                        summ = ""
                        for m, s in res.get("summaries", {}).items():
                            summ += f"\n=== Summary from {m} ===\n{s.strip()}\n"
                        if res.get("reflection"):
                            summ += f"\n--- Reflection ---\n{res['reflection']}"
                        if res.get("hallucination"):
                            summ += f"\n--- Hallucination Score ---\n{res['hallucination']}"
                        cache["summary"] = summ.strip()
                    except Exception:
                        cache["summary"] = ""
                result["summary"] = cache["summary"]

            elif task == "classify":
                if "classification" not in cache:
                    text = cache.get("text", "")
                    if text.strip():
                        try:
                            cache["classification"] = classifier.classify(text)
                        except Exception:
                            cache["classification"] = {}
                    else:
                        cache["classification"] = {}
                result["classification"] = cache["classification"]

            elif task == "ner":
                if "ner" not in cache:
                    text = cache.get("text", "")
                    if text.strip():
                        try:
                            cache["ner"] = ner_extractor.extract_fields(text)
                        except Exception:
                            cache["ner"] = {}
                    else:
                        cache["ner"] = {}
                result["ner"] = cache["ner"]

            elif task == "translate":
                if "translated" not in cache:
                    text = cache.get("text", "")
                    if text.strip() and target_lang:
                        try:
                            cache["translated"] = translate_text(text, target_lang)
                        except Exception:
                            cache["translated"] = ""
                    else:
                        cache["translated"] = ""
                result["translated"] = cache["translated"]

            else:
                return JSONResponse(status_code=400, content={"error": f"Unknown task: {task}"})

    # 5️⃣ Persist instruction & pipeline config
    cache["instruction"]     = instruction
    cache["pipeline_config"] = pipeline_cfg
    cache.pop("filename", None)
    upsert_entry(filename, **cache)

    # 6️⃣ Cleanup temp files
    for path in temp_files.values():
        os.remove(path)

    # 7️⃣ Return final result
    return {"filename": filename, "result": result}


from fastapi import File, Form, UploadFile
import os
import tempfile

# Save temp file
def save_temp_file(file: UploadFile) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(file.file.read())
    tmp.flush()
    upload_to_s3(tmp.name, f"documents/{file.filename}")
    print(f"📤 Uploaded user file {file.filename} to S3 → documents/{file.filename}")
    return tmp.name


@api.post("/api/text")
async def extract_text_api(
    file: UploadFile = File(...),
    filename: str = Form(...),
    start_page: int = Form(...),
    end_page: int = Form(...)
):
    cache = get_entry(filename) or {}
    if "text" in cache:
        return {"text": cache["text"]}

    path = save_temp_file(file)
    with open(path, "rb") as fh:
        cache["text"] = extract_text_from_file(fh, start_page, end_page, filename)
    os.remove(path)
    cache.pop("filename", None)
    upsert_entry(filename, **cache)
    return {"text": cache["text"]}


@api.post("/api/tables")
async def extract_table_api(
    file: UploadFile = File(...),
    filename: str = Form(...),
    start_page: int = Form(...),
    end_page: int = Form(...)
):
    cache = get_entry(filename) or {}
    if "tables" in cache:
        return {"tables": cache["tables"]}

    path = save_temp_file(file)
    with open(path, "rb") as fh:
        cache["tables"] = extract_tables_from_file(fh, start_page, end_page, filename)
    os.remove(path)
    cache.pop("filename", None)
    upsert_entry(filename, **cache)
    return {"tables": cache["tables"]}


@api.post("/api/describe-images")
async def describe_image_api(
    file: UploadFile = File(...),
    filename: str = Form(...),
    start_page: int = Form(...),
    end_page: int = Form(...)
):
    cache = get_entry(filename) or {}
    if "captions" in cache:
        return {"captions": cache["captions"]}

    path = save_temp_file(file)
    try:
        imgs = extract_images_with_fitz(path)
        cache["captions"] = extract_captions_from_file(imgs, start_page, end_page)
    except Exception:
        cache["captions"] = []
    os.remove(path)
    cache.pop("filename", None)
    upsert_entry(filename, **cache)
    return {"captions": cache["captions"]}


@api.post("/api/summarize")
async def summarize_api(
    file: UploadFile = File(...),
    filename: str = Form(...),
    start_page: int = Form(...),
    end_page: int = Form(...)
):
    cache = get_entry(filename) or {}
    if "summary" in cache:
        return {"summary": cache["summary"]}

    path = save_temp_file(file)
    if "text" not in cache:
        with open(path, "rb") as fh:
            cache["text"] = extract_text_from_file(fh, start_page, end_page, filename)
    if "tables" not in cache:
        with open(path, "rb") as fh:
            cache["tables"] = extract_tables_from_file(fh, start_page, end_page, filename)
    if "captions" not in cache:
        imgs = extract_images_with_fitz(path)
        cache["captions"] = extract_captions_from_file(imgs, start_page, end_page)
    os.remove(path)

    combined = f"--- Extracted Text ---\n{cache.get('text', '')}\n\n"
    combined += f"--- Tables ---\n{cache.get('tables', '')}\n\n"
    combined += f"--- Captions ---\n{cache.get('captions', '')}\n\n"

    res = summarizer.summarize(combined.strip())
    summary = ""
    for m, s in res.get("summaries", {}).items():
        summary += f"\n=== Summary from {m} ===\n{s.strip()}\n"
    if res.get("reflection"):
        summary += f"\n--- Reflection ---\n{res['reflection']}"
    if res.get("hallucination"):
        summary += f"\n--- Hallucination Score ---\n{res['hallucination']}"

    cache["summary"] = summary.strip()
    cache.pop("filename", None)
    upsert_entry(filename, **cache)
    return {"summary": cache["summary"]}


@api.post("/api/classify")
async def classify_api(
    file: UploadFile = File(...),
    filename: str = Form(...),
    start_page: int = Form(...),
    end_page: int = Form(...),
):
    cache = get_entry(filename) or {}
    if "text" not in cache:
        path = save_temp_file(file)
        with open(path, "rb") as fh:
            cache["text"] = extract_text_from_file(fh, start_page, end_page, filename)
        os.remove(path)
    
    text = cache["text"]
    classification = classifier.classify(text) if text.strip() else {}
    cache["classification"] = classification
    cache.pop("filename", None)
    upsert_entry(filename, **cache)
    return {"classification": classification}



@api.post("/api/ner")
async def ner_api(
    file: UploadFile = File(...),
    filename: str = Form(...),
    start_page: int = Form(...),
    end_page: int = Form(...)
):
    cache = get_entry(filename) or {}
    if "ner" in cache:
        return {"ner": cache["ner"]}

    path = save_temp_file(file)
    if "text" not in cache:
        with open(path, "rb") as fh:
            cache["text"] = extract_text_from_file(fh, start_page, end_page, filename)
    os.remove(path)

    cache["ner"] = ner_extractor.extract_fields(cache["text"]) if cache["text"].strip() else {}
    cache.pop("filename", None)
    upsert_entry(filename, **cache)
    return {"ner": cache["ner"]}


@api.post("/api/translate")
async def translate_api(
    file: UploadFile = File(...),
    filename: str = Form(...),
    start_page: int = Form(...),
    end_page: int = Form(...),
    target_lang: str = Form(...)
):
    cache = get_entry(filename) or {}
    if "translated" in cache:
        return {"translated": cache["translated"]}

    path = save_temp_file(file)
    if "text" not in cache:
        with open(path, "rb") as fh:
            cache["text"] = extract_text_from_file(fh, start_page, end_page, filename)
    os.remove(path)

    cache["translated"] = translate_text(cache["text"], target_lang) if cache["text"].strip() else ""
    cache.pop("filename", None)
    upsert_entry(filename, **cache)
    return {"translated": cache["translated"]}
    

@api.post("/api/signature-verification")
async def signature_verification_api(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        tmp.write(file.file.read())
        tmp.flush()
        tmp_path = tmp.name

    try:
        result = process_signature_verification(tmp_path)
    except Exception as e:
        os.remove(tmp_path)
        return JSONResponse(status_code=500, content={"error": str(e)})

    os.remove(tmp_path)
    return result

@api.post("/api/stamp-detection")
async def stamp_detection_api(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file.file.read())
        tmp.flush()
        tmp_path = tmp.name

    try:
        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            from services.stamp_detection import process_image_stamp_detection
            with open(tmp_path, "rb") as f:
                result = process_image_stamp_detection(f.read())
        elif ext == ".pdf":
            from services.stamp_detection import process_pdf_stamp_detection
            result = process_pdf_stamp_detection(tmp_path)
        else:
            os.remove(tmp_path)
            return JSONResponse(status_code=400, content={"error": "Unsupported file type. Use image or PDF."})
    except Exception as e:
        os.remove(tmp_path)
        return JSONResponse(status_code=500, content={"error": str(e)})

    os.remove(tmp_path)
    return {"filename": file.filename, "stamp_detection_result": result}




# Dev run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:api", host="0.0.0.0", port=3000, reload=True)



