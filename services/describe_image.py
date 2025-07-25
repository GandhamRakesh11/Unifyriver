import os
import tempfile
import logging
from google.generativeai import upload_file, GenerativeModel, configure
from mistralai import Mistral

from dotenv import load_dotenv

load_dotenv()

from utilities.image_utils import (
    clean_text,
    compute_caption_metrics,
    upload_to_imagekit,
    is_visually_dense,
    image_entropy
)
from prompts.image_caption_prompt import gemini_prompt

logger = logging.getLogger(__name__)


def describe_image(img, page_number):
    gemini_text = "[Gemini not executed]"
    gemini_metrics = {}
    mistral_text = "[Mistral response unavailable]"
    mistral_metrics = {}

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp.name)
            local_path = tmp.name

        try:
            configure(api_key=os.getenv("GOOGLE_API_KEY"))
            gemini_model = GenerativeModel("gemini-1.5-flash")
            uploaded = upload_file(local_path, mime_type="image/png")
            gemini_response = gemini_model.generate_content([uploaded, gemini_prompt])
            gemini_text = clean_text(gemini_response.text)
            gemini_metrics = compute_caption_metrics(gemini_text)
        except Exception as ge:
            logger.error(f"[Page {page_number}] Gemini error: {ge}")
            gemini_text = f"[Gemini Error: {ge}]"

        mistral_key = os.getenv("MISTRAL_API_KEY")
        if mistral_key:
            try:
                mistral_client = Mistral(api_key=mistral_key)
                image_url = upload_to_imagekit(local_path)
                if not image_url:
                    mistral_text = "[Image upload failed — invalid URL]"
                else:
                    mistral_response = mistral_client.chat.complete(
                        model="pixtral-12b-latest",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": gemini_prompt},
                                    {"type": "image_url", "image_url": image_url}
                                ]
                            }
                        ]
                    )
                    mistral_text = mistral_response.choices[0].message.content.strip()
                    mistral_metrics = compute_caption_metrics(mistral_text)
            except Exception as me:
                logger.error(f"[Page {page_number}] Mistral error: {me}")
                mistral_text = f"[Mistral Error: {me}]"
        else:
            mistral_text = "[Mistral API key not set]"

        os.remove(local_path)

        return format_caption_output(
            gemini_text, gemini_metrics, mistral_text, mistral_metrics
        )

    except Exception as e:
        logger.error(f"[Page {page_number}] Caption error: {e}")
        return f"[Caption Error: {e}]"


def format_caption_output(gemini_text, gemini_metrics, mistral_text, mistral_metrics):
    return (
        f"======================= Gemini Analysis =======================\n"
        f"{gemini_text}\n"
        f"\n--- Gemini Metrics ---\n{gemini_metrics}\n"
        f"\n======================= Mistral Analysis ======================\n"
        f"{mistral_text}\n"
        f"\n--- Mistral Metrics ---\n{mistral_metrics}\n"
        f"===============================================================\n"
    )


def extract_captions_from_file(images, doc=None, start_page=1, end_page=None):
    captions = []
    total_pages = len(images)
    end = min(end_page or total_pages, total_pages)

    for i in range(len(images)):
        page_number = i + 1
        if not (start_page <= page_number <= end):
            continue

        img = images[i]
        if (not doc or doc[page_number - 1].get_images(full=True)
            or is_visually_dense(img)
            or image_entropy(img)
            or len(doc[page_number - 1].get_text().strip()) < 30):

            caption = describe_image(img, page_number)
            captions.append(f"Page {page_number}:\n{caption}")

    return "\n\n".join(captions)
