import re
import numpy as np
import cv2
import base64
import requests
from collections import Counter
from requests.auth import HTTPBasicAuth
import logging
from PIL import Image

logger = logging.getLogger(__name__)


def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()


def compute_caption_metrics(text):
    length_score = len(text.split())
    lower = text.lower()

    density_terms = ["axis", "legend", "trend", "outlier", "distribution", "variance", "correlat", "increase", "decrease"]
    density_score = sum(lower.count(term) for term in density_terms)

    words = text.split()
    word_counts = Counter(words)
    redundancy_score = sum(c for w, c in word_counts.items() if c > 1) / max(len(words), 1)

    noun_like = [w for w in words if w[0].isupper()]
    specificity_score = len(set(noun_like)) / max(len(words), 1)

    insight_keywords = ["trend", "increase", "decrease", "anomaly", "correlat", "cluster", "outlier", "distribution"]
    insight_score = sum(lower.count(k) for k in insight_keywords) / max(len(words), 1)

    hallucination_risk = length_score > 100 and "no graph" in lower

    return {
        "length_score": length_score,
        "density_score": density_score,
        "redundancy_score": redundancy_score,
        "specificity_score": specificity_score,
        "insight_score": insight_score,
        "hallucination_risk": hallucination_risk
    }


def upload_to_imagekit(file_path):
    try:
        with open(file_path, 'rb') as f:
            file_data = base64.b64encode(f.read()).decode()

        url = 'https://upload.imagekit.io/api/v1/files/upload'
        payload = {
            'file': file_data,
            'fileName': 'mistral_image.png',
            'useUniqueFileName': 'true'
        }

        PRIVATE_API_KEY = 'private_mDVE6rmy2NS2regbjNNZI9QAJO4='
        response = requests.post(url, data=payload, auth=HTTPBasicAuth(PRIVATE_API_KEY, ''))

        if response.status_code == 200:
            return response.json().get('url')
        else:
            logger.error(f"ImageKit upload failed: {response.text}")
            return None
    except Exception as e:
        logger.error(f"ImageKit upload exception: {e}")
        return None


def is_visually_dense(image, threshold=0.02):
    edges = cv2.Canny(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY), 100, 200)
    return (np.count_nonzero(edges) / edges.size) > threshold


def image_entropy(image, threshold=4.0):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist /= hist.sum()
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    return entropy > threshold
