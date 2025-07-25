import re
import unicodedata

LANGUAGE_MAP = {
    "english": "en_XX", "hindi": "hi_IN", "french": "fr_XX",
    "spanish": "es_XX", "german": "de_DE", "japanese": "ja_XX",
    "korean": "ko_KR", "arabic": "ar_AR", "russian": "ru_RU",
    "portuguese": "pt_XX", "italian": "it_IT", "chinese": "zh_CN"
}

HALLUCINATION_THRESHOLD = 0.65


def remove_non_printable(text):
    return ''.join(c for c in text if unicodedata.category(c)[0] != "C")


def filter_ocr_noise(text):
    lines = text.splitlines()
    return '\n'.join(
        line for line in lines if sum(c.isalpha() for c in line) > 0.5 * len(line)
    )


class DocumentProcessor:
    def clean_text(self, text):
        text = remove_non_printable(text)
        text = filter_ocr_noise(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def chunk_text(self, text, words_per_chunk=200):
        paragraphs = re.split(r'\n\s*\n', text)
        if len(paragraphs) <= 1:
            paragraphs = re.split(r'(?<=[.?!])\s+', text)

        chunks, current_chunk, current_word_count = [], [], 0
        for para in paragraphs:
            words = para.split()
            if current_word_count + len(words) > words_per_chunk and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [para]
                current_word_count = len(words)
            else:
                current_chunk.append(para)
                current_word_count += len(words)
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
