import requests

# === BASE CONFIGURATION ===
API_BASE = "https://redhairedshanks1-docproc1.hf.space/api"  # Replace with your deployed URL or localhost
PDF_FILE_PATH = r"E:\downloads\5_Fuzzing.pdf"  # Update to your local file path
FILENAME = "sample1.pdf"
START_PAGE = 1
END_PAGE = 1
TARGET_LANG = "french"

# === COMMON FORM DATA ===
form_data = {
    "filename": FILENAME,
    "start_page": str(START_PAGE),
    "end_page": str(END_PAGE)
}


def post_file(endpoint, extra_data=None, send_filename=True, override_file=None):
    url = f"{API_BASE}/{endpoint}"
    data = form_data.copy() if send_filename else {}
    if extra_data:
        data.update(extra_data)

    filepath = override_file or PDF_FILE_PATH
    print(f"\n➡️ Sending request to: {url}")
    print(f"📄 Form data: {data}")
    print(f"📁 File: {filepath}")

    with open(filepath, "rb") as file:
        response = requests.post(
            url,
            files={"file": (FILENAME, file, "application/pdf")},
            data=data
        )

    print(f"\n🔹 Endpoint: {endpoint} | Status: {response.status_code}")
    try:
        print("📦 Response JSON:\n", response.json())
    except Exception:
        print("⚠️ Response is not JSON:\n", response.text)


def test_text():
    post_file("text")


def test_tables():
    post_file("tables")


def test_describe_images():
    post_file("describe-images")


def test_summarize():
    post_file("summarize")


def test_ner():
    post_file("ner")


def test_translate():
    post_file("translate", extra_data={"target_lang": TARGET_LANG})


def test_signature_verification():
    # Signature API does not need start_page, end_page, or filename
    post_file("signature-verification", send_filename=False)


def test_stamp_detection():
    # Stamp API auto-detects file type (PDF/image)
    post_file("stamp-detection")


# === Run selected tests ===
if __name__ == "__main__":
    print("🚀 Starting individual API tests...")

    test_text()
    # test_tables()
    # test_describe_images()
    # test_summarize()
    # test_ner()
    # test_translate()
    # test_signature_verification()
    # test_stamp_detection()
