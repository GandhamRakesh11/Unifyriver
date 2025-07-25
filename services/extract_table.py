import os
import fitz  # PyMuPDF (not used in this script, but often for PDF handling)
import pdfplumber  # For extracting tables from PDFs
import pandas as pd  # For handling tabular data (CSV, Excel)
from docx.api import Document  # For reading DOCX documents
import logging

# Setup logging to ensure messages are visible during standalone use
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_tables_from_file(file, start_page=None, end_page=None, filename=None):
    """
    Extracts tables from a document, depending on its file extension.

    Supports PDF, DOCX, CSV, XLS/XLSX formats.

    Args:
        file: File-like object.
        start_page (int, optional): Start page for partial PDF parsing.
        end_page (int, optional): End page for partial PDF parsing.
        filename (str, optional): Filename used to determine file extension.

    Returns:
        str: All extracted tables formatted as a single string.
    """
    ext = os.path.splitext(filename or "")[-1].lower()
    tables = []

    # ------------------ PDF (.pdf) Extraction ------------------ #
    if ext == ".pdf":
        try:
            with pdfplumber.open(file.name) as pdf:
                total_pages = len(pdf.pages)
                start = max(start_page or 1, 1)
                end = min(end_page or total_pages, total_pages)

                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    if not (start <= page_num <= end):
                        continue
                    try:
                        for table in page.extract_tables():
                            # Join each row into a string with pipe-separated columns
                            rows = [" | ".join(cell or "" for cell in row) for row in table if row]
                            tables.append(f"Page {page_num} Table:\n" + "\n".join(rows))
                    except Exception as e:
                        logger.warning(f"PDF table extraction failed on page {page_num}: {e}")
        except Exception as e:
            logger.error(f"Failed to read PDF file: {e}")

    # ------------------ DOCX (.docx) Extraction ------------------ #
    elif ext == ".docx":
        try:
            doc = Document(file.name)
            for t in doc.tables:
                # Extract text from each table row
                rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in t.rows]
                tables.append("\n".join(rows))
        except Exception as e:
            logger.error(f"DOCX table extraction failed: {e}")

    # ------------------ CSV (.csv) Extraction ------------------ #
    elif ext == ".csv":
        try:
            df = pd.read_csv(file.name)
            tables.append(df.to_string(index=False))  # Convert DataFrame to string without index
        except Exception as e:
            logger.warning(f"CSV parsing error: {e}")

    # ------------------ Excel (.xls, .xlsx) Extraction ------------------ #
    elif ext in [".xls", ".xlsx"]:
        try:
            xl = pd.ExcelFile(file.name)
            for s in xl.sheet_names:
                sheet_df = xl.parse(s)
                tables.append(f"Sheet: {s}\n{sheet_df.to_string(index=False)}")
        except Exception as e:
            logger.warning(f"Excel parsing error: {e}")

    # ------------------ Unsupported File Type ------------------ #
    else:
        logger.warning(f"Unsupported file type: {ext}")

    # Join all extracted tables into a single string separated by newlines
    return "\n\n".join(tables)
