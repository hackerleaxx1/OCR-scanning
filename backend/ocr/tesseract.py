"""
Tesseract OCR Integration for Invoice Processing.

Provides text and spatial extraction from invoice images and PDFs using Tesseract OCR.
Supports both single-page images and multi-page PDF documents.

The OCR output includes:
- Raw text (reconstructed from word positions)
- Word-level bounding boxes with position information
- Per-word confidence scores

This module is the foundation of the extraction pipeline - all subsequent
field extraction depends on the quality of this OCR output.
"""

import pytesseract
from PIL import Image
from pathlib import Path
from config import config
import uuid
import fitz  # PyMuPDF for PDF handling
import io


# =============================================================================
# PDF Conversion Utilities
# =============================================================================

def _convert_pdf_to_images(pdf_path: str) -> list:
    """
    Convert PDF document pages to PIL Images for OCR processing.

    PDFs are rendered at 2x resolution to improve OCR accuracy, especially
    for invoices with small text or complex layouts.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        list: List of PIL Image objects, one per page
    """
    images = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render at 2x resolution for better OCR accuracy
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)
    doc.close()
    return images


# =============================================================================
# OCR Core Functions
# =============================================================================

def extract_text(image_path: str) -> dict:
    """
    Extract text and bounding boxes from an invoice image using Tesseract OCR.

    This is a lower-level function that returns structured OCR data.
    For most use cases, use extract_full_image_data() instead.

    Args:
        image_path: Path to the invoice image file

    Returns:
        dict: OCR results containing:
            - text: Reconstructed text string
            - tokens: List of individual words
            - boxes: List of bounding boxes (left, top, width, height)
            - confidences: Per-word confidence scores
            - avg_confidence: Average confidence across all words
    """
    img = Image.open(image_path)

    # Extract text with bounding box data using Tesseract's JSON output format
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    text_parts = []
    boxes = []
    confidences = []

    # Process each word returned by Tesseract
    for i, text in enumerate(data['text']):
        if text.strip():  # Skip empty text
            text_parts.append(text)
            boxes.append({
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i],
            })
            # Tesseract returns -1 for low-confidence detections; default to 50
            conf = int(data['conf'][i])
            confidences.append(conf if conf > 0 else 50)

    raw_text = ' '.join(text_parts)

    return {
        'text': raw_text,
        'tokens': text_parts,
        'boxes': boxes,
        'confidences': confidences,
        'avg_confidence': sum(confidences) / len(confidences) if confidences else 0
    }


def extract_full_image_data(image_path: str) -> dict:
    """
    Extract structured data from invoice image or PDF document.

    This is the main entry point for OCR processing. It handles both
    images (JPEG, PNG) and PDF documents transparently.

    For PDFs, all pages are processed and words are combined in order.
    The image dimensions are taken from the first page.

    Args:
        image_path: Path to the invoice image or PDF file

    Returns:
        dict: Structured OCR results containing:
            - text: Full text content (all pages combined for PDFs)
            - words: List of word dicts with text and position info
            - image_width: Width of the image/page in pixels
            - image_height: Height of the image/page in pixels
    """
    path = Path(image_path)
    all_words = []
    combined_text = []

    if path.suffix.lower() == '.pdf':
        # Convert PDF to images and process each page
        images = _convert_pdf_to_images(image_path)
        first_width, first_height = None, None
        for i, img in enumerate(images):
            words, text = _extract_words_from_image(img)
            all_words.extend(words)
            combined_text.append(text)
            # Store dimensions from first page for template matching
            if i == 0:
                first_width, first_height = img.size
        combined_text = ' '.join(combined_text)
        return {
            'text': combined_text,
            'words': all_words,
            'image_width': first_width,
            'image_height': first_height
        }
    else:
        # Process single image directly
        img = Image.open(image_path)
        width, height = img.size
        words, combined_text = _extract_words_from_image(img)
        return {
            'text': combined_text,
            'words': words,
            'image_width': width,
            'image_height': height
        }


def _extract_words_from_image(img: Image.Image) -> tuple:
    """
    Extract words with position data from a single PIL Image.

    Uses Tesseract's data output to get per-word bounding boxes and
    confidence scores. This is the core OCR extraction for each page.

    Args:
        img: PIL Image object to process

    Returns:
        tuple: (words list, combined text string)
            - words: List of dicts with text, left, top, width, height, conf
            - combined_text: All words joined into a single string
    """
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    words = []
    text_parts = []
    for i, text in enumerate(data['text']):
        if text.strip():
            words.append({
                'text': text,
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i],
                # Tesseract returns '-1' as string for low confidence detections
                'conf': float(data['conf'][i]) if data['conf'][i] != '-1' else 50.0
            })
            text_parts.append(text)

    return words, ' '.join(text_parts)
