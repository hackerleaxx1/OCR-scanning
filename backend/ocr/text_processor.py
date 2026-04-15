"""
Text Processing Utilities for Invoice OCR.

Provides text normalization, TF-IDF vectorization, and pattern extraction
functions used throughout the ML and extraction pipeline.

Key functions:
- normalize_text: Standardize OCR text for feature extraction
- build_vocabulary: Create TF-IDF vectorizer from training texts
- load_vocabulary: Load pre-built vocabulary for inference
- text_to_features: Convert text to TF-IDF feature vector
- extract_numeric/date/invoice_number: Extract specific field values via regex
"""

import re
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# =============================================================================
# Text Normalization
# =============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize OCR text for better feature extraction.

    Steps:
    1. Convert to lowercase
    2. Remove punctuation (keep word characters and spaces)
    3. Collapse multiple spaces to single space
    4. Strip leading/trailing whitespace

    Args:
        text: Raw OCR text

    Returns:
        Normalized text string
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)       # Collapse whitespace
    return text.strip()


def tokenize(text: str) -> list:
    """
    Simple whitespace tokenization.

    Args:
        text: Input text

    Returns:
        List of tokens (words)
    """
    return text.split()


# =============================================================================
# TF-IDF Vocabulary Management
# =============================================================================

def build_vocabulary(texts: list, vendor_id: str, vocab_dir: Path, max_features: int = 500):
    """
    Build TF-IDF vocabulary from training texts for a vendor.

    Each vendor has its own vocabulary since invoice formats vary by vendor.
    The vocabulary maps terms to feature indices for the neural network.

    Args:
        texts: List of OCR text strings from validated invoices
        vendor_id: Vendor identifier for namespacing
        vocab_dir: Path to vendor's directory
        max_features: Maximum vocabulary size (default 500)

    Returns:
        Fitted TfidfVectorizer
    """
    normalized = [normalize_text(t) for t in texts if t and normalize_text(t)]

    # Ensure we have at least one text for vocabulary building
    if len(normalized) < 1:
        normalized = ['invoice placeholder text']

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),              # Unigrams and bigrams for context
        stop_words=None,                  # Don't remove stop words - they may be informative
        min_df=1,                        # Include all terms (vendor vocabularies are small)
        max_df=1.0,                      # Allow all terms
        token_pattern=r'(?u)\b\w+\b'      # Include single-char tokens
    )

    try:
        vectorizer.fit(normalized)
    except ValueError:
        # Fallback: use character n-grams if word tokenization fails
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(2, 4),
            analyzer='char_wb',           # Character n-grams within word boundaries
            min_df=1,
            max_df=1.0
        )
        vectorizer.fit(normalized)

    # Persist vocabulary for inference
    vocab_data = {
        'vocabulary': {k: int(v) for k, v in vectorizer.vocabulary_.items()},
        'idf': [float(x) for x in vectorizer.idf_.tolist()],
        'max_features': max_features
    }

    vocab_path = vocab_dir / "vocabulary.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, 'w') as f:
        json.dump(vocab_data, f)

    return vectorizer


def load_vocabulary(vendor_id: str, vendor_dir: Path) -> TfidfVectorizer or None:
    """
    Load existing vocabulary or return None if doesn't exist.

    Args:
        vendor_id: Vendor identifier
        vendor_dir: Path to vendor's directory

    Returns:
        Loaded TfidfVectorizer, or None if no vocabulary file exists
    """
    vocab_path = vendor_dir / "vocabulary.json"

    if not vocab_path.exists():
        return None

    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)

    vectorizer = TfidfVectorizer(
        max_features=vocab_data['max_features'],
        ngram_range=(1, 2),
        stop_words='english'
    )

    # Restore vocabulary state
    vocab = vocab_data['vocabulary']
    idf = vocab_data['idf']
    vectorizer.vocabulary_ = vocab
    vectorizer.idf_ = np.array(idf)

    return vectorizer


def text_to_features(text: str, vectorizer: TfidfVectorizer) -> np.ndarray:
    """
    Convert raw text to TF-IDF feature vector.

    Args:
        text: Raw OCR text
        vectorizer: Fitted TfidfVectorizer

    Returns:
        Feature vector of shape (max_features,)
    """
    normalized = normalize_text(text)
    features = vectorizer.transform([normalized])
    return features.toarray()[0]


# =============================================================================
# Field Value Extraction via Regex
# =============================================================================

def extract_numeric(text: str, pattern: str) -> float or None:
    """
    Extract a numeric value from text using regex pattern.

    Args:
        text: Text to search
        pattern: Regex pattern with one capturing group for the number

    Returns:
        Float value, or None if pattern doesn't match
    """
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group().replace(',', ''))
        except ValueError:
            pass
    return None


def extract_date(text: str) -> str or None:
    """
    Extract date from OCR text.

    Supports multiple date formats:
    - YYYY-MM-DD (ISO format)
    - MM/DD/YYYY (US format)
    - MM-DD-YYYY
    - DD Mon YYYY (e.g., "15 Jan 2024")

    Args:
        text: Text to search for date

    Returns:
        Date string in the format it was found, or None if no date found
    """
    patterns = [
        r'\d{4}-\d{2}-\d{2}',           # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',           # MM/DD/YYYY
        r'\d{2}-\d{2}-\d{4}',           # MM-DD-YYYY
        r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}',  # DD Mon YYYY
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group()

    return None


def extract_invoice_number(text: str) -> str or None:
    """
    Extract invoice number from OCR text.

    Looks for common invoice number patterns:
    - "Invoice No.: INV-12345"
    - "Inv #: 12345"
    - Standalone INV- prefixes

    Args:
        text: Text to search

    Returns:
        Invoice number string, or None if not found
    """
    patterns = [
        r'(?:invoice|inv|inv\.?)\s*(?:no\.?|number|#)?\s*[:.]?\s*([A-Z0-9\-]+)',
        r'(?:invoice|inv)\s*[:.]?\s*([A-Z0-9\-]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None
