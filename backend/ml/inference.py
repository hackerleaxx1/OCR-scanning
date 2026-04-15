"""
KNN-based Invoice Field Prediction.

A lightweight inference approach using K-Nearest Neighbors (KNN) with
cosine similarity on TF-IDF features. This approach works well with
small datasets (even 1-5 samples) where training a neural network would overfit.

For each new invoice:
1. Convert OCR text to TF-IDF features
2. Find most similar validated invoices (nearest neighbors)
3. Use the best match's field values as predictions
4. For amounts, compute similarity-weighted average

This is a fallback method used when:
- No position template exists (generic extraction)
- Neural network isn't trained yet
- Confidence from other methods is low
"""

import numpy as np
from pathlib import Path
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ocr.text_processor import normalize_text, extract_date, extract_invoice_number, extract_numeric


class InvoicePredictor:
    """
    KNN-based invoice field predictor.

    Uses validated invoices as reference points. For new invoices:
    - Finds similar invoices by OCR text content
    - Returns best match's field values as predictions
    - Computes weighted averages for numeric fields

    Advantages:
    - No explicit training needed
    - Works with small datasets (even 1-5 samples)
    - Vendor-specific (each vendor has separate training data)
    - Supports multiple line items

    The K in KNN is自适应 (adaptive) - uses k=min(3, n_samples).
    """

    def __init__(self, vendor_id: str, vendor_dir: Path):
        """
        Initialize predictor for a vendor.

        Args:
            vendor_id: Unique vendor identifier
            vendor_dir: Path to vendor's directory containing training_data.csv
        """
        self.vendor_id = vendor_id
        self.vendor_dir = Path(vendor_dir)
        self.df = None
        self.vectorizer = None
        self.X_train = None
        self._load_training_data()

    def _load_training_data(self):
        """
        Load validated invoices and build TF-IDF features.

        If training_data.csv exists, loads it and builds the vectorizer.
        The vectorizer learns vocabulary from all training texts to enable
        comparison of new invoices against stored ones.
        """
        csv_path = self.vendor_dir / "training_data.csv"
        if not csv_path.exists():
            return

        self.df = pd.read_csv(csv_path)
        if len(self.df) == 0:
            self.df = None
            return

        # Build vocabulary from all training texts
        texts = self.df['ocr_text'].tolist()
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words=None,
            min_df=1,
            max_df=1.0
        )
        self.X_train = self.vectorizer.fit_transform([normalize_text(t) for t in texts])

    def predict(self, ocr_text: str) -> dict:
        """
        Predict invoice fields including multiple line items.

        Uses cosine similarity to find the most similar validated invoice(s),
        then returns those values as predictions. For numeric fields like
        invoice_amount, computes a similarity-weighted average.

        Args:
            ocr_text: Raw OCR text from the invoice image

        Returns:
            Dict with:
            - predictions: Field values (invoice_number, date, amount, items)
            - confidences: Per-field confidence scores (0-1)
        """
        if self.df is None or len(self.df) == 0:
            return self._fallback_predictions(ocr_text)

        # Compute TF-IDF features for input text
        norm_text = normalize_text(ocr_text)
        X_test = self.vectorizer.transform([norm_text])

        # Find most similar training sample(s) using cosine similarity
        similarities = cosine_similarity(X_test, self.X_train)[0]

        # Use weighted KNN with k=min(3, n_samples)
        k = min(3, len(self.df))
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        top_k_sims = similarities[top_k_idx]

        predictions = {}
        confidences = {}

        # Get the most similar invoice's header fields
        best_idx = top_k_idx[0]
        best_sim = top_k_sims[0]

        # Header fields from best match
        header_fields = ['invoice_number', 'invoice_date', 'invoice_amount']
        for field in header_fields:
            if field in self.df.columns:
                val = self.df[field].iloc[best_idx]
                if pd.notna(val):
                    # Convert numpy types to native Python types for JSON serialization
                    if isinstance(val, (np.integer, np.int64, np.int32)):
                        val = int(val)
                    elif isinstance(val, (np.floating, np.float64, np.float32)):
                        val = float(val)
                    else:
                        val = str(val)
                else:
                    val = ''
                predictions[field] = val

        # For invoice_amount, compute similarity-weighted average
        # More similar invoices have more influence on the predicted amount
        if 'invoice_amount' in self.df.columns:
            amounts = self.df['invoice_amount'].iloc[top_k_idx].tolist()
            weighted_sum = sum(a * s for a, s in zip(amounts, top_k_sims) if pd.notna(a))
            weight_sum = sum(s for a, s in zip(amounts, top_k_sims) if pd.notna(a))
            if weight_sum > 0:
                predictions['invoice_amount'] = round(weighted_sum / weight_sum, 2)

        # Set confidences based on similarity score
        # Higher similarity = higher confidence
        for field in header_fields:
            confidences[field] = float(best_sim)

        # Items - use the best matching invoice's items as template
        items = []
        if 'items_json' in self.df.columns:
            items_json = self.df['items_json'].iloc[best_idx]
            if pd.notna(items_json) and items_json != '[]':
                try:
                    items = json.loads(items_json)
                except:
                    items = []
        # Fallback: check old schema with separate item columns
        if not items and 'item_name' in self.df.columns:
            item_name = self.df['item_name'].iloc[best_idx]
            if pd.notna(item_name) and str(item_name).strip() and str(item_name) != 'nan':
                items = [{
                    'item_no': int(self.df['item_no'].iloc[best_idx]) if 'item_no' in self.df.columns and pd.notna(self.df['item_no'].iloc[best_idx]) else None,
                    'item_name': str(item_name),
                    'item_quantity': int(self.df['item_quantity'].iloc[best_idx]) if pd.notna(self.df['item_quantity'].iloc[best_idx]) else 1,
                    'per_item_price': float(self.df['per_item_price'].iloc[best_idx]) if pd.notna(self.df['per_item_price'].iloc[best_idx]) else 0,
                    'total_item_price': float(self.df['total_item_price'].iloc[best_idx]) if pd.notna(self.df['total_item_price'].iloc[best_idx]) else 0
                }]

        # If no items found, create a default item using invoice amount
        if not items:
            items = [{
                'item_no': None,
                'item_name': 'Unknown Item',
                'item_quantity': 1,
                'per_item_price': predictions.get('invoice_amount', 0) or 0,
                'total_item_price': predictions.get('invoice_amount', 0) or 0
            }]

        predictions['items'] = items

        # Format confidences for items (same confidence for all item fields)
        item_confidences = []
        for item in items:
            item_conf = {
                'item_no': float(best_sim),
                'item_name': float(best_sim),
                'item_quantity': float(best_sim),
                'per_item_price': float(best_sim),
                'total_item_price': float(best_sim)
            }
            item_confidences.append(item_conf)

        confidences['items'] = item_confidences

        return {
            'predictions': predictions,
            'confidences': confidences
        }

    def _fallback_predictions(self, ocr_text: str) -> dict:
        """
        Rule-based fallback when no training data exists.

        Uses regex patterns to extract fields when there are no validated
        invoices to learn from. This is less accurate but provides
        reasonable predictions for new vendors.

        Args:
            ocr_text: Raw OCR text

        Returns:
            Dict with predictions and confidences (lower confidence than KNN)
        """
        invoice_number = extract_invoice_number(ocr_text)
        invoice_date = extract_date(ocr_text)

        # Find largest amount - typically the total
        amount_patterns = [
            r'(?:total|amount|sum|grand\s*total)[:.\s]*[$]?\s*([\d,]+\.?\d*)',
            r'[$]\s*([\d,]+\.?\d*)',
        ]
        invoice_amount = None
        for pattern in amount_patterns:
            val = extract_numeric(ocr_text, pattern)
            if val and (invoice_amount is None or val > invoice_amount):
                invoice_amount = val

        predictions = {
            'invoice_number': invoice_number or 'Not found',
            'invoice_date': invoice_date or 'Not found',
            'invoice_amount': invoice_amount or 0.0,
            'items': [{
                'item_no': None,
                'item_name': 'Unknown Item',
                'item_quantity': 1,
                'per_item_price': invoice_amount or 0.0,
                'total_item_price': invoice_amount or 0.0
            }]
        }

        confidences = {
            'invoice_number': 0.5,
            'invoice_date': 0.5,
            'invoice_amount': 0.4,
            'items': [{
                'item_name': 0.3,
                'item_quantity': 0.4,
                'per_item_price': 0.4,
                'total_item_price': 0.4
            }]
        }

        return {
            'predictions': predictions,
            'confidences': confidences
        }
