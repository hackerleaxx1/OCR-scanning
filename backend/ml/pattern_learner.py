"""
Pattern-based Field Extraction Fallback System.

Learns extraction patterns from validated invoices to improve field extraction
accuracy. This module is used as a fallback when position-based extraction
isn't available or returns low confidence.

What it learns:
- Invoice number formats/regex patterns and common prefixes
- Date formats
- Amount patterns and expected value ranges
- Line item name patterns and value ranges

The learned patterns are stored per-vendor and improve with each
validated invoice. They provide a "second line of defense" when
generic extraction fails to find a field.
"""

import re
import json
from pathlib import Path
from typing import Optional
from collections import Counter
import numpy as np


class PatternLearner:
    """
    Learns extraction patterns from validated invoices.

    Stores learned patterns per vendor for future extraction.
    Each time an invoice is validated, its patterns are added to
    improve future extraction accuracy.

    Pattern types:
    - invoice_number: regex patterns, prefixes, digit counts
    - invoice_date: date formats
    - invoice_amount: regex patterns, value ranges
    - items: common names, name patterns, quantity/price ranges
    """

    def __init__(self, vendor_id: str, vendor_dir: Path):
        """
        Initialize pattern learner for a vendor.

        Args:
            vendor_id: Unique vendor identifier
            vendor_dir: Path to vendor's directory
        """
        self.vendor_id = vendor_id
        self.vendor_dir = Path(vendor_dir)
        self.patterns = self._load_patterns()

    def _get_patterns_path(self) -> Path:
        """Get path to patterns.json file."""
        return self.vendor_dir / "patterns.json"

    def _load_patterns(self) -> dict:
        """
        Load learned patterns from disk or return defaults.

        If patterns.json exists, load it. Otherwise return default patterns
        that provide reasonable extraction for new vendors.
        """
        path = self._get_patterns_path()
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return self._default_patterns()

    def _default_patterns(self) -> dict:
        """
        Default patterns for new vendors with no training data.

        These are generic patterns that match common invoice formats.
        They are less accurate than learned patterns but provide
        reasonable fallback extraction.
        """
        return {
            'invoice_number': {
                'patterns': [
                    r'(?:invoice\s*(?:no\.?|number|#)?:?\s*)([A-Z0-9\-]+)',
                    r'(?<!\w)(INV-?\d+[-\d]*)(?!\w)',
                    r'(?<!\w)(INV[_-]\d+[-\d]*)(?!\w)',
                ],
                'common_prefixes': ['INV', 'Invoice', 'inv'],
                'prefix_whitelist': ['INV', 'INVOICE', 'inv', 'Inv']
            },
            'invoice_date': {
                'formats': ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d'],
                'patterns': [
                    r'\b(\d{4}-\d{2}-\d{2})\b',
                    r'\b(\d{2}/\d{2}/\d{4})\b',
                    r'\b(\d{2}-\d{2}-\d{4})\b',
                ]
            },
            'invoice_amount': {
                'patterns': [
                    r'(?:total|amount|grand\s*total|balance\s*due)[:.\s]*[$]?\s*([\d,]+\.?\d*)',
                    r'[$]\s*([\d,]+\.?\d*)',
                ],
                'range': {'min': 0, 'max': 1000000}
            },
            'items': {
                'common_names': [],
                'name_patterns': [],
                'item_no_patterns': [],
                'quantity_range': {'min': 1, 'max': 100},
                'price_range': {'min': 0, 'max': 100000}
            }
        }

    def learn_from_sample(self, sample: dict):
        """
        Update patterns based on a validated invoice sample.

        This is called during validation. For each field, it analyzes
        the validated value and updates patterns accordingly.

        Args:
            sample: dict with validated invoice data:
                   invoice_number, invoice_date, invoice_amount, items
        """
        # Learn invoice number pattern
        inv_num = sample.get('invoice_number', '')
        if inv_num and str(inv_num) != 'nan':
            inv_num = str(inv_num).strip()
            # Detect and learn prefix pattern (e.g., "INV", "Invoice")
            prefix_match = re.match(r'^([A-Za-z]+)', inv_num)
            if prefix_match:
                prefix = prefix_match.group(1).upper()
                if prefix not in self.patterns['invoice_number'].get('prefix_whitelist', []):
                    self.patterns['invoice_number']['prefix_whitelist'].append(prefix)

            # Learn digit count (e.g., 5 digits in "INV-12345")
            num_match = re.search(r'(\d+)', inv_num)
            if num_match:
                digits = len(num_match.group(1))
                if digits not in self.patterns['invoice_number'].get('digit_counts', []):
                    if 'digit_counts' not in self.patterns['invoice_number']:
                        self.patterns['invoice_number']['digit_counts'] = []
                    self.patterns['invoice_number']['digit_counts'].append(digits)

        # Learn date format
        date_str = sample.get('invoice_date', '')
        if date_str and str(date_str) != 'nan':
            date_str = str(date_str).strip()
            if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                if '%Y-%m-%d' not in self.patterns['invoice_date']['formats']:
                    self.patterns['invoice_date']['formats'].insert(0, '%Y-%m-%d')
            elif re.match(r'^\d{2}/\d{2}/\d{4}$', date_str):
                if '%m/%d/%Y' not in self.patterns['invoice_date']['formats']:
                    self.patterns['invoice_date']['formats'].insert(0, '%m/%d/%Y')

        # Learn amount range (typical invoice amounts for this vendor)
        amount = sample.get('invoice_amount', 0)
        try:
            amount = float(amount)
            if amount > 0:
                ranges = self.patterns['invoice_amount'].get('range', {})
                if 'values' not in ranges:
                    ranges['values'] = []
                ranges['values'].append(amount)
                ranges['min'] = min(ranges.get('min', amount), amount)
                ranges['max'] = max(ranges.get('max', amount), amount)
                # Keep only last 100 values for range calculation
                ranges['values'] = ranges['values'][-100:]
                self.patterns['invoice_amount']['range'] = ranges
        except (ValueError, TypeError):
            pass

        # Learn line item patterns
        items = sample.get('items', [])
        if items:
            if 'items' not in self.patterns:
                self.patterns['items'] = {
                    'common_names': [],
                    'name_patterns': [],
                    'item_no_patterns': [],
                    'quantity_range': {'min': 1, 'max': 100},
                    'price_range': {'min': 0, 'max': 100000}
                }

            for item in items:
                item_name = item.get('item_name', '')
                item_no = item.get('item_no')
                if item_name and str(item_name).lower() not in ['unknown item', 'unknown', '']:
                    # Learn common item names for fuzzy matching
                    if item_name.lower() not in self.patterns['items']['common_names']:
                        self.patterns['items']['common_names'].append(item_name.lower())

                    # Learn item name patterns - extract meaningful words
                    words = re.findall(r'[A-Za-z]+', item_name.lower())
                    for word in words:
                        if len(word) > 2:  # Ignore short words
                            if word not in self.patterns['items']['name_patterns']:
                                self.patterns['items']['name_patterns'].append(word)

                # Learn item number pattern
                if item_no is not None:
                    if 'item_no_patterns' not in self.patterns['items']:
                        self.patterns['items']['item_no_patterns'] = []
                    if item_no not in self.patterns['items']['item_no_patterns']:
                        self.patterns['items']['item_no_patterns'].append(item_no)

                # Learn quantity range
                qty = item.get('item_quantity', 0)
                try:
                    qty = float(qty)
                    item_range = self.patterns['items']['quantity_range']
                    item_range['min'] = min(item_range.get('min', qty), qty)
                    item_range['max'] = max(item_range.get('max', qty), qty)
                except (ValueError, TypeError):
                    pass

                # Learn price range
                price = item.get('per_item_price', 0)
                try:
                    price = float(price)
                    if price > 0:
                        price_range = self.patterns['items']['price_range']
                        price_range['min'] = min(price_range.get('min', price), price)
                        price_range['max'] = max(price_range.get('max', price), price)
                except (ValueError, TypeError):
                    pass

            # Keep lists manageable (most recent items only)
            self.patterns['items']['common_names'] = self.patterns['items']['common_names'][-50:]
            self.patterns['items']['name_patterns'] = self.patterns['items']['name_patterns'][-100:]

        self._save_patterns()

    def _save_patterns(self):
        """Save learned patterns to disk."""
        path = self._get_patterns_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.patterns, f, indent=2)


class CandidateExtractor:
    """
    Extracts all candidate values matching learned patterns.

    For each field, finds all values that match learned patterns,
    then ranks them by confidence based on:
    - Pattern match quality
    - Context (near relevant keywords)
    - Agreement with learned ranges

    Used as a fallback when position-based extraction fails.
    """

    def __init__(self, patterns: dict, ocr_text: str, words: list = None):
        """
        Initialize candidate extractor.

        Args:
            patterns: Learned patterns from PatternLearner
            ocr_text: Raw OCR text from invoice
            words: Optional word list with bounding boxes for position extraction
        """
        self.patterns = patterns
        self.ocr_text = ocr_text
        self.words = words or []
        self._build_word_lookup()

    def _build_word_lookup(self):
        """Build lookup for position-based extraction if words available."""
        self._word_lookup = {}
        for i, word in enumerate(self.words):
            norm = word.get('text', '').lower()
            self._word_lookup[norm] = i

    def extract_invoice_numbers(self) -> list:
        """
        Extract all invoice number candidates.

        Applies learned regex patterns to find all matching strings,
        validates against learned prefixes and digit counts,
        then ranks by confidence.

        Returns:
            List of dicts with value, confidence, and pattern
        """
        candidates = []

        for pattern in self.patterns['invoice_number']['patterns']:
            for match in re.finditer(pattern, self.ocr_text, re.IGNORECASE):
                value = match.group(1) if match.groups() else match.group()
                value = value.strip()

                # Validate against learned patterns
                if self._validate_invoice_number(value):
                    conf = self._calculate_inv_num_confidence(value, match)
                    candidates.append({
                        'value': value,
                        'confidence': conf,
                        'pattern': pattern
                    })

        # Dedupe and return top candidates
        seen = set()
        unique = []
        for c in candidates:
            if c['value'] not in seen:
                seen.add(c['value'])
                unique.append(c)

        return sorted(unique, key=lambda x: x['confidence'], reverse=True)

    def _validate_invoice_number(self, value: str) -> bool:
        """
        Validate invoice number against learned patterns.

        Checks:
        - Prefix is in learned whitelist
        - Digit count matches training data (when we have enough samples)
        """
        if not value:
            return False

        # Check prefix whitelist
        prefix_match = re.match(r'^([A-Za-z]+)', value)
        if prefix_match:
            prefix = prefix_match.group(1).upper()
            whitelist = self.patterns['invoice_number'].get('prefix_whitelist', [])
            if whitelist and prefix not in whitelist:
                # Only enforce if we have enough training data
                if len(whitelist) >= 3:
                    return False

        # Check digit count (lenient)
        digit_counts = self.patterns['invoice_number'].get('digit_counts', [])
        if digit_counts:
            num_match = re.search(r'(\d+)', value)
            if num_match:
                digits = len(num_match.group(1))
                if digits not in digit_counts:
                    pass  # Lenient - don't reject

        return True

    def _calculate_inv_num_confidence(self, value: str, match) -> float:
        """
        Calculate confidence for invoice number candidate.

        Factors:
        - Base confidence
        - Matches learned prefix (+0.15)
        - Matches learned digit count (+0.15)
        - Appears multiple times in document (-0.2)
        """
        base = 0.6

        # Higher confidence if matches learned prefix
        prefix_match = re.match(r'^([A-Za-z]+)', value)
        if prefix_match:
            prefix = prefix_match.group(1).upper()
            if prefix in self.patterns['invoice_number'].get('prefix_whitelist', []):
                base += 0.15

        # Higher confidence if digit count matches training
        digit_counts = self.patterns['invoice_number'].get('digit_counts', [])
        num_match = re.search(r'(\d+)', value)
        if num_match and digit_counts:
            if len(num_match.group(1)) in digit_counts:
                base += 0.15

        # Lower confidence if value appears multiple times
        count = self.ocr_text.lower().count(value.lower())
        if count > 1:
            base -= 0.2

        return max(0.1, min(0.95, base))

    def extract_dates(self) -> list:
        """
        Extract all date candidates.

        Uses learned date formats to find and rank date matches.

        Returns:
            List of dicts with value, confidence, and pattern
        """
        candidates = []

        for pattern in self.patterns['invoice_date']['patterns']:
            for match in re.finditer(pattern, self.ocr_text):
                value = match.group()
                conf = self._calculate_date_confidence(value, pattern)
                candidates.append({
                    'value': value,
                    'confidence': conf,
                    'pattern': pattern
                })

        # Dedupe and sort
        seen = set()
        unique = []
        for c in candidates:
            if c['value'] not in seen:
                seen.add(c['value'])
                unique.append(c)

        return sorted(unique, key=lambda x: x['confidence'], reverse=True)

    def _calculate_date_confidence(self, value: str, pattern: str) -> float:
        """
        Calculate confidence for date candidate.

        Factors:
        - Matches learned date format
        - Appears near date-related keywords
        """
        base = 0.7

        # Check if format matches learned formats
        formats = self.patterns['invoice_date'].get('formats', [])
        if '%Y-%m-%d' in formats and re.match(r'^\d{4}-\d{2}-\d{2}$', value):
            base += 0.15
        elif '%m/%d/%Y' in formats and re.match(r'^\d{2}/\d{2}/\d{4}$', value):
            base += 0.15

        # Higher confidence for dates near context keywords
        context_lower = self.ocr_text.lower()
        date_idx = context_lower.find(value.lower())
        if date_idx > 0:
            window = context_lower[max(0, date_idx-20):date_idx+len(value)+20].lower()
            if any(w in window for w in ['date', 'dated', 'invoice', 'inv']):
                base += 0.1

        return max(0.1, min(0.95, base))

    def extract_amounts(self) -> list:
        """
        Extract all amount candidates.

        First finds amounts near "total" keywords (high confidence),
        then finds standalone dollar amounts (lower confidence).

        Returns:
            List of dicts with value, confidence, and is_total flag
        """
        candidates = []
        seen_values = set()

        # First try total-related patterns (higher confidence)
        total_patterns = [
            r'(?:total|amount|grand\s*total|balance\s*due|subtotal)[:.\s]*[$]?\s*([\d,]+\.?\d*)',
        ]

        for pattern in total_patterns:
            for match in re.finditer(pattern, self.ocr_text, re.IGNORECASE):
                value_str = match.group(1) if match.groups() else match.group()
                value_str = value_str.replace(',', '')
                try:
                    value = float(value_str)
                    if value > 0 and value not in seen_values:
                        seen_values.add(value)
                        conf = self._calculate_amount_confidence(value, match, is_total=True)
                        candidates.append({
                            'value': value,
                            'confidence': conf,
                            'is_total': True
                        })
                except ValueError:
                    pass

        # Then generic dollar amounts (lower confidence)
        generic_pattern = r'[$]\s*([\d,]+\.?\d*)'
        for match in re.finditer(generic_pattern, self.ocr_text):
            value_str = match.group(1).replace(',', '')
            try:
                value = float(value_str)
                if value > 0 and value not in seen_values:
                    seen_values.add(value)
                    conf = self._calculate_amount_confidence(value, match, is_total=False)
                    candidates.append({
                        'value': value,
                        'confidence': conf,
                        'is_total': False
                    })
            except ValueError:
                pass

        # Sort: totals first (higher priority), then by confidence
        totals = [c for c in candidates if c.get('is_total')]
        non_totals = [c for c in candidates if not c.get('is_total')]

        totals.sort(key=lambda x: (x['confidence'], x['value']), reverse=True)
        non_totals.sort(key=lambda x: (x['confidence'], x['value']), reverse=True)

        return totals + non_totals

    def _calculate_amount_confidence(self, value: float, match, is_total: bool) -> float:
        """
        Calculate confidence for amount candidate.

        Factors:
        - Is it near total keywords (+0.4 base if near total)
        - Is it in learned range (+0.1)
        - Has context keywords (+0.1)
        """
        base = 0.9 if is_total else 0.5

        # Check against learned range
        ranges = self.patterns['invoice_amount'].get('range', {})
        if 'min' in ranges and 'max' in ranges:
            min_val, max_val = ranges['min'], ranges['max']
            if min_val <= value <= max_val:
                base += 0.1
            elif 0.5 * min_val <= value <= 2 * max_val:
                base += 0.0
            else:
                base -= 0.2

        # Context boost
        context = self.ocr_text[max(0, match.start()-30):match.end()+30].lower()
        if any(w in context for w in ['total', 'amount', 'due', 'grand']):
            base += 0.1

        return max(0.1, min(0.95, base))

    def get_top_candidates(self, max_per_field: int = 3) -> dict:
        """
        Get top candidates for all fields.

        Args:
            max_per_field: Maximum candidates to return per field

        Returns:
            Dict with top candidates for each field
        """
        return {
            'invoice_number': self.extract_invoice_numbers()[:max_per_field],
            'invoice_date': self.extract_dates()[:max_per_field],
            'invoice_amount': self.extract_amounts()[:max_per_field],
            'items': self.extract_items()[:max_per_field]
        }

    def extract_items(self) -> list:
        """
        Extract line item candidates from OCR text.

        Uses patterns to identify rows containing item data and
        extracts the relevant fields (name, qty, price, total).

        Returns:
            List of item dicts with confidence scores
        """
        items = []
        item_patterns = [
            # Match rows with item name, qty, price
            r'([A-Za-z0-9\s\-\+]+?)\s+(\d+)\s+([\d,]+\.?\d*)\s+([\d,]+\.?\d*)',
            # Match items near common keywords
            r'(?:item|product|description|service)[:.\s]+([A-Za-z0-9\s\-\+]+?)(?:\s+\d+\s+[$]?[\d,]+\.?\d*)',
        ]

        # Check for learned item names first
        learned_names = self.patterns.get('items', {}).get('common_names', [])
        learned_words = self.patterns.get('items', {}).get('name_patterns', [])

        # Look for line items in OCR text
        lines = self.ocr_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to extract quantity and price from line
            qty_match = re.search(r'\b(\d+)\s*(?:pcs?|pieces?|units?|boxes?|sets?|x\b)', line, re.IGNORECASE)
            price_match = re.search(r'[$]?\s*([\d,]+\.?\d*)\s*(?:each|per|ea\.?)?$', line, re.IGNORECASE)
            total_match = re.search(r'[$]?\s*([\d,]+\.?\d*)\s*$', line)

            qty = float(qty_match.group(1)) if qty_match else 1.0
            price = float(price_match.group(1).replace(',', '')) if price_match else 0.0
            total = float(total_match.group(1).replace(',', '')) if total_match else price * qty

            # Calculate confidence based on learned patterns
            conf = 0.3  # base confidence
            line_lower = line.lower()

            # Check if line contains learned item words
            for word in learned_words:
                if word in line_lower:
                    conf = max(conf, 0.6)
                    break

            # Check for learned item names
            for name in learned_names:
                if name in line_lower:
                    conf = max(conf, 0.75)
                    break

            # Boost confidence if qty and price found together
            if qty_match and price_match:
                conf = min(conf + 0.15, 0.95)

            # Clean up item name
            item_name = re.sub(r'[\d\s]+(?:pcs?|pieces?|units?|boxes?|sets?)?\s*[$]?[\d,]+\.?\d*\s*[$]?[\d,]+\.?\d*$',
                              '', line, flags=re.IGNORECASE).strip()
            item_name = re.sub(r'^\d+\s+', '', item_name).strip()  # Remove leading qty

            # Extract item number if present
            item_no = None
            item_no_match = re.match(r'^(\d+)[\s\.\)]+', line)
            if item_no_match:
                item_no = int(item_no_match.group(1))
                item_name = re.sub(r'^\d+\s+', '', item_name).strip()

            if item_name and len(item_name) > 1:
                items.append({
                    'value': {
                        'item_no': item_no,
                        'item_name': item_name,
                        'item_quantity': qty,
                        'per_item_price': price,
                        'total_item_price': total
                    },
                    'confidence': conf,
                    'raw_line': line
                })

        # If no items but have learned names, create candidates from them
        if not items and learned_names:
            price_range = self.patterns.get('items', {}).get('price_range', {})
            qty_range = self.patterns.get('items', {}).get('quantity_range', {})
            for name in learned_names[:5]:
                items.append({
                    'value': {
                        'item_name': name.title(),
                        'item_quantity': qty_range.get('min', 1) or 1,
                        'per_item_price': price_range.get('min', 0) or 0,
                        'total_item_price': price_range.get('min', 0) or 0
                    },
                    'confidence': 0.4,
                    'raw_line': name
                })

        # Sort by confidence
        items.sort(key=lambda x: x['confidence'], reverse=True)
        return items


def learn_patterns(vendor_id: str, vendor_dir: Path, training_samples: list):
    """
    Update learned patterns from validated invoices.

    Args:
        vendor_id: Vendor identifier
        vendor_dir: Path to vendor directory
        training_samples: List of validated invoice dicts

    Returns:
        Updated patterns dict
    """
    learner = PatternLearner(vendor_id, vendor_dir)
    for sample in training_samples:
        learner.learn_from_sample(sample)
    return learner.patterns


def extract_candidates(patterns: dict, ocr_text: str, words: list = None) -> dict:
    """
    Extract and rank field candidates using learned patterns.

    Args:
        patterns: Learned pattern dict
        ocr_text: Raw OCR text
        words: Optional word list with bounding boxes

    Returns:
        Dict with top candidates per field
    """
    extractor = CandidateExtractor(patterns, ocr_text, words)
    return extractor.get_top_candidates()
