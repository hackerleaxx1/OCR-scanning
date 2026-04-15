"""
Position-based Generic Field Extractor for Invoice Processing.

This extractor uses bounding box information from Tesseract OCR to find
field values by locating anchor labels and extracting values at expected
relative positions. It serves as the fallback when no vendor-specific
template exists.

The extraction strategy:
1. Find anchor labels (e.g., "Invoice #:", "Total:", "Date:")
2. Determine the label's position relative to its value (right, below, etc.)
3. Extract the value at the expected relative position
4. Fall back to regex patterns if position-based extraction fails

This is less accurate than the learned position templates but provides
reasonable extraction for new vendors without training data.
"""

import re
from typing import Optional
from dataclasses import dataclass


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BBox:
    """
    Bounding box representing a word's position in the invoice image.

    Attributes:
        text: The actual text content of the word
        left: X coordinate of the left edge (pixels)
        top: Y coordinate of the top edge (pixels)
        width: Width of the bounding box (pixels)
        height: Height of the bounding box (pixels)
        conf: OCR confidence score (0-100, default 50)
    """
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float = 50.0

    @property
    def right(self) -> int:
        """X coordinate of the right edge."""
        return self.left + self.width

    @property
    def bottom(self) -> int:
        """Y coordinate of the bottom edge."""
        return self.top + self.height

    @property
    def center_x(self) -> int:
        """X coordinate of the center point."""
        return self.left + self.width // 2

    @property
    def center_y(self) -> int:
        """Y coordinate of the center point."""
        return self.top + self.height // 2


# =============================================================================
# Field Extractor
# =============================================================================

class FieldExtractor:
    """
    Extracts invoice fields using position-based template matching.

    For fixed-format invoices, field positions are predictable relative to
    their labels. This extractor combines multiple strategies:

    1. Position-based: Find label, extract value at learned relative position
    2. Pattern-based regex: Fallback when position doesn't yield results
    3. Heuristic rules: For line items, uses number patterns and positioning

    The extraction order of priority for each field is:
    - Try position-based first (most accurate for fixed layouts)
    - Fall back to regex patterns if position fails
    - For amounts, find largest value matching expected patterns
    """

    # -------------------------------------------------------------------------
    # Label Patterns - ordered from most specific to least specific
    # More specific patterns match first to avoid false positives
    # -------------------------------------------------------------------------
    LABEL_PATTERNS = {
        'invoice_number': [
            r'^invoice\s+number\s*#?:?$',
            r'^inv\.?\s+no\.?:?$',
            r'^invoice\s*#:?$',
            r'^inv\.?\s*(?:no\.?|number|#)?$',
            r'^invoice\s*(?:no\.?|number)?$',
        ],
        'invoice_date': [
            r'^invoice\s+date:?$',
            r'^dated:?$',
            r'^date:?$',
        ],
        'invoice_amount': [
            r'^grand\s*total:?$',
            r'^total\s*amount:?$',
            r'^total\s*due:?$',
            r'^balance\s*due:?$',
            r'^invoice\s+total:?$',
            r'^amount\s*due:?$',
            r'^net\s*amount:?$',
            r'^gross\s*total:?$',
            r'^total:?$',
            r'^amount:?$',
        ],
        'subtotal': [
            r'^subtotal:?$',
            r'^invoice\s*subtotal:?$',
            r'^sub-total:?$',
        ],
        'tax': [
            r'^tax:?$',
            r'^sales\s*tax:?$',
            r'^cgst:?$',
            r'^sgst:?$',
            r'^vat:?$',
            r'^gst:?$',
        ],
        'item_quantity': [
            r'^qty:?$',
            r'^quantity:?$',
            r'^units:?$',
        ],
        'item_price': [
            r'^unit\s*price:?$',
            r'^price:?$',
            r'^rate:?$',
            r'^per\s*item:?$',
        ],
        'item_total': [
            r'^total:?$',
            r'^item\s*total:?$',
            r'^line\s*total:?$',
        ],
    }

    # Regex patterns for field value extraction (position-based fallback)
    VALUE_PATTERNS = {
        'invoice_number': [
            r'(?:invoice\s*(?:no\.?|number|#)?:?\s*)([A-Z0-9\-]+)',
            r'(?:inv\.?\s*(?:no\.?|number|#)?:?\s*)([A-Z0-9\-]+)',
            r'(?<!\w)(INV-?\d+[-\d]*)(?!\w)',
            r'(?<!\w)(INV[_-]\d+[-\d]*)(?!\w)',
        ],
        'invoice_date': [
            r'\b(\d{4}-\d{2}-\d{2})\b',
            r'\b(\d{2}/\d{2}/\d{4})\b',
            r'\b(\d{2}-\d{2}-\d{4})\b',
            r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b',
        ],
        'invoice_amount': [
            r'(?:total|amount|grand\s*total|balance\s*due)[:.\s]*[$]?\s*([\d,]+\.?\d*)',
            r'[$]\s*([\d,]+\.?\d*)',
        ],
        'quantity': [
            r'\b(\d+)\s*(?:pcs?|pieces?|units?|boxes?|sets?)\b',
            r'\b(\d+)\s*x\b',
        ],
        'price': [
            r'\$\s*([\d,]+\.?\d*)',
            r'([\d,]+\.?\d*)\s*(?:each|per|ea\.?)',
        ],
    }

    def __init__(self, words: list):
        """
        Initialize with OCR words including bounding boxes.

        Args:
            words: List of dicts with keys: text, left, top, width, height, conf
                  These come from tesseract._extract_words_from_image()
        """
        # Convert raw OCR word dicts to BBox objects for easier manipulation
        self.words = [BBox(
            text=w['text'],
            left=int(w['left']),
            top=int(w['top']),
            width=int(w['width']),
            height=int(w['height']),
            conf=float(w.get('conf', 50.0))
        ) for w in words if w.get('text', '').strip()]
        self._build_text_map()

    def _build_text_map(self):
        """
        Build fast lookup from normalized text to word index.

        This allows O(1) lookup when searching for specific label text.
        Multiple words with the same text are stored as a list of indices.
        """
        self._text_to_idx = {}
        for i, word in enumerate(self.words):
            norm = word.text.lower().strip()
            if norm not in self._text_to_idx:
                self._text_to_idx[norm] = []
            self._text_to_idx[norm].append(i)

    def _find_label(self, label_patterns: list) -> Optional[BBox]:
        """
        Find a label matching any of the given patterns.

        Prefers longer (more specific) matches to avoid false positives.
        First checks multi-word labels (e.g., "Invoice Number" as two words),
        then falls back to single-word labels.

        Args:
            label_patterns: List of regex patterns to match against

        Returns:
            BBox of the matching label word, or None if not found
        """
        best_match = None
        best_len = 0
        best_idx = -1

        # First, try to find multi-word labels (e.g., "Invoice number" as two words)
        for i in range(len(self.words) - 1):
            two_words = ' '.join(w.text.lower() for w in self.words[i:i+2])
            for pattern in label_patterns:
                if re.match(pattern, two_words, re.IGNORECASE):
                    if len(two_words) > best_len:
                        best_len = len(two_words)
                        # Use the last word as the anchor (e.g., "number" in "Invoice number")
                        best_match = self.words[i + 1]
                        best_idx = i + 1
                        break

        # Then try single-word labels
        for norm_text, indices in self._text_to_idx.items():
            for pattern in label_patterns:
                if re.match(pattern, norm_text, re.IGNORECASE):
                    # Prefer longer matches (more specific labels)
                    if len(norm_text) > best_len:
                        best_len = len(norm_text)
                        best_match = self.words[indices[0]]
                        best_idx = indices[0]
        return best_match

    def _get_text_at_position(
        self,
        anchor: BBox,
        direction: str = 'right',
        max_distance: int = 300
    ) -> Optional[str]:
        """
        Get text at a position relative to anchor label.

        Searches for words in the specified direction from the anchor label
        and returns the closest one within the maximum distance.

        Args:
            anchor: The anchor bounding box (label position)
            direction: 'right', 'below', or 'down-right'
            max_distance: Maximum pixels to search from anchor edge

        Returns:
            Text content of the value word, or None if not found
        """
        candidates = []

        for word in self.words:
            if word is anchor:
                continue

            if direction == 'right':
                # Same line, to the right
                if (abs(word.top - anchor.top) < anchor.height * 0.5 and
                    anchor.right < word.left < anchor.right + max_distance):
                    candidates.append((word, word.left - anchor.right))
            elif direction == 'below':
                # Below the anchor, same column (allowing some x overlap)
                if (anchor.bottom < word.top < anchor.bottom + max_distance and
                    word.left >= anchor.left - anchor.width and
                    word.right <= anchor.right + anchor.width * 2):
                    candidates.append((word, word.top - anchor.bottom))
            elif direction == 'down-right':
                # Below and to the right
                if (anchor.bottom < word.top < anchor.bottom + max_distance * 1.5 and
                    anchor.right < word.left < anchor.right + max_distance):
                    candidates.append((word, abs(word.left - anchor.right) + (word.top - anchor.bottom)))

        if not candidates:
            return None

        # Return the closest candidate
        closest = min(candidates, key=lambda x: x[1])
        return closest[0].text

    def _extract_amount(self, text: str) -> Optional[float]:
        """
        Extract numeric amount from text.

        Handles currency symbols ($, €, £), commas as thousands separators,
        and parenthesized negative numbers.

        Args:
            text: Text potentially containing a numeric amount

        Returns:
            Float value, or None if no valid amount found
        """
        if not text:
            return None
        # Remove currency symbols and commas
        cleaned = re.sub(r'[$€£]', '', text)
        # Handle "(500.00)" style negative numbers
        is_negative = '(' in text and ')' in text
        match = re.search(r'([\d,]+\.?\d*)', cleaned)
        if match:
            value = float(match.group(1).replace(',', ''))
            return -value if is_negative else value
        return None

    def _extract_date(self, text: str) -> Optional[str]:
        """
        Normalize and validate date string.

        Recognizes multiple date formats but returns the raw matched text
        without conversion - format validation only.

        Args:
            text: Text potentially containing a date

        Returns:
            Date string if valid pattern found, None otherwise
        """
        patterns = [
            (r'\d{4}-\d{2}-\d{2}', 'YYYY-MM-DD'),
            (r'\d{2}/\d{2}/\d{4}', 'MM/DD/YYYY'),
            (r'\d{2}-\d{2}-\d{4}', 'MM-DD-YYYY'),
        ]
        for pattern, fmt in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()
        return None

    # =========================================================================
    # Field-Specific Extraction Methods
    # =========================================================================

    def extract_invoice_number(self) -> Optional[str]:
        """
        Extract invoice number using position-based or regex extraction.

        Strategy:
        1. Find "Invoice No." label, extract value to the right
        2. Fall back to regex patterns matching common invoice number formats

        Returns:
            Invoice number string, or None if not found
        """
        # Try position-based first
        label = self._find_label(self.LABEL_PATTERNS['invoice_number'])
        if label:
            value = self._get_text_at_position(label, 'right', max_distance=200)
            if value:
                return value

        # Fall back to regex on all text
        all_text = ' '.join(w.text for w in self.words)
        for pattern in self.VALUE_PATTERNS['invoice_number']:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def extract_invoice_date(self) -> Optional[str]:
        """
        Extract invoice date using position-based or regex extraction.

        Strategy:
        1. Find "Date" or "Invoice Date" label, extract value to the right
        2. Fall back to regex patterns matching date formats (YYYY-MM-DD, MM/DD/YYYY, etc.)

        Returns:
            Date string if found, None otherwise
        """
        # Try position-based first
        label = self._find_label(self.LABEL_PATTERNS['invoice_date'])
        if label:
            value = self._get_text_at_position(label, 'right', max_distance=200)
            if value:
                date = self._extract_date(value)
                if date:
                    return date

        # Fall back to regex
        all_text = ' '.join(w.text for w in self.words)
        for pattern in self.VALUE_PATTERNS['invoice_date']:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                return match.group()
        return None

    def extract_invoice_amount(self) -> Optional[float]:
        """
        Extract total invoice amount.

        Strategy:
        1. Find "Total" label, extract value to the right or below
        2. Fall back to regex finding the largest amount on the invoice

        The "total" amount is typically the largest dollar value on the invoice.

        Returns:
            Total amount as float, or None if not found
        """
        # Try position-based first - look for "Total" label
        label = self._find_label(self.LABEL_PATTERNS['invoice_amount'])
        if label:
            value = self._get_text_at_position(label, 'right', max_distance=250)
            if value:
                amount = self._extract_amount(value)
                if amount:
                    return amount

            # Try below if right didn't work
            value = self._get_text_at_position(label, 'below', max_distance=50)
            if value:
                amount = self._extract_amount(value)
                if amount:
                    return amount

        # Fall back to regex - find largest amount
        all_text = ' '.join(w.text for w in self.words)
        amounts = []
        for pattern in self.VALUE_PATTERNS['invoice_amount']:
            for match in re.finditer(pattern, all_text, re.IGNORECASE):
                amount = self._extract_amount(match.group())
                if amount and amount > 0:
                    amounts.append(amount)

        if amounts:
            return max(amounts)
        return None

    def extract_all(self) -> dict:
        """
        Extract all common invoice fields including line items.

        Returns:
            Dict with invoice_number, invoice_date, invoice_amount, and items
        """
        result = {
            'invoice_number': self.extract_invoice_number(),
            'invoice_date': self.extract_invoice_date(),
            'invoice_amount': self.extract_invoice_amount(),
        }

        # Extract line items using table structure detection
        items = self._extract_line_items()
        result['items'] = items

        return result

    # =========================================================================
    # Line Item Extraction
    # =========================================================================

    def _extract_line_items(self) -> list:
        """
        Extract line items from the invoice table.

        Identifies table rows by grouping words vertically (same Y position),
        then parses each row to extract item name, quantity, unit price, and total.

        Skips:
        - Header rows (labeled "Total", "Subtotal", "Tax", etc.)
        - Rows that are purely numeric (likely not item rows)

        Returns:
            List of item dicts with item_no, item_name, item_quantity,
            per_item_price, total_item_price
        """
        items = []

        # Group words by vertical position (table rows)
        lines = self._group_words_by_line()

        for line_words in lines:
            line_text = ' '.join(w.text for w in line_words)
            if not line_text.strip():
                continue

            # Skip header/total lines that are just labels (short text)
            lower_line = line_text.lower()
            if any(skip in lower_line for skip in ['total', 'subtotal', 'tax', 'amount', 'due', 'balance', 'invoice']):
                if len(line_text) < 30:  # Short lines that are likely labels
                    continue

            # Try to extract item data from this line
            item = self._parse_item_from_line(line_words, line_text)
            if item:
                items.append(item)

        return items

    def _group_words_by_line(self) -> list:
        """
        Group words into lines based on vertical position.

        Words with similar Y coordinates (within 15 pixels) are considered
        on the same line. Lines are sorted by X position within each line.

        Returns:
            List of lines, where each line is a list of BBox objects sorted by X
        """
        if not self.words:
            return []

        # Sort by top position, then left position
        sorted_words = sorted(self.words, key=lambda w: (w.top, w.left))

        lines = []
        current_line = [sorted_words[0]]
        current_baseline = sorted_words[0].top

        for word in sorted_words[1:]:
            # If word is on roughly the same line (within half a line height)
            if abs(word.top - current_baseline) < 15:
                current_line.append(word)
            else:
                # New line detected
                lines.append(sorted(current_line, key=lambda w: w.left))
                current_line = [word]
                current_baseline = word.top

        if current_line:
            lines.append(sorted(current_line, key=lambda w: w.left))

        return lines

    def _parse_item_from_line(self, line_words: list, line_text: str) -> Optional[dict]:
        """
        Parse a single table row into item data.

        Extracts item name, quantity, unit price, and total from a row of words.
        Uses position-based heuristics to identify which numbers represent
        which values (qty vs price vs total).

        Heuristics:
        - Item number is typically a small integer at the start of the line
        - Total is typically the largest number in the row
        - Quantity and unit price are in between
        - Item name is the text before the first number

        Args:
            line_words: List of BBox objects for words in this row
            line_text: All text in the row joined together

        Returns:
            Item dict, or None if parsing failed
        """
        if len(line_words) < 2:
            return None

        # Find all numeric values in the line with their positions
        numbers = []
        for word in line_words:
            num_match = re.search(r'([\d,]+\.?\d*)', word.text.replace(',', ''))
            if num_match:
                try:
                    num_val = float(num_match.group(1))
                    if num_val > 0:
                        numbers.append({
                            'value': num_val,
                            'word': word,
                            'position': word.left
                        })
                except ValueError:
                    pass

        if len(numbers) < 2:
            # Need at least 2 numbers (qty and total, or price and total)
            return None

        # Sort numbers by horizontal position (left to right)
        numbers.sort(key=lambda x: x['position'])

        item_name_words = []
        item_no = None
        qty = 1
        unit_price = 0.0
        total_price = 0.0

        # Identify item number - small integer at start of line
        if numbers and len(numbers) >= 2:
            first_num = numbers[0]['value']
            # Item numbers are typically 1-99 and come at the start of the line
            if first_num == int(first_num) and 0 < first_num < 100:
                remaining = numbers[1:]
                if len(remaining) >= 2:
                    # Has at least 2 more numbers - likely item data
                    item_no = int(first_num)
                    numbers = numbers[1:]

        # Find words that look like item names (non-numeric, meaningful length)
        for word in line_words:
            # Skip purely numeric words
            if re.match(r'^[\d,\.]+$', word.text):
                continue
            # Skip common non-item words (column headers, units, etc.)
            lower = word.text.lower()
            if lower in ['qty', 'quantity', 'unit', 'price', 'total', 'x', 'ea', 'each', 'pcs', 'no', 'item']:
                continue
            if len(word.text) < 2:
                continue
            item_name_words.append(word.text)

        item_name = ' '.join(item_name_words).strip()

        if not item_name:
            # Fallback: use text before first number as name
            text_before_first_num = line_text
            for n in numbers:
                idx = text_before_first_num.find(str(int(n['value'])))
                if idx > 0:
                    text_before_first_num = text_before_first_num[:idx]
                    break
            item_name = text_before_first_num.strip()

        if not item_name or len(item_name) < 2:
            return None

        # Assign values from remaining numbers
        if len(numbers) == 0:
            return None
        elif len(numbers) == 1:
            # Single number - likely the total
            total_price = numbers[0]['value']
        elif len(numbers) == 2:
            # Two numbers: could be (qty, total) or (unit_price, total)
            num0, num1 = numbers[0]['value'], numbers[1]['value']
            if num0 <= 1000 and num1 > num0 * 10:
                # num0 is small (likely qty), num1 is large (likely total)
                qty = int(num0) if num0 == int(num0) else 1
                total_price = num1
                if qty > 0:
                    unit_price = total_price / qty
            else:
                # Treat as unit_price, total
                unit_price = num0
                total_price = num1
        elif len(numbers) >= 3:
            # Three or more numbers: typically qty, unit_price, total
            num0, num1 = numbers[0]['value'], numbers[1]['value']
            num_last = numbers[-1]['value']

            # Check if qty * unit_price = total (allowing 1% tolerance)
            if num0 * num1 == num_last or abs(num_last - num0 * num1) < 0.01 * num_last:
                qty = int(num0) if num0 == int(num0) and num0 < 10000 else 1
                unit_price = num1
                total_price = num_last
            else:
                # Try alternative interpretations or use defaults
                qty = int(num0) if num0 == int(num0) and 0 < num0 < 10000 else 1
                unit_price = num_last  # Default to last as unit_price
                total_price = num_last

        # Calculate missing values
        if total_price > 0 and qty > 0 and unit_price == 0:
            unit_price = total_price / qty

        return {
            'item_no': item_no,
            'item_name': item_name.title() if item_name else 'Unknown Item',
            'item_quantity': qty,
            'per_item_price': round(unit_price, 2),
            'total_item_price': round(total_price, 2)
        }

    def get_confidence(self, field: str, value) -> float:
        """
        Estimate confidence for an extracted field value.

        Confidence scoring factors:
        - Position-based extraction vs regex fallback
        - Value format validation (dates, amounts within reasonable ranges)
        - Pattern matching against known formats

        Args:
            field: Field name (invoice_number, invoice_date, invoice_amount)
            value: Extracted value

        Returns:
            Confidence score between 0.0 and 0.98
        """
        if value is None:
            return 0.0

        base_confidence = 0.5  # Default for regex fallback

        # If value was found via position extraction, boost confidence
        label_matched = self._find_label(self.LABEL_PATTERNS.get(field, []))
        if label_matched:
            base_confidence = 0.85

        # Field-specific validation boosts/penalties
        if field == 'invoice_number' and value:
            # Invoice numbers should contain digits
            has_digits = bool(re.search(r'\d', str(value)))
            if has_digits:
                base_confidence += 0.1
            else:
                base_confidence -= 0.3  # Penalize non-numeric invoice numbers

            # Match against common patterns
            if re.match(r'^[A-Z]*\d+[A-Z0-9\-]*$', str(value)):
                base_confidence += 0.1

        elif field == 'invoice_date' and value:
            if self._extract_date(str(value)):
                base_confidence += 0.1
            else:
                base_confidence -= 0.2  # Penalize invalid date formats

        elif field == 'invoice_amount' and value:
            if isinstance(value, (int, float)) and 0 < value < 1_000_000:
                base_confidence += 0.1
            elif isinstance(value, str):
                # Try to extract amount from string
                extracted = self._extract_amount(value)
                if extracted and 0 < extracted < 1_000_000:
                    base_confidence += 0.1

        return max(0.0, min(base_confidence, 0.98))


# =============================================================================
# Main Entry Point
# =============================================================================

def extract_fields_from_ocr(ocr_result: dict) -> dict:
    """
    Main entry point for generic field extraction.

    This function is called when no vendor-specific position template exists.
    It wraps the FieldExtractor class with a simpler interface.

    Args:
        ocr_result: Dict from tesseract.extract_full_image_data()
                   Contains: text, words (with bbox), image_width, image_height

    Returns:
        Dict with:
            - predictions: Extracted field values
            - confidences: Confidence score per field (0.0 to 0.98)
    """
    if not ocr_result.get('words'):
        return {
            'predictions': {
                'invoice_number': None,
                'invoice_date': None,
                'invoice_amount': None,
            },
            'confidences': {
                'invoice_number': 0.0,
                'invoice_date': 0.0,
                'invoice_amount': 0.0,
            }
        }

    extractor = FieldExtractor(ocr_result['words'])

    predictions = extractor.extract_all()
    confidences = {
        'invoice_number': extractor.get_confidence('invoice_number', predictions['invoice_number']),
        'invoice_date': extractor.get_confidence('invoice_date', predictions['invoice_date']),
        'invoice_amount': extractor.get_confidence('invoice_amount', predictions['invoice_amount']),
    }

    return {
        'predictions': predictions,
        'confidences': confidences,
    }
