"""
Position-based Invoice Field Extraction using Learned Vendor Templates.

With only 3-4 validated invoices, this module learns:
1. Label positions (where field labels appear on the page)
2. Value offsets (where values appear relative to labels)
3. Table structure (row/column positions for line items)

This enables highly accurate extraction for fixed-format vendor invoices,
as subsequent invoices can use the learned template instead of generic heuristics.

The learning process:
1. On validation, the system analyzes OCR word positions + corrected values
2. It identifies which words were labels and which were values
3. It learns the spatial relationship (offset) between labels and values
4. For future invoices, it uses these learned positions for extraction
"""

import re
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class BBox:
    """
    Bounding box for a word detected by OCR.

    Attributes:
        text: The actual text content of the word
        left: X coordinate of left edge (pixels from image left)
        top: Y coordinate of top edge (pixels from image top)
        width: Width of bounding box in pixels
        height: Height of bounding box in pixels
        conf: Tesseract confidence score (0-100, default 50)
    """
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float = 50.0

    @property
    def right(self) -> int:
        """X coordinate of right edge."""
        return self.left + self.width

    @property
    def bottom(self) -> int:
        """Y coordinate of bottom edge."""
        return self.top + self.height

    @property
    def center_x(self) -> int:
        """X coordinate of center point."""
        return self.left + self.width // 2

    @property
    def center_y(self) -> int:
        """Y coordinate of center point."""
        return self.top + self.height // 2


@dataclass
class FieldTemplate:
    """
    Learned template for extracting a single field value.

    Stores the spatial relationship between a label and its value:
    - Where the label appears on the page (absolute position)
    - Where the value appears relative to the label (offset and direction)

    Attributes:
        field_name: Name of the field (e.g., 'invoice_number')
        label_texts: List of label variations that were learned
        label_x: X coordinate of label's left edge
        label_y: Y coordinate of label's top edge
        value_offset_x: Horizontal distance from label to value
        value_offset_y: Vertical distance from label to value
        value_position: Direction of value relative to label ('right', 'left', 'below')
        confidence_boost: Extra confidence this template provides (accumulates with each sample)
    """
    field_name: str
    label_texts: list
    label_x: int
    label_y: int
    value_offset_x: int
    value_offset_y: int
    value_position: str
    confidence_boost: float


@dataclass
class TableTemplate:
    """
    Learned template for extracting line item table data.

    Stores the structure of the invoice's item table:
    - Which column positions contain item numbers, names, quantities, prices, totals
    - The vertical positions for table start and row height

    Attributes:
        start_y: Y coordinate where the table starts
        row_height: Expected height of each row (for row counting)
        column_positions: X coordinates for each column type
        item_no_col: X coordinate for item number column
        item_name_col: X coordinate for item name column
        qty_col: X coordinate for quantity column
        price_col: X coordinate for unit price column
        total_col: X coordinate for total price column
    """
    start_y: int
    row_height: int
    column_positions: dict
    item_no_col: int
    item_name_col: int
    qty_col: int
    price_col: int
    total_col: int


class VendorTemplateLearner:
    """
    Learns vendor-specific invoice templates from validated invoices.

    This is the core of the "learning" system. After 3-4 validated invoices,
    this class has learned enough about the invoice layout to enable accurate
    position-based extraction.

    What it learns from each validated invoice:
    - Header fields: Where labels like "Invoice No.", "Date", "Total" appear
      and how far the values are from those labels (offset + direction)
    - Line items: The table structure - column positions, row heights, etc.

    How it uses what it learns:
    - For future invoices, it searches near the learned label positions
    - It applies the learned offsets to find values
    - Confidence is higher because the template was learned from real data

    The template improves with each validated invoice - offsets are averaged,
    new label variations are added, and table structure becomes more accurate.
    """

    def __init__(self, vendor_id: str, vendor_dir: Path):
        """
        Initialize template learner for a vendor.

        Args:
            vendor_id: Unique vendor identifier
            vendor_dir: Path to vendor's storage directory
        """
        self.vendor_id = vendor_id
        self.vendor_dir = Path(vendor_dir)
        self.templates: dict = {}                    # Field name -> FieldTemplate
        self.table_template: Optional[TableTemplate] = None  # Table structure
        self.image_width: int = 0                    # For scale-normalized learning
        self.image_height: int = 0

    def learn_from_invoice(
        self,
        ocr_words: list,
        ground_truth: dict,
        image_width: int,
        image_height: int
    ):
        """
        Learn field positions from a validated invoice.

        This is called during the validation step. It compares the OCR output
        (word positions) with the corrected data to learn where values are
        located relative to their labels.

        For each field in ground_truth, it:
        1. Searches for the label text (e.g., "invoice", "date", "total")
        2. Finds which word in the OCR is the actual value for that field
        3. Records the spatial relationship (label position, value position)
        4. For line items, learns the table structure

        Args:
            ocr_words: List of {text, left, top, width, height, conf}
                      from Tesseract OCR
            ground_truth: Dict with validated field values:
                         {invoice_number, invoice_date, invoice_amount, items}
            image_width: Width of the invoice image (for scale normalization)
            image_height: Height of the invoice image
        """
        self.image_width = image_width
        self.image_height = image_height

        # Convert OCR words to BBox objects for easier manipulation
        words = [BBox(
            text=w['text'],
            left=int(w['left']),
            top=int(w['top']),
            width=int(w['width']),
            height=int(w['height']),
            conf=float(w.get('conf', 50.0))
        ) for w in ocr_words if w.get('text', '').strip()]

        # Build lookup table for fast text search
        text_to_indices = {}
        for i, word in enumerate(words):
            norm = word.text.lower().strip()
            if norm not in text_to_indices:
                text_to_indices[norm] = []
            text_to_indices[norm].append(i)

        # Learn header field positions (invoice_number, invoice_date, invoice_amount)
        self._learn_header_field(words, text_to_indices, 'invoice_number', str(ground_truth.get('invoice_number', '')))
        self._learn_header_field(words, text_to_indices, 'invoice_date', str(ground_truth.get('invoice_date', '')))
        # For invoice_amount, we use special context-aware learning (looks for 'total' label)
        self._learn_amount_field(words, text_to_indices, ground_truth.get('invoice_amount', 0))

        # Learn table structure from line items
        self._learn_table_template(words, ground_truth.get('items', []))

    def _find_value_near_label(self, words: list, label_bbox: BBox, value_text: str, label_idx: int) -> tuple:
        """
        Find the value bounding box near a label.

        Searches for the OCR word that corresponds to the ground truth value
        by checking if any word matches or partially matches the value text.
        Among matches, selects the one closest to the label position.

        The search considers three positional relationships:
        - 'right': Same line, to the right of label (common for "Invoice No.: VALUE")
        - 'left': Same line, to the left of label (unusual but possible)
        - 'below': On the next line below the label

        Args:
            words: List of all BBox objects
            label_bbox: Bounding box of the label word
            value_text: The ground truth value text to find
            label_idx: Index of the label word in words list

        Returns:
            Tuple of (value_bounding_box, position_string) or (None, None) if not found
        """
        if not value_text:
            return None, None

        value_text_lower = value_text.lower().strip()

        # Find all words that could be the value (fuzzy matching)
        candidates = []
        for i, word in enumerate(words):
            if i == label_idx:
                continue
            word_lower = word.text.lower().strip()

            # Exact match, contains match, or substring match
            if value_text_lower == word_lower or value_text_lower in word_lower or word_lower in value_text_lower:
                candidates.append((word, i))

        if not candidates:
            return None, None

        best_candidate = None
        best_distance = float('inf')
        position_type = None

        for candidate, _ in candidates:
            # Check if it's to the RIGHT (same line) - most common layout
            if abs(candidate.top - label_bbox.top) < label_bbox.height * 0.6:
                if candidate.left > label_bbox.right:
                    dist = candidate.left - label_bbox.right
                    if dist < best_distance:
                        best_distance = dist
                        best_candidate = candidate
                        position_type = 'right'
                # Also check if value is to the LEFT of label (less common)
                elif candidate.right < label_bbox.left:
                    dist = label_bbox.left - candidate.right
                    if dist < best_distance:
                        best_distance = dist
                        best_candidate = candidate
                        position_type = 'left'

            # Check if it's BELOW (next line) - common for totals
            elif candidate.top > label_bbox.bottom:
                if candidate.left <= label_bbox.center_x <= candidate.right or abs(candidate.left - label_bbox.left) < label_bbox.width * 2:
                    dist = candidate.top - label_bbox.bottom
                    if dist < best_distance:
                        best_distance = dist
                        best_candidate = candidate
                        position_type = 'below'

        return best_candidate, position_type

    def _learn_header_field(self, words: list, text_to_indices: dict, field_name: str, ground_truth_value: str):
        """
        Learn position template for a header field.

        Searches for label words matching known patterns for the field,
        then finds the actual value word near the label and records
        the spatial offset between them.

        For invoice_number, handles multi-word labels like "Invoice Number" or "Invoice No."

        Args:
            words: List of BBox objects
            text_to_indices: Lookup from normalized text to word indices
            field_name: Name of field to learn ('invoice_number', 'invoice_date')
            ground_truth_value: The validated value for this field
        """
        # Patterns to search for in labels (ordered by specificity)
        label_patterns = {
            'invoice_number': ['invoice', 'inv', 'no.'],
            'invoice_date': ['date', 'dated', 'invoice date'],
        }

        # Find the label - handle multi-word labels first (e.g., "Invoice No.")
        label_word = None
        label_idx = -1
        label_text = ""

        # For invoice_number, look for "Invoice No." or "Invoice Number" as a pair
        if field_name == 'invoice_number':
            for i in range(len(words) - 1):
                two_words = (words[i].text.lower() + ' ' + words[i+1].text.lower()).strip()
                # Check for "invoice no" or "invoice number" patterns
                if 'invoice' in two_words and ('no' in two_words or 'number' in two_words):
                    label_word = words[i+1]   # Use second word as anchor
                    label_idx = i + 1
                    label_text = two_words
                    break

        if label_word is None:
            # General label search - look for any pattern match
            for norm_text, indices in text_to_indices.items():
                for pattern in label_patterns.get(field_name, []):
                    if pattern in norm_text:
                        for idx in indices:
                            # Skip "invoice" word itself for invoice_number
                            if field_name == 'invoice_number' and words[idx].text.lower() in ['invoice', 'inv']:
                                continue
                            label_word = words[idx]
                            label_idx = idx
                            label_text = norm_text
                            break
                        if label_word:
                            break
                if label_word:
                    break

        if label_word is None:
            return  # Could not find label for this field

        # Find where the actual value is relative to the label
        value_bbox, position = self._find_value_near_label(words, label_word, ground_truth_value, label_idx)

        if value_bbox and position:
            # Calculate offset from label to value
            if position == 'right':
                offset_x = value_bbox.left - label_word.right
                offset_y = value_bbox.top - label_word.top
            elif position == 'left':
                offset_x = value_bbox.right - label_word.left
                offset_y = value_bbox.top - label_word.top
            else:  # 'below'
                offset_x = value_bbox.left - label_word.left
                offset_y = value_bbox.top - label_word.bottom

            # Validate invoice_number should contain digits
            if field_name == 'invoice_number':
                if not any(c.isdigit() for c in value_bbox.text):
                    return

            # Create or update template (averaging offsets over multiple invoices)
            if field_name not in self.templates:
                self.templates[field_name] = FieldTemplate(
                    field_name=field_name,
                    label_texts=[],
                    label_x=label_word.left,
                    label_y=label_word.top,
                    value_offset_x=offset_x,
                    value_offset_y=offset_y,
                    value_position=position,
                    confidence_boost=0.0
                )
            else:
                template = self.templates[field_name]
                # Running average for position (more samples = more accurate)
                template.label_x = int((template.label_x + label_word.left) / 2)
                template.label_y = int((template.label_y + label_word.top) / 2)
                template.value_offset_x = int((template.value_offset_x + offset_x) / 2)
                template.value_offset_y = int((template.value_offset_y + offset_y) / 2)
                template.value_position = position if position else template.value_position

            # Track new label text variations
            if label_text not in self.templates[field_name].label_texts:
                self.templates[field_name].label_texts.append(label_text)

            # Confidence increases with each validated invoice (capped at 0.9)
            self.templates[field_name].confidence_boost = min(self.templates[field_name].confidence_boost + 0.15, 0.9)

    def _learn_amount_field(self, words: list, text_to_indices: dict, ground_truth_value: float):
        """
        Learn invoice amount field position - looks for 'total' label and value below it.

        The amount field uses special heuristics since "Total" can appear with
        various labels (Grand Total, Total Due, Balance Due). It finds the "total"
        keyword and looks for amounts in the expected value range.
        """
        # Find "total" label - look for common variants
        total_label = None
        for word in words:
            if word.text.lower() in ['total', 'amount', 'due', 'balance', 'grand']:
                total_label = word
                break

        if total_label is None:
            return

        # Find amounts near the total label that match ground truth
        candidates = []
        for word in words:
            amt = self._parse_amount(word.text)
            if amt and abs(amt - float(ground_truth_value)) < 1:  # Match ground truth value
                candidates.append((word, abs(word.top - total_label.top)))

        if candidates:
            # Pick the one closest to the label
            candidates.sort(key=lambda x: x[1])
            value_bbox = candidates[0][0]
            offset_y = value_bbox.top - total_label.bottom

            self.templates['invoice_amount'] = FieldTemplate(
                field_name='invoice_amount',
                label_texts=[total_label.text.lower()],
                label_x=total_label.left,
                label_y=total_label.top,
                value_offset_x=0,
                value_offset_y=offset_y,  # Use actual offset
                value_position='below',
                confidence_boost=0.6
            )

    def _learn_table_template(self, words: list, items: list):
        """
        Learn table structure from line items.

        Groups words into lines, identifies rows with numeric data,
        and maps each numeric column to its semantic meaning (qty, price, total).

        For each ground truth item, finds the corresponding row by matching
        the known total price value, then deduces column positions from
        the other values in that row.
        """
        if not items:
            return

        lines = self._group_words_by_line(words)

        # Find all rows with at least 2 numbers (likely data rows, not headers)
        candidate_rows = []
        for line_words in lines:
            numbers_in_line = []
            for word in line_words:
                num_match = re.search(r'[\d,]+\.?\d*', word.text)
                if num_match:
                    try:
                        val = float(num_match.group().replace(',', ''))
                        if val > 0:
                            numbers_in_line.append((word, val))
                    except ValueError:
                        pass

            if len(numbers_in_line) >= 2:
                candidate_rows.append({
                    'words': line_words,
                    'numbers': numbers_in_line,
                    'y': line_words[0].top if line_words else 0
                })

        if not candidate_rows:
            return

        # For each ground truth item, find the matching row by total price
        item_no_positions = []
        item_name_positions = []
        qty_positions = []
        price_positions = []
        total_positions = []
        matched_rows_y = []  # Track which rows actually matched items

        for gt_item in items:
            gt_qty = gt_item.get('item_quantity', 0)
            gt_price = gt_item.get('per_item_price', 0)
            gt_total = gt_item.get('total_item_price', 0)

            best_match = None
            best_total_match_idx = -1
            best_total_dist = float('inf')

            # First pass: find row where TOTAL matches exactly
            for row in candidate_rows:
                for j, (num_word, val) in enumerate(row['numbers']):
                    if abs(val - gt_total) < 1:
                        dist = abs(val - gt_total)
                        if dist < best_total_dist:
                            best_total_dist = dist
                            best_match = row
                            best_total_match_idx = j

            # Only use this row if we found a total match
            if best_match:
                matched_rows_y.append(best_match['y'])
                numbers = best_match['numbers']
                words_row = best_match['words']
                assigned_x = set()

                # Total position is where we found the match
                total_word = numbers[best_total_match_idx][0]
                total_positions.append(total_word.left)
                assigned_x.add(total_word.left)

                # Find price and qty in the SAME row
                for num_word, val in numbers:
                    if num_word.left in assigned_x:
                        continue
                    # Price: look for value matching per_item_price (tolerance 0.1)
                    if abs(val - gt_price) < 0.1:
                        price_positions.append(num_word.left)
                        assigned_x.add(num_word.left)
                        break

                for num_word, val in numbers:
                    if num_word.left in assigned_x:
                        continue
                    # Qty: look for value matching item_quantity
                    # Qty is typically a small integer (1, 10, 100)
                    if abs(val - gt_qty) < 0.1:
                        qty_positions.append(num_word.left)
                        assigned_x.add(num_word.left)
                        break

                # If qty not found yet, look for smallest reasonable qty value
                if not any(num_word.left in assigned_x for num_word, _ in numbers if abs(num_word.left - qty_positions[0]) < 50 if qty_positions):
                    for num_word, val in numbers:
                        if num_word.left in assigned_x:
                            continue
                        # Qty is typically 1-10000 and often a round number
                        if 1 <= val <= 50000 and val == int(val):
                            qty_positions.append(num_word.left)
                            assigned_x.add(num_word.left)
                            break

                # Item name is text before the first number in the row
                if numbers:
                    first_num_x = numbers[0][0].left
                    for w in words_row:
                        if w.left < first_num_x:
                            item_name_positions.append(w.left)

        # Build table template from matched rows
        if matched_rows_y:
            start_y = min(matched_rows_y)
            last_y = max(matched_rows_y)
            # Calculate average row height from matched rows
            row_height = (last_y - start_y) / max(len(matched_rows_y) - 1, 1) if len(matched_rows_y) > 1 else 30

            self.table_template = TableTemplate(
                start_y=start_y,
                row_height=max(int(row_height), 20),
                column_positions={},
                item_no_col=int(np.median(item_no_positions)) if item_no_positions else 50,
                item_name_col=int(np.median(item_name_positions)) if item_name_positions else 150,
                qty_col=int(np.median(qty_positions)) if qty_positions else 400,
                price_col=int(np.median(price_positions)) if price_positions else 500,
                total_col=int(np.median(total_positions)) if total_positions else 600
            )

    def _group_words_by_line(self, words: list) -> list:
        """
        Group words into lines based on vertical position.

        Words with Y coordinates within 15 pixels of each other are
        considered on the same line. Lines are sorted by X position.

        Args:
            words: List of BBox objects

        Returns:
            List of lines, where each line is a list of BBox sorted by X
        """
        if not words:
            return []

        sorted_words = sorted(words, key=lambda w: (w.top, w.left))
        lines = []
        current_line = [sorted_words[0]]
        current_baseline = sorted_words[0].top

        for word in sorted_words[1:]:
            # Words within 15 pixels of baseline are on the same line
            if abs(word.top - current_baseline) < 15:
                current_line.append(word)
            else:
                lines.append(sorted(current_line, key=lambda w: w.left))
                current_line = [word]
                current_baseline = word.top

        if current_line:
            lines.append(sorted(current_line, key=lambda w: w.left))

        return lines

    def _parse_amount(self, text: str) -> Optional[float]:
        """Parse amount from text, removing currency symbols and commas."""
        cleaned = re.sub(r'[$€£,]', '', text)
        match = re.search(r'([\d,]+\.?\d*)', cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        return None

    def save_template(self):
        """
        Save learned template to disk.

        Stores the template as JSON in the vendor's directory.
        Called automatically after learning from a validated invoice.
        """
        template_path = self.vendor_dir / "position_template.json"
        template_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'templates': {k: asdict(v) for k, v in self.templates.items()},
            'table_template': asdict(self.table_template) if self.table_template else None,
            'image_width': self.image_width,
            'image_height': self.image_height
        }

        with open(template_path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_template(cls, vendor_dir: Path) -> Optional['VendorTemplateLearner']:
        """
        Load existing template from disk.

        Args:
            vendor_dir: Path to vendor's directory

        Returns:
            VendorTemplateLearner instance with loaded template,
            or None if no template exists
        """
        template_path = vendor_dir / "position_template.json"
        if not template_path.exists():
            return None

        with open(template_path, 'r') as f:
            data = json.load(f)

        learner = cls.__new__(cls)
        learner.vendor_id = vendor_dir.name
        learner.vendor_dir = vendor_dir
        learner.templates = {k: FieldTemplate(**v) for k, v in data.get('templates', {}).items()}
        learner.table_template = TableTemplate(**data['table_template']) if data.get('table_template') else None
        learner.image_width = data.get('image_width', 0)
        learner.image_height = data.get('image_height', 0)

        return learner


class PositionBasedExtractor:
    """
    Extracts invoice fields using learned vendor-specific positions.

    This is the extraction counterpart to VendorTemplateLearner.
    Given a learned template, it uses the stored positions and offsets
    to accurately extract field values from new invoice images.

    For each field, it:
    1. Finds the label at the learned position (or by text search)
    2. Applies the learned offset to locate the value
    3. Extracts and validates the value

    This is more accurate than generic extraction because it accounts
    for each vendor's unique invoice layout.
    """

    def __init__(self, words: list, template: VendorTemplateLearner):
        """
        Initialize extractor with OCR words and learned template.

        Args:
            words: List of OCR word dicts with position info
            template: Learned VendorTemplateLearner instance
        """
        # Convert to BBox objects for easier manipulation
        self.words = [BBox(
            text=w['text'],
            left=int(w['left']),
            top=int(w['top']),
            width=int(w['width']),
            height=int(w['height']),
            conf=float(w.get('conf', 50.0))
        ) for w in words if w.get('text', '').strip()]

        self.template = template
        self._build_word_lookup()

    def _build_word_lookup(self):
        """
        Build lookup tables for fast word access.

        Creates a mapping from normalized (lowercase, stripped) text
        to list of word indices. Allows O(1) lookup when searching for labels.
        """
        self._text_to_idx = {}
        for i, word in enumerate(self.words):
            norm = word.text.lower().strip()
            if norm not in self._text_to_idx:
                self._text_to_idx[norm] = []
            self._text_to_idx[norm].append(i)

    def extract(self) -> dict:
        """
        Extract all fields using the learned template.

        Returns:
            Dict with invoice_number, invoice_date, invoice_amount, and items
        """
        predictions = {}

        # Extract header fields using learned templates
        for field_name in ['invoice_number', 'invoice_date']:
            if field_name in self.template.templates:
                value = self._extract_field(field_name, self.template.templates[field_name])
                predictions[field_name] = value
            else:
                predictions[field_name] = None

        # Invoice amount uses special extraction (finds largest amount near 'total')
        predictions['invoice_amount'] = self._extract_amount_with_context()

        # Extract line items using learned table structure
        predictions['items'] = self._extract_items()

        return predictions

    def _extract_field(self, field_name: str, field_template: FieldTemplate) -> any:
        """
        Extract a single field using its learned template.

        First tries to find the label at the learned position, then
        falls back to text search. Finally applies the learned offset
        to find the value.

        Args:
            field_name: Name of field to extract
            field_template: Learned FieldTemplate with position info

        Returns:
            Extracted value, or None if not found
        """
        label_word = self._find_label_at_position(field_template)

        if label_word is None:
            label_word = self._find_label_by_text(field_template)

        if label_word is None:
            return None

        value_word = self._find_value_at_offset(label_word, field_template)

        if value_word is None:
            return None

        raw_value = value_word.text
        if field_name == 'invoice_number':
            return raw_value
        elif field_name == 'invoice_date':
            return self._parse_date(raw_value)

        return raw_value

    def _find_label_at_position(self, field_template: FieldTemplate):
        """
        Find label near the learned position.

        Searches within a radius of 100 pixels from the learned position
        for words matching expected label patterns.

        Args:
            field_template: FieldTemplate with learned label position

        Returns:
            BBox of label word, or None if not found within radius
        """
        target_x = field_template.label_x
        target_y = field_template.label_y

        best_match = None
        best_dist = float('inf')
        search_radius = 100

        for word in self.words:
            dist = ((word.left - target_x) ** 2 + (word.top - target_y) ** 2) ** 0.5
            if dist < best_dist and dist < search_radius:
                word_lower = word.text.lower()
                if any(label in word_lower for label in ['invoice', 'inv', 'no', '#', 'date']):
                    best_dist = dist
                    best_match = word

        return best_match

    def _find_label_by_text(self, field_template: FieldTemplate):
        """
        Find label by searching for known label text patterns.

        Uses keyword matching when position search fails.

        Args:
            field_template: FieldTemplate with field_name

        Returns:
            BBox of first matching label, or None
        """
        label_keywords = {
            'invoice_number': ['invoice', 'inv', 'no'],
            'invoice_date': ['date', 'dated'],
            'invoice_amount': ['total', 'tot', 'amount', 'amt', 'grand'],
        }

        keywords = label_keywords.get(field_template.field_name, [])

        for norm_text, indices in self._text_to_idx.items():
            for keyword in keywords:
                if keyword in norm_text:
                    return self.words[indices[0]]

        return None

    def _find_value_at_offset(self, label_word: BBox, field_template: FieldTemplate):
        """
        Find value at the learned offset from label.

        Uses the learned offset and direction to locate the value word.

        Args:
            label_word: BBox of the label word
            field_template: FieldTemplate with offset and direction

        Returns:
            BBox of value word, or None if not found
        """
        position = field_template.value_position
        offset_x = field_template.value_offset_x
        offset_y = field_template.value_offset_y

        candidates = []

        for word in self.words:
            if word is label_word:
                continue

            if position == 'right':
                if abs(word.top - label_word.top) < label_word.height * 0.6:
                    expected_x = label_word.right + offset_x
                    if word.left >= label_word.right:
                        dist = abs(word.left - expected_x)
                        candidates.append((word, dist, 'right'))

            elif position == 'left':
                if abs(word.top - label_word.top) < label_word.height * 0.6:
                    expected_x = label_word.left + offset_x
                    if word.right <= label_word.left:
                        dist = abs(word.right - expected_x)
                        candidates.append((word, dist, 'left'))

            elif position == 'below':
                if word.top >= label_word.bottom + offset_y - 20:
                    expected_x = label_word.left + offset_x
                    dist = abs(word.top - (label_word.bottom + offset_y)) + abs(word.left - expected_x)
                    candidates.append((word, dist, 'below'))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def _extract_amount_with_context(self):
        """
        Extract invoice amount using label context + largest amount heuristic.

        First looks for "total" keyword and finds amounts below it.
        If no label found, uses the largest amount in the lower half of invoice.
        """
        total_word = None
        for word in self.words:
            if word.text.lower() in ['total', 'tot', 'amount', 'amt', 'grand']:
                total_word = word
                break

        if total_word is None:
            # Fallback: find largest amount in lower half of invoice
            mid_y = self.words[-1].top // 2 if self.words else 500
            amounts = []
            for word in self.words:
                if word.top > mid_y:
                    amt = self._parse_amount(word.text)
                    if amt and amt > 100:
                        amounts.append(amt)
            if amounts:
                return max(amounts)
            return None

        # Find amounts below or to the right of "total"
        candidates = []
        for word in self.words:
            amt = self._parse_amount(word.text)
            if amt and amt > 100:
                if word.top > total_word.top - 20 or word.left > total_word.left:
                    candidates.append((word, amt))

        if not candidates:
            return None

        # Prefer amounts below the label
        below_candidates = [(w, a) for w, a in candidates if w.top > total_word.top + 5]

        if below_candidates:
            return max(a for _, a in below_candidates)

        return max(a for _, a in candidates) if candidates else None

    def _extract_items(self) -> list:
        """
        Extract line items using learned table structure.

        Uses the learned column positions and row height to identify
        and parse each row in the table.

        Returns:
            List of item dicts with item_no, item_name, qty, price, total
        """
        if not self.template.table_template:
            return []

        table = self.template.table_template
        items = []

        lines = self._group_words_by_line()

        for line_words in lines:
            line_y = line_words[0].top

            # Check if line is at table start or in expected row position
            if abs(line_y - table.start_y) < table.row_height * 0.5:
                item = self._parse_table_row(line_words, table)
                if item:
                    items.append(item)
            elif line_y > table.start_y:
                expected_row = table.start_y + (len(items)) * table.row_height
                if abs(line_y - expected_row) < table.row_height * 0.7:
                    item = self._parse_table_row(line_words, table)
                    if item:
                        items.append(item)

        return items

    def _parse_table_row(self, line_words: list, table: TableTemplate):
        """Parse a single table row into item data."""
        if not line_words:
            return None

        # Get full line text to check for non-item keywords
        line_text = ' '.join(w.text.lower() for w in line_words)

        # Skip rows that are purely tax/subtotal/header rows with NO real item name
        # A row like "Net Amount: 5000" should be skipped
        # But "Dressing Table 500 CGST 14%" should NOT be skipped

        # Non-numeric words that indicate this is a header/total row
        header_keywords = {'net', 'amount', 'subtotal', 'grand', 'total', 'balance', 'due',
                          'item', 'description', 'quantity', 'unit', 'price'}
        # Tax keywords that should not be considered as "item names"
        tax_keywords = {'cgst', 'sgst', 'igst', 'gst', 'tax', 'taxes'}

        # Non-numeric words in this line
        line_non_numeric_words = set()
        for w in line_words:
            word_lower = w.text.lower()
            # Remove punctuation
            word_clean = re.sub(r'[^\w]', '', word_lower)
            if word_clean and not word_clean.isdigit():
                line_non_numeric_words.add(word_clean)

        # Meaningful words = non-numeric words that are NOT header or tax keywords
        meaningful_words = line_non_numeric_words - header_keywords - tax_keywords

        # If there are no meaningful words AND the row has tax/header keywords, skip it
        if not meaningful_words and line_non_numeric_words:
            return None

        # Check if line has a meaningful item name (at least 3 letters)
        has_item_name = any(len(w.text) >= 3 and not w.text.isdigit() for w in line_words)
        if not has_item_name:
            return None

        # Skip if the line is more than 90% numeric (likely not an item row)
        word_count = len(line_words)
        numeric_words = sum(1 for w in line_words if re.match(r'^[\d,\.]+$', w.text))
        if numeric_words >= word_count * 0.9:
            return None

        numbers = []
        item_name_parts = []
        item_no = None

        for word in sorted(line_words, key=lambda w: w.left):
            num_match = re.search(r'[\d,]+\.?\d*', word.text)
            if num_match:
                try:
                    val = float(num_match.group().replace(',', ''))
                    if val > 0:
                        # Check if this could be item_no (small integer at start of line)
                        if item_no is None and val < 100 and val == int(val) and word.left < 300:
                            item_no = int(val)
                        else:
                            numbers.append({'word': word, 'value': val, 'x': word.left})
                except ValueError:
                    pass
            else:
                word_lower = word.text.lower()
                if word_lower not in ['qty', 'quantity', 'unit', 'price', 'total', 'x', 'ea', 'each', 'pcs', 'no', 'item']:
                    item_name_parts.append(word.text)

        if not numbers:
            return None

        item_name = ' '.join(item_name_parts).strip() or 'Unknown Item'

        # Skip if item name is too short or looks like a total row
        if len(item_name) < 2:
            return None

        qty = 1
        unit_price = 0.0
        total_price = 0.0

        numbers.sort(key=lambda x: x['x'])

        # Assign numbers to columns based on position relative to learned columns
        col_qty = table.qty_col
        col_price = table.price_col
        col_total = table.total_col

        qty = 1
        unit_price = 0.0
        total_price = 0.0

        # Find the closest number to each learned column position
        for num_data in numbers:
            x = num_data['x']
            val = num_data['value']

            # Calculate distance to each column
            dist_to_qty = abs(x - col_qty)
            dist_to_price = abs(x - col_price)
            dist_to_total = abs(x - col_total)

            # Find the closest column
            min_dist = min(dist_to_qty, dist_to_price, dist_to_total)

            # Assign based on closest column (with tolerance)
            if min_dist < 100:  # Within 100 pixels
                if min_dist == dist_to_qty:
                    qty = int(val) if val == int(val) and val < 10000 else 1
                elif min_dist == dist_to_price:
                    unit_price = val
                elif min_dist == dist_to_total:
                    total_price = val

        # Calculate missing values
        if unit_price == 0 and total_price > 0 and qty > 0:
            unit_price = total_price / qty

        if total_price == 0 and unit_price > 0 and qty > 0:
            total_price = unit_price * qty

        if item_no is None and numbers:
            item_no = len(numbers)

        return {
            'item_no': item_no,
            'item_name': item_name.title(),
            'item_quantity': qty,
            'per_item_price': round(unit_price, 2),
            'total_item_price': round(total_price, 2)
        }

    def _group_words_by_line(self) -> list:
        """Group words into lines."""
        if not self.words:
            return []

        sorted_words = sorted(self.words, key=lambda w: (w.top, w.left))
        lines = []
        current_line = [sorted_words[0]]
        current_baseline = sorted_words[0].top

        for word in sorted_words[1:]:
            if abs(word.top - current_baseline) < 15:
                current_line.append(word)
            else:
                lines.append(sorted(current_line, key=lambda w: w.left))
                current_line = [word]
                current_baseline = word.top

        if current_line:
            lines.append(sorted(current_line, key=lambda w: w.left))

        return lines

    def _parse_date(self, text: str):
        """Parse date from text."""
        patterns = [
            (r'\d{2}/\d{2}/\d{4}', 'MM/DD/YYYY'),
            (r'\d{2}-\d{2}-\d{4}', 'MM-DD-YYYY'),
            (r'\d{4}-\d{2}-\d{2}', 'YYYY-MM-DD'),
        ]
        for pattern, _ in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()
        return text if text else None

    def _parse_amount(self, text: str) -> Optional[float]:
        """Parse amount from text."""
        cleaned = re.sub(r'[$€£,]', '', text)
        match = re.search(r'([\d,]+\.?\d*)', cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        return None


def learn_vendor_template(
    vendor_id: str,
    vendor_dir: Path,
    validated_invoices: list
) -> VendorTemplateLearner:
    """
    Learn a vendor template from validated invoices.

    Args:
        vendor_id: Vendor identifier
        vendor_dir: Path to vendor directory
        validated_invoices: List of {
            ocr_words: [...],
            ground_truth: {...},
            image_width: int,
            image_height: int
        }

    Returns:
        VendorTemplateLearner with learned positions
    """
    learner = VendorTemplateLearner(vendor_id, vendor_dir)

    for invoice_data in validated_invoices:
        learner.learn_from_invoice(
            ocr_words=invoice_data['ocr_words'],
            ground_truth=invoice_data['ground_truth'],
            image_width=invoice_data.get('image_width', 800),
            image_height=invoice_data.get('image_height', 1000)
        )

    learner.save_template()
    return learner


def extract_with_template(
    ocr_words: list,
    vendor_dir: Path
) -> Optional[dict]:
    """
    Extract fields using learned vendor template.

    Args:
        ocr_words: List of {text, left, top, width, height, conf}
        vendor_dir: Path to vendor directory

    Returns:
        Extracted predictions dict or None if no template exists
    """
    learner = VendorTemplateLearner.load_template(vendor_dir)
    if learner is None:
        return None

    extractor = PositionBasedExtractor(ocr_words, learner)
    return extractor.extract()
