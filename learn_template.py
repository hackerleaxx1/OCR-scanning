"""
Script to learn vendor templates from validated invoices.

Usage:
    python learn_template.py

This will:
1. Load PDFs and their corresponding JSON ground truth
2. Run OCR to get word positions
3. Learn position templates for the vendor
4. Save the template to vendors/<vendor_id>/position_template.json
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io


@dataclass
class BBox:
    """Bounding box for a word."""
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float = 50.0

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center_x(self) -> int:
        return self.left + self.width // 2

    @property
    def center_y(self) -> int:
        return self.top + self.height // 2


@dataclass
class FieldTemplate:
    """Learned template for a single field."""
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
    """Learned template for line item table."""
    start_y: int
    row_height: int
    column_positions: dict
    item_no_col: int
    item_name_col: int
    qty_col: int
    price_col: int
    total_col: int


def _convert_pdf_to_images(pdf_path: str) -> list:
    """Convert PDF pages to PIL Images."""
    images = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)
    doc.close()
    return images


def _extract_words_from_image(img: Image.Image) -> tuple:
    """Extract words and text from a PIL Image."""
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
                'conf': float(data['conf'][i]) if data['conf'][i] != '-1' else 50.0
            })
            text_parts.append(text)

    return words, ' '.join(text_parts)


def extract_pdf_words(pdf_path: Path) -> tuple:
    """Extract words with bounding boxes from PDF."""
    images = _convert_pdf_to_images(str(pdf_path))
    all_words = []
    width, height = 0, 0

    for img in images:
        words, text = _extract_words_from_image(img)
        all_words.extend(words)
        width, height = img.size

    return all_words, width, height


class VendorTemplateLearner:
    """
    Learns vendor-specific invoice templates from validated invoices.
    """

    def __init__(self, vendor_id: str, vendor_dir: Path):
        self.vendor_id = vendor_id
        self.vendor_dir = Path(vendor_dir)
        self.templates: dict = {}
        self.table_template: TableTemplate = None
        self.image_width: int = 0
        self.image_height: int = 0

    def learn_from_invoice(self, ocr_words: list, ground_truth: dict, image_width: int, image_height: int):
        """Learn field positions from a validated invoice."""
        self.image_width = image_width
        self.image_height = image_height

        words = [BBox(
            text=w['text'],
            left=int(w['left']),
            top=int(w['top']),
            width=int(w['width']),
            height=int(w['height']),
            conf=float(w.get('conf', 50.0))
        ) for w in ocr_words if w.get('text', '').strip()]

        # Build word lookup
        text_to_indices = {}
        for i, word in enumerate(words):
            norm = word.text.lower().strip()
            if norm not in text_to_indices:
                text_to_indices[norm] = []
            text_to_indices[norm].append(i)

        # Learn header field positions
        self._learn_header_field(words, text_to_indices, 'invoice_number', str(ground_truth.get('invoice_number', '')))
        self._learn_header_field(words, text_to_indices, 'invoice_date', str(ground_truth.get('invoice_date', '')))
        self._learn_header_field(words, text_to_indices, 'invoice_amount', str(ground_truth.get('invoice_amount', '')))

        # Learn table structure from items
        self._learn_table_template(words, ground_truth.get('items', []))

    def _find_value_near_label(self, words: list, label_bbox: BBox, value_text: str, label_idx: int) -> tuple:
        """Find the value bounding box near a label."""
        if not value_text:
            return None, None

        value_text_lower = value_text.lower().strip()

        # Find words that could be the value
        candidates = []
        for i, word in enumerate(words):
            if i == label_idx:
                continue
            word_lower = word.text.lower().strip()

            if value_text_lower == word_lower or value_text_lower in word_lower or word_lower in value_text_lower:
                candidates.append((word, i))

        if not candidates:
            return None, None

        best_candidate = None
        best_distance = float('inf')
        position_type = None

        for candidate, _ in candidates:
            # Check if it's to the right (same line)
            if abs(candidate.top - label_bbox.top) < label_bbox.height * 0.6:
                if candidate.left > label_bbox.right:
                    dist = candidate.left - label_bbox.right
                    if dist < best_distance:
                        best_distance = dist
                        best_candidate = candidate
                        position_type = 'right'
                # Also check if value is to the LEFT of label (same line)
                elif candidate.right < label_bbox.left:
                    dist = label_bbox.left - candidate.right
                    if dist < best_distance:
                        best_distance = dist
                        best_candidate = candidate
                        position_type = 'left'

            # Check if it's below (next line)
            elif candidate.top > label_bbox.bottom:
                if candidate.left <= label_bbox.center_x <= candidate.right or abs(candidate.left - label_bbox.left) < label_bbox.width * 2:
                    dist = candidate.top - label_bbox.bottom
                    if dist < best_distance:
                        best_distance = dist
                        best_candidate = candidate
                        position_type = 'below'

        return best_candidate, position_type

    def _learn_header_field(self, words: list, text_to_indices: dict, field_name: str, ground_truth_value: str):
        """Learn position template for a header field."""
        label_patterns = {
            'invoice_number': ['invoice', 'inv', 'no.'],
            'invoice_date': ['date', 'dated', 'invoice date'],
            'invoice_amount': ['total', 'amount', 'due', 'balance', 'grand'],
        }

        # Find the label - look for invoice header first
        label_word = None
        label_idx = -1
        label_text = ""

        # For invoice_number, look for "Invoice No." as a pair or "INVOICE" header
        if field_name == 'invoice_number':
            # Look for multi-word label "Invoice No."
            for i in range(len(words) - 1):
                two_words = (words[i].text.lower() + ' ' + words[i+1].text.lower()).strip()
                if 'invoice' in two_words and 'no' in two_words:
                    label_word = words[i+1]  # Use the second word as anchor
                    label_idx = i + 1
                    label_text = two_words
                    break

            # If not found, try "INVOICE" header
            if label_word is None:
                for i, w in enumerate(words):
                    if w.text.upper() == 'INVOICE' and i + 1 < len(words):
                        # Invoice number might be the next word after INVOICE
                        label_word = words[i]
                        label_idx = i
                        label_text = 'invoice'
                        break

        if label_word is None:
            # General label search for other fields
            for norm_text, indices in text_to_indices.items():
                for pattern in label_patterns.get(field_name, []):
                    if pattern in norm_text:
                        for idx in indices:
                            if field_name == 'invoice_number' and words[idx].text.lower() in ['invoice', 'inv']:
                                continue  # Skip standalone invoice/inv for invoice_number
                            label_word = words[idx]
                            label_idx = idx
                            label_text = norm_text
                            break
                        if label_word:
                            break
                if label_word:
                    break

        if label_word is None:
            print(f"  Warning: Could not find label for {field_name}")
            return

        # Find where the actual value is
        value_bbox, position = self._find_value_near_label(words, label_word, ground_truth_value, label_idx)

        if value_bbox and position:
            if position == 'right':
                offset_x = value_bbox.left - label_word.right
                offset_y = value_bbox.top - label_word.top
            elif position == 'left':
                offset_x = value_bbox.right - label_word.left
                offset_y = value_bbox.top - label_word.top
            else:  # below
                offset_x = value_bbox.left - label_word.left
                offset_y = value_bbox.top - label_word.bottom

            # Validate extracted position makes sense for invoice_number
            if field_name == 'invoice_number':
                # Invoice number should be a numeric string like "100001"
                if not any(c.isdigit() for c in value_bbox.text):
                    print(f"  Warning: Invoice number value '{value_bbox.text}' doesn't contain digits, skipping")
                    return

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
                template.label_x = int((template.label_x + label_word.left) / 2)
                template.label_y = int((template.label_y + label_word.top) / 2)
                template.value_offset_x = int((template.value_offset_x + offset_x) / 2)
                template.value_offset_y = int((template.value_offset_y + offset_y) / 2)
                template.value_position = position if position else template.value_position

            if label_text not in self.templates[field_name].label_texts:
                self.templates[field_name].label_texts.append(label_text)

            self.templates[field_name].confidence_boost = min(self.templates[field_name].confidence_boost + 0.15, 0.9)

            print(f"  {field_name}: label at ({label_word.left}, {label_word.top}), value at ({value_bbox.left}, {value_bbox.top}), offset ({offset_x}, {offset_y}) [{position}]")
        else:
            print(f"  Warning: Could not find value position for {field_name}")

    def _learn_table_template(self, words: list, items: list):
        """Learn table structure from line items."""
        if not items:
            return

        # Group words by lines
        lines = self._group_words_by_line(words)

        # Find rows that match item patterns
        item_rows = []
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
                item_rows.append({
                    'words': line_words,
                    'numbers': numbers_in_line,
                    'y': line_words[0].top if line_words else 0
                })

        if not item_rows:
            return

        # Learn column positions
        item_no_positions = []
        item_name_positions = []
        qty_positions = []
        price_positions = []
        total_positions = []

        for i, row in enumerate(item_rows):
            if i >= len(items):
                break

            gt_item = items[i]
            gt_qty = gt_item.get('item_quantity', 0)
            gt_price = gt_item.get('per_item_price', 0)
            gt_total = gt_item.get('total_item_price', 0)

            numbers = row['numbers']
            words_row = row['words']

            # Track which numbers we've assigned
            assigned_x = set()

            for num_word, val in numbers:
                # Check if this matches total
                if abs(val - gt_total) < 1 and num_word.left not in assigned_x:
                    total_positions.append(num_word.left)
                    assigned_x.add(num_word.left)

            for num_word, val in numbers:
                # Check if this matches unit price
                if abs(val - gt_price) < 1 and num_word.left not in assigned_x:
                    price_positions.append(num_word.left)
                    assigned_x.add(num_word.left)

            for num_word, val in numbers:
                # Check if this matches qty
                if abs(val - gt_qty) < 0.1 and num_word.left not in assigned_x:
                    qty_positions.append(num_word.left)
                    assigned_x.add(num_word.left)

            # Item no is the first unassigned small number
            for num_word, val in numbers:
                if val < 100 and val == int(val) and num_word.left not in assigned_x:
                    item_no_positions.append(num_word.left)
                    assigned_x.add(num_word.left)

            # Item name is text before first number
            if numbers:
                first_num_x = numbers[0][0].left
                for w in words_row:
                    if w.left < first_num_x:
                        item_name_positions.append(w.left)

        if item_rows:
            first_row_y = item_rows[0]['y']
            last_row_y = item_rows[-1]['y']
            row_height = (last_row_y - first_row_y) / max(len(item_rows) - 1, 1) if len(item_rows) > 1 else 30

            self.table_template = TableTemplate(
                start_y=first_row_y,
                row_height=max(int(row_height), 20),
                column_positions={},
                item_no_col=int(np.median(item_no_positions)) if item_no_positions else 50,
                item_name_col=int(np.median(item_name_positions)) if item_name_positions else 150,
                qty_col=int(np.median(qty_positions)) if qty_positions else 400,
                price_col=int(np.median(price_positions)) if price_positions else 500,
                total_col=int(np.median(total_positions)) if total_positions else 600
            )

            print(f"  Table: start_y={first_row_y}, row_height={row_height:.0f}")
            print(f"    Cols: item_no={self.table_template.item_no_col}, name={self.table_template.item_name_col}, qty={self.table_template.qty_col}, price={self.table_template.price_col}, total={self.table_template.total_col}")

    def _group_words_by_line(self, words: list) -> list:
        """Group words into lines based on vertical position."""
        if not words:
            return []

        sorted_words = sorted(words, key=lambda w: (w.top, w.left))
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

    def save_template(self):
        """Save learned template to disk."""
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
    def load_template(cls, vendor_dir: Path) -> 'VendorTemplateLearner':
        """Load existing template from disk."""
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
    """Extracts invoice fields using learned vendor-specific positions."""

    def __init__(self, words: list, template: VendorTemplateLearner):
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
        """Build lookup tables for fast access."""
        self._text_to_idx = {}
        for i, word in enumerate(self.words):
            norm = word.text.lower().strip()
            if norm not in self._text_to_idx:
                self._text_to_idx[norm] = []
            self._text_to_idx[norm].append(i)

    def extract(self) -> dict:
        """Extract all fields using the learned template."""
        predictions = {}

        for field_name, field_template in self.template.templates.items():
            value = self._extract_field(field_name, field_template)
            predictions[field_name] = value

        predictions['items'] = self._extract_items()
        return predictions

    def _extract_field(self, field_name: str, field_template: FieldTemplate) -> any:
        """Extract a single field using its learned template."""
        if field_name == 'invoice_amount':
            return self._extract_amount_with_context()

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
        elif field_name == 'invoice_amount':
            return self._parse_amount(raw_value)

        return raw_value

    def _extract_amount_with_context(self):
        """Extract invoice amount using label context + largest amount heuristic."""
        # Find the word "total" or "amount" which marks the invoice total
        total_word = None
        for word in self.words:
            if word.text.lower() in ['total', 'tot', 'amount', 'amt', 'grand']:
                total_word = word
                break

        if total_word is None:
            # Fallback: find the largest amount in the lower half of the invoice
            mid_y = self.words[-1].top // 2 if self.words else 500
            amounts = []
            for word in self.words:
                if word.top > mid_y:  # Lower half
                    amt = self._parse_amount(word.text)
                    if amt and amt > 100:
                        amounts.append(amt)
            if amounts:
                return max(amounts)
            return None

        # Find amounts that are to the right or below the total label
        candidates = []
        for word in self.words:
            amt = self._parse_amount(word.text)
            if amt and amt > 100:
                # Must be below or to the right of "total" word
                if word.top > total_word.top - 20 or word.left > total_word.left:
                    candidates.append((word, amt))

        if not candidates:
            return None

        # Prefer amounts that are below the total label (on their own line)
        below_candidates = [(w, a) for w, a in candidates if w.top > total_word.top + 10]

        if below_candidates:
            # Return the largest amount below
            return max(a for _, a in below_candidates)

        # Fallback: largest amount to the right
        return max(a for _, a in candidates)

    def _find_label_at_position(self, field_template: FieldTemplate):
        """Find label near the learned position."""
        target_x = field_template.label_x
        target_y = field_template.label_y

        best_match = None
        best_dist = float('inf')
        search_radius = 100

        for word in self.words:
            dist = ((word.left - target_x) ** 2 + (word.top - target_y) ** 2) ** 0.5
            if dist < best_dist and dist < search_radius:
                word_lower = word.text.lower()
                if any(label in word_lower for label in ['invoice', 'inv', 'no', '#', 'date', 'total', 'amount', 'due']):
                    best_dist = dist
                    best_match = word

        return best_match

    def _find_label_by_text(self, field_template: FieldTemplate):
        """Find label by searching for label text patterns."""
        label_keywords = {
            'invoice_number': ['invoice', 'inv', 'no', '#'],
            'invoice_date': ['date', 'dated'],
            'invoice_amount': ['total', 'amount', 'due', 'balance', 'grand'],
        }

        keywords = label_keywords.get(field_template.field_name, [])

        for norm_text, indices in self._text_to_idx.items():
            for keyword in keywords:
                if keyword in norm_text:
                    return self.words[indices[0]]

        return None

    def _find_value_at_offset(self, label_word: BBox, field_template: FieldTemplate):
        """Find value at the learned offset from label."""
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
                # Below: use both offset_x and offset_y
                if word.top >= label_word.bottom + offset_y - 20:  # Allow some tolerance
                    expected_x = label_word.left + offset_x
                    dist = abs(word.top - (label_word.bottom + offset_y)) + abs(word.left - expected_x)
                    candidates.append((word, dist, 'below'))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

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

    def _parse_amount(self, text: str):
        """Parse amount from text."""
        cleaned = re.sub(r'[$€£,]', '', text)
        match = re.search(r'([\d,]+\.?\d*)', cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        return None

    def _extract_items(self) -> list:
        """Extract line items using learned table structure."""
        if not self.template.table_template:
            return []

        table = self.template.table_template
        items = []

        lines = self._group_words_by_line()

        for line_words in lines:
            line_y = line_words[0].top

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

        numbers = []
        item_name_parts = []

        for word in sorted(line_words, key=lambda w: w.left):
            num_match = re.search(r'[\d,]+\.?\d*', word.text)
            if num_match:
                try:
                    val = float(num_match.group().replace(',', ''))
                    if val > 0:
                        numbers.append({'word': word, 'value': val, 'x': word.left})
                except ValueError:
                    pass
            else:
                word_lower = word.text.lower()
                if word_lower not in ['qty', 'quantity', 'unit', 'price', 'total', 'x', 'ea', 'each', 'pcs']:
                    item_name_parts.append(word.text)

        if not numbers:
            return None

        item_name = ' '.join(item_name_parts).strip() or 'Unknown Item'

        qty = 1
        unit_price = 0.0
        total_price = 0.0
        item_no = None

        numbers.sort(key=lambda x: x['x'])

        if len(numbers) >= 1:
            first_val = numbers[0]['value']
            if first_val < 100 and first_val == int(first_val):
                item_no = int(first_val)
                qty = item_no
            total_price = numbers[-1]['value']

        if len(numbers) >= 2:
            unit_price = numbers[-1]['value']
            if qty > 0:
                unit_price = numbers[-1]['value'] / qty

        if len(numbers) >= 3:
            unit_price = numbers[len(numbers)//2]['value']

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


def main():
    # Configuration
    data_dir = Path(__file__).parent / "Data"
    vendor_dir = Path(__file__).parent / "vendors" / "sample_vendor"

    # Find all PDF files
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in Data folder")
        return

    print(f"Found {len(pdf_files)} invoice PDFs\n")

    validated_invoices = []

    for pdf_path in pdf_files:
        invoice_id = pdf_path.stem
        json_path = data_dir / f"{invoice_id}.json"

        if not json_path.exists():
            print(f"Warning: No JSON found for {pdf_path.name}, skipping")
            continue

        with open(json_path, 'r') as f:
            ground_truth = json.load(f)

        print(f"Processing {invoice_id}:")
        print(f"  Invoice #: {ground_truth.get('invoice_number')}")
        print(f"  Date: {ground_truth.get('invoice_date')}")
        print(f"  Amount: {ground_truth.get('invoice_amount')}")
        print(f"  Items: {len(ground_truth.get('items', []))}")

        words, img_width, img_height = extract_pdf_words(pdf_path)
        print(f"  OCR: {len(words)} words detected, image size: {img_width}x{img_height}")

        validated_invoices.append({
            'ocr_words': words,
            'ground_truth': ground_truth,
            'image_width': img_width,
            'image_height': img_height
        })

    if not validated_invoices:
        print("No valid invoice-JSON pairs found")
        return

    # Learn template
    print("\n" + "="*50)
    print("Learning vendor template from validated invoices...")
    print("="*50 + "\n")

    learner = VendorTemplateLearner("sample_vendor", vendor_dir)

    for invoice in validated_invoices:
        print(f"Learning from invoice...")
        learner.learn_from_invoice(
            ocr_words=invoice['ocr_words'],
            ground_truth=invoice['ground_truth'],
            image_width=invoice['image_width'],
            image_height=invoice['image_height']
        )

    learner.save_template()
    print(f"\nTemplate saved to: {vendor_dir / 'position_template.json'}")

    # Print summary
    print("\n" + "="*50)
    print("LEARNED TEMPLATE SUMMARY")
    print("="*50)

    for field_name, template in learner.templates.items():
        print(f"\n{field_name.upper()}:")
        print(f"  Label position: ({template.label_x}, {template.label_y})")
        print(f"  Value offset: ({template.value_offset_x}, {template.value_offset_y})")
        print(f"  Position type: {template.value_position}")
        print(f"  Label variations: {template.label_texts}")
        print(f"  Confidence boost: {template.confidence_boost:.2f}")

    if learner.table_template:
        print(f"\nTABLE TEMPLATE:")
        print(f"  Start Y: {learner.table_template.start_y}")
        print(f"  Row height: {learner.table_template.row_height}")
        print(f"  Columns: item_no={learner.table_template.item_no_col}, name={learner.table_template.item_name_col}, qty={learner.table_template.qty_col}, price={learner.table_template.price_col}, total={learner.table_template.total_col}")

    # Test extraction
    print("\n" + "="*50)
    print("TESTING EXTRACTION")
    print("="*50)

    for invoice in validated_invoices:
        gt = invoice['ground_truth']
        print(f"\nInvoice {gt.get('invoice_number')}:")
        print(f"  Ground truth: #{gt.get('invoice_number')} | {gt.get('invoice_date')} | ${gt.get('invoice_amount')}")

        extractor = PositionBasedExtractor(invoice['ocr_words'], learner)
        extracted = extractor.extract()

        print(f"  Extracted:   #{extracted.get('invoice_number')} | {extracted.get('invoice_date')} | ${extracted.get('invoice_amount')}")

        gt_items = gt.get('items', [])
        ex_items = extracted.get('items', [])
        print(f"  Items: GT={len(gt_items)} | Extracted={len(ex_items)}")

        if gt_items and ex_items:
            for i, (gt_item, ex_item) in enumerate(zip(gt_items, ex_items)):
                gt_name = gt_item.get('item_name', '').upper()
                ex_name = ex_item.get('item_name', '').upper()
                match = "OK" if gt_name == ex_name else "DIFF"
                print(f"    {match} {gt_name} -> {ex_name}")


if __name__ == "__main__":
    main()
