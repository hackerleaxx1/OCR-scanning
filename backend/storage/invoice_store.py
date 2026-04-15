"""
Invoice Storage and Management.

Provides CRUD operations for processed invoices.
Invoices are stored as individual JSON files in the invoices storage directory.

Invoice Lifecycle:
1. UPLOAD: Invoice image uploaded, OCR run, fields predicted -> status: 'pending'
2. VALIDATION: User corrects predictions -> status: 'validated', corrected_data stored
3. LEARNING: On validation, data feeds into vendor's training data and patterns

Each invoice record contains:
- id: Unique invoice identifier
- vendor_id: Which vendor this invoice belongs to
- status: 'pending' (awaiting validation) or 'validated' (corrected)
- image_path: Path to the uploaded invoice image
- ocr_text: Raw text from OCR
- predictions: ML-predicted field values
- confidence_scores: Confidence for each predicted field
- corrected_data: User-corrected values (after validation)
- timestamps: created_at, validated_at
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional
from config import config


class InvoiceStore:
    """
    Manages invoice CRUD operations.

    Invoice storage format: one JSON file per invoice in the storage directory.
    This allows:
    - Direct access to individual invoices
    - Easy inspection and debugging
    - Simple backup and replication

    The storage is stateless - validated invoices still contain all the
    original predictions for comparison and learning analysis.
    """

    def __init__(self, storage_dir: str = None):
        """
        Initialize invoice store.

        Args:
            storage_dir: Override for storage directory path
        """
        self.storage_dir = Path(storage_dir or config['invoices']['storage_dir'])
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _invoice_path(self, invoice_id: str) -> Path:
        """Get path to invoice's JSON file."""
        return self.storage_dir / f"{invoice_id}.json"

    def create_invoice(self, vendor_id: str, predictions: dict,
                       ocr_text: str, image_path: str, confidence_scores: dict) -> dict:
        """
        Create a new invoice record after OCR processing.

        Called after uploading an invoice. Stores the OCR results and
        ML predictions for later validation.

        Args:
            vendor_id: Which vendor this invoice belongs to
            predictions: ML-predicted field values
            ocr_text: Raw text extracted by OCR
            image_path: Filename of the stored invoice image
            confidence_scores: Confidence for each predicted field

        Returns:
            Created invoice record dict
        """
        invoice_id = f"inv_{uuid.uuid4().hex[:8]}"

        invoice_data = {
            'id': invoice_id,
            'vendor_id': vendor_id,
            'status': 'pending',              # Awaiting user validation
            'image_path': image_path,
            'ocr_text': ocr_text,
            'predictions': predictions,
            'confidence_scores': confidence_scores,
            'corrected_data': None,           # Filled after validation
            'created_at': datetime.now().isoformat(),
            'validated_at': None              # Filled after validation
        }

        with open(self._invoice_path(invoice_id), 'w') as f:
            json.dump(invoice_data, f, indent=2)

        return invoice_data

    def get_invoice(self, invoice_id: str) -> Optional[dict]:
        """
        Get invoice by ID.

        Args:
            invoice_id: Unique invoice identifier

        Returns:
            Invoice record dict, or None if not found
        """
        path = self._invoice_path(invoice_id)
        if not path.exists():
            return None
        with open(path, 'r') as f:
            return json.load(f)

    def list_invoices(self, status: str = None, vendor_id: str = None,
                      page: int = 1, limit: int = 20) -> dict:
        """
        List invoices with optional filtering and pagination.

        Args:
            status: Filter by status ('pending', 'validated', or 'all')
            vendor_id: Filter by specific vendor
            page: Page number (1-indexed)
            limit: Number of invoices per page

        Returns:
            Dict with:
            - invoices: List of invoice records for this page
            - pagination: Metadata (page, limit, total, total_pages)
        """
        invoices = []

        for path in self.storage_dir.glob("*.json"):
            try:
                with open(path, 'r') as f:
                    inv = json.load(f)
            except (json.JSONDecodeError, IOError):
                # Skip malformed or locked files
                continue

            # Apply filters
            if status and status != 'all' and inv.get('status') != status:
                continue
            if vendor_id and inv.get('vendor_id') != vendor_id:
                continue

            invoices.append(inv)

        # Sort by creation date (newest first)
        invoices.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        # Apply pagination
        total = len(invoices)
        start = (page - 1) * limit
        end = start + limit
        paginated = invoices[start:end]

        return {
            'invoices': paginated,
            'pagination': {
                'page': page,
                'limit': limit,
                'total': total,
                'pages': (total + limit - 1) // limit  # Ceiling division
            }
        }

    def update_invoice(self, invoice_id: str, updates: dict) -> Optional[dict]:
        """
        Update invoice fields.

        Args:
            invoice_id: Unique invoice identifier
            updates: Dict of fields to update

        Returns:
            Updated invoice record, or None if not found
        """
        invoice = self.get_invoice(invoice_id)
        if not invoice:
            return None

        invoice.update(updates)
        # Set validated_at timestamp if not explicitly provided
        if 'validated_at' not in updates:
            invoice['validated_at'] = datetime.now().isoformat()

        with open(self._invoice_path(invoice_id), 'w') as f:
            json.dump(invoice, f, indent=2)

        return invoice

    def validate_invoice(self, invoice_id: str, corrected_data: dict) -> Optional[dict]:
        """
        Mark invoice as validated with user corrections.

        This is the critical learning step in the pipeline. The corrected
        data is stored and used to update:
        - Vendor's position template
        - Vendor's regex patterns
        - Vendor's training data for KNN inference

        Args:
            invoice_id: Unique invoice identifier
            corrected_data: Dict with user-corrected field values

        Returns:
            Updated invoice record
        """
        updates = {
            'status': 'validated',
            'corrected_data': corrected_data
        }

        # If corrected_data contains items, preserve them for future reference
        if 'items' in corrected_data and corrected_data['items']:
            updates['validated_items'] = corrected_data['items']

        return self.update_invoice(invoice_id, updates)

    def delete_invoice(self, invoice_id: str) -> bool:
        """
        Delete an invoice.

        Args:
            invoice_id: Unique invoice identifier

        Returns:
            True if invoice was deleted, False if not found
        """
        path = self._invoice_path(invoice_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def get_stats(self) -> dict:
        """
        Get invoice processing statistics.

        Returns:
            Dict with counts of total, pending, and validated invoices
        """
        total = 0
        pending = 0
        validated = 0

        for path in self.storage_dir.glob("*.json"):
            try:
                with open(path, 'r') as f:
                    inv = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
            total += 1
            if inv.get('status') == 'pending':
                pending += 1
            elif inv.get('status') == 'validated':
                validated += 1

        return {
            'total': total,
            'pending': pending,
            'validated': validated
        }
