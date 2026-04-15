"""
Vendor Storage and Management.

Provides CRUD operations for vendors and manages their associated data:
- Vendor metadata (name, description, status, training count)
- Training data (CSV of validated invoices)
- Per-vendor ML artifacts (models, patterns, templates)

Each vendor is stored in its own directory under the vendors storage path.
Directory structure:
  vendors/
    {vendor_id}/
      metadata.json       # Vendor metadata
      training_data.csv   # Validated invoice samples
      position_template.json  # Learned position template
      patterns.json       # Learned regex patterns
      vocabulary.json     # TF-IDF vocabulary
      weights.csv         # Model weights (if trained)
      biases.csv          # Model biases (if trained)
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
from config import config


class VendorStore:
    """
    Manages vendor CRUD operations and model weight persistence.

    Provides an abstraction layer over the file-based storage of vendors.
    Each vendor has their own directory containing metadata, training data,
    and ML model artifacts.

    The training_data.csv accumulates validated invoices over time.
    This data is used for:
    - KNN-based inference
    - Neural network training
    - Pattern learning
    - Position template learning
    """

    def __init__(self, storage_dir: str = None):
        """
        Initialize vendor store.

        Args:
            storage_dir: Override for storage directory path
        """
        self.storage_dir = Path(storage_dir or config['vendors']['storage_dir'])
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _vendor_dir(self, vendor_id: str) -> Path:
        """Get path to vendor's directory."""
        return self.storage_dir / vendor_id

    def _ensure_vendor_dir(self, vendor_id: str) -> Path:
        """Ensure vendor directory exists, create if needed."""
        vendor_path = self._vendor_dir(vendor_id)
        vendor_path.mkdir(parents=True, exist_ok=True)
        return vendor_path

    def create_vendor(self, name: str, description: str = "") -> dict:
        """
        Register a new vendor.

        Creates the vendor directory and initializes:
        - metadata.json with vendor info
        - Empty training_data.csv with proper columns

        Args:
            name: Vendor display name
            description: Optional vendor description

        Returns:
            Vendor metadata dict including generated ID
        """
        # Generate unique 8-character vendor ID
        vendor_id = str(uuid.uuid4())[:8]
        vendor_path = self._ensure_vendor_dir(vendor_id)

        vendor_data = {
            'id': vendor_id,
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'status': 'untrained',
            'training_samples': 0
        }

        with open(vendor_path / "metadata.json", 'w') as f:
            json.dump(vendor_data, f)

        # Initialize empty training data CSV with schema
        training_cols = [
            'id', 'invoice_number', 'invoice_date', 'invoice_amount',
            'items_json', 'ocr_text', 'timestamp'
        ]
        pd.DataFrame(columns=training_cols).to_csv(vendor_path / "training_data.csv", index=False)

        return vendor_data

    def get_vendor(self, vendor_id: str) -> Optional[dict]:
        """
        Get vendor metadata by ID.

        Args:
            vendor_id: Unique vendor identifier

        Returns:
            Vendor metadata dict, or None if not found
        """
        meta_path = self._vendor_dir(vendor_id) / "metadata.json"
        if not meta_path.exists():
            return None
        with open(meta_path, 'r') as f:
            return json.load(f)

    def list_vendors(self) -> list:
        """
        List all registered vendors.

        Returns:
            List of vendor metadata dicts, sorted by creation date (newest first)
        """
        vendors = []
        for vendor_path in self.storage_dir.iterdir():
            if vendor_path.is_dir():
                meta_path = vendor_path / "metadata.json"
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        vendors.append(json.load(f))
        return sorted(vendors, key=lambda v: v.get('created_at', ''), reverse=True)

    def update_vendor(self, vendor_id: str, updates: dict) -> Optional[dict]:
        """
        Update vendor fields.

        Args:
            vendor_id: Unique vendor identifier
            updates: Dict of fields to update

        Returns:
            Updated vendor metadata, or None if vendor not found
        """
        vendor = self.get_vendor(vendor_id)
        if not vendor:
            return None

        vendor.update(updates)
        vendor['updated_at'] = datetime.now().isoformat()

        with open(self._vendor_dir(vendor_id) / "metadata.json", 'w') as f:
            json.dump(vendor, f)

        return vendor

    def add_training_sample(self, vendor_id: str, invoice_data: dict) -> bool:
        """
        Add a validated invoice to vendor's training data.

        This is called during invoice validation. The validated invoice
        data is appended to the vendor's training_data.csv for use in
        future KNN inference, pattern learning, and model training.

        Args:
            vendor_id: Unique vendor identifier
            invoice_data: Dict with invoice field values

        Returns:
            True on success
        """
        vendor_path = self._vendor_dir(vendor_id)
        csv_path = vendor_path / "training_data.csv"

        # Build row with all required fields
        row = {
            'id': str(uuid.uuid4())[:8],
            'invoice_number': invoice_data.get('invoice_number', ''),
            'invoice_date': invoice_data.get('invoice_date', ''),
            'invoice_amount': invoice_data.get('invoice_amount', 0.0),
            'items_json': invoice_data.get('items_json', '[]'),
            'ocr_text': invoice_data.get('ocr_text', ''),
            'timestamp': datetime.now().isoformat()
        }

        # Append to CSV
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(csv_path, index=False)

        # Update training sample count in metadata
        self.update_vendor(vendor_id, {'training_samples': len(df)})

        return True

    def get_training_count(self, vendor_id: str) -> int:
        """
        Get number of training samples for a vendor.

        Args:
            vendor_id: Unique vendor identifier

        Returns:
            Number of validated invoices in training data
        """
        csv_path = self._vendor_dir(vendor_id) / "training_data.csv"
        if not csv_path.exists():
            return 0
        df = pd.read_csv(csv_path)
        return len(df)

    def delete_vendor(self, vendor_id: str) -> bool:
        """
        Delete a vendor and all associated data.

        This removes the entire vendor directory including:
        - Metadata
        - Training data
        - Learned patterns and templates
        - ML model weights

        Use with caution - this cannot be undone.

        Args:
            vendor_id: Unique vendor identifier

        Returns:
            True if vendor was deleted, False if not found
        """
        import shutil
        vendor_path = self._vendor_dir(vendor_id)
        if vendor_path.exists():
            shutil.rmtree(vendor_path)
            return True
        return False
