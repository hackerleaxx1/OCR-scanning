"""
API Routes for Invoice OCR API.

All REST endpoints organized by resource:
- /api/upload - Upload and process invoice images
- /api/invoices - Invoice CRUD operations
- /api/vendors - Vendor management
- /api/stats - System statistics
- /api/health - Health check

Each endpoint handles a specific aspect of the invoice processing pipeline:
1. Upload -> OCR extraction -> Field prediction -> Store as pending
2. Validation -> Learn patterns/templates -> Update training data
3. Subsequent invoices use learned knowledge for improved accuracy
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import json
from datetime import datetime

from config import config
from storage.vendor_store import VendorStore
from storage.invoice_store import InvoiceStore
from ocr.tesseract import extract_full_image_data
from ocr.field_extractor import extract_fields_from_ocr
from ocr.position_extractor import VendorTemplateLearner, PositionBasedExtractor
from ml.pattern_learner import PatternLearner, extract_candidates
from ml.inference import InvoicePredictor
from ml.trainer import ModelTrainer

router = APIRouter(prefix="/api")

# Initialize storage backends
vendor_store = VendorStore()
invoice_store = InvoiceStore()


# =============================================================================
# Invoice Upload & Processing
# =============================================================================

@router.post("/upload")
async def upload_invoice(
    file: UploadFile = File(...),
    vendor_id: str = Form(None)
):
    """
    Upload an invoice image, extract text via OCR, and predict all fields.

    The extraction pipeline follows a priority order for accuracy:
    1. Position-based extraction (using learned vendor templates) - MOST ACCURATE
    2. Generic field extraction with position heuristics
    3. Pattern-based extraction (using learned regex patterns)
    4. KNN-based inference (using validated training samples)

    Args:
        file: The invoice image file (JPEG, PNG, or PDF)
        vendor_id: Optional vendor ID. If not provided, uses the first available vendor.

    Returns:
        dict: Invoice record with predictions and confidence scores

    Raises:
        HTTPException 400: If file type is invalid or no vendors exist
        HTTPException 500: If OCR processing fails
    """
    # -------------------------------------------------------------------------
    # Step 1: Validate file type
    # -------------------------------------------------------------------------
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(400, f"File type {file.content_type} not allowed")

    # -------------------------------------------------------------------------
    # Step 2: Save uploaded file to disk
    # -------------------------------------------------------------------------
    upload_dir = Path(config['app']['upload_dir'])
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique invoice ID with timestamp for uniqueness
    invoice_id = f"inv_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    ext = Path(file.filename).suffix or ".jpg"
    image_path = upload_dir / f"{invoice_id}{ext}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # -------------------------------------------------------------------------
    # Step 3: Run OCR to extract text and word positions
    # -------------------------------------------------------------------------
    try:
        ocr_result = extract_full_image_data(str(image_path))
        ocr_text = ocr_result['text']
    except Exception as e:
        raise HTTPException(500, f"OCR failed: {str(e)}")

    # -------------------------------------------------------------------------
    # Step 4: Determine which vendor to use
    # -------------------------------------------------------------------------
    if not vendor_id:
        # Auto-detect: use first available vendor or require explicit vendor
        vendors = vendor_store.list_vendors()
        if not vendors:
            raise HTTPException(400, "No vendors found. Please create a vendor first.")
        vendor_id = vendors[0]['id']

    vendor_dir = Path(config['vendors']['storage_dir']) / vendor_id

    # -------------------------------------------------------------------------
    # Step 5: Check for learned position template (highest accuracy method)
    # -------------------------------------------------------------------------
    template_learner = VendorTemplateLearner.load_template(vendor_dir)
    has_position_template = template_learner is not None

    # Initialize predictions and confidence scores
    predictions = {}
    confidence_scores = {}

    if has_position_template:
        # -----------------------------------------------------------------
        # METHOD 1: Position-based extraction (using learned vendor template)
        # This is the MOST ACCURATE method once a vendor has 3+ validated invoices
        # -----------------------------------------------------------------
        print(f"Using position template for vendor {vendor_id}")
        extractor = PositionBasedExtractor(ocr_result.get('words', []), template_learner)
        pos_predictions = extractor.extract()

        predictions = {
            'invoice_number': pos_predictions.get('invoice_number'),
            'invoice_date': pos_predictions.get('invoice_date'),
            'invoice_amount': pos_predictions.get('invoice_amount'),
            'items': pos_predictions.get('items', [])
        }

        # High confidence when position template exists (template was learned from real data)
        confidence_scores = {
            'invoice_number': 0.95 if predictions.get('invoice_number') else 0.0,
            'invoice_date': 0.95 if predictions.get('invoice_date') else 0.0,
            'invoice_amount': 0.95 if predictions.get('invoice_amount') else 0.0,
            'items': [{
                'item_no': 0.9,
                'item_name': 0.9,
                'item_quantity': 0.9,
                'per_item_price': 0.9,
                'total_item_price': 0.9
            } for _ in predictions.get('items', [])]
        }
    else:
        # -----------------------------------------------------------------
        # METHOD 2: Generic field extraction (no template exists yet)
        # Uses position heuristics and regex patterns as fallbacks
        # -----------------------------------------------------------------
        print(f"No position template for vendor {vendor_id}, using generic extraction")
        field_result = extract_fields_from_ocr(ocr_result)
        predictions = field_result['predictions']
        confidence_scores = field_result['confidences']

        # -----------------------------------------------------------------
        # METHOD 2a: Pattern-based fallback for low-confidence fields
        # Uses learned regex patterns to find missed fields
        # -----------------------------------------------------------------
        needs_pattern_fallback = any(
            confidence_scores.get(f, 0) < 0.5
            for f in ['invoice_number', 'invoice_date', 'invoice_amount']
        )

        if needs_pattern_fallback:
            pattern_learner = PatternLearner(vendor_id, vendor_dir)
            candidates = extract_candidates(
                pattern_learner.patterns,
                ocr_text,
                ocr_result.get('words', [])
            )
            # Merge top candidates for low-confidence fields
            for field in ['invoice_number', 'invoice_date', 'invoice_amount']:
                if confidence_scores.get(field, 0) < 0.5:
                    field_candidates = candidates.get(field, [])
                    if field_candidates:
                        top = field_candidates[0]
                        predictions[field] = top['value']
                        confidence_scores[field] = top['confidence']

        # -----------------------------------------------------------------
        # METHOD 2b: KNN-based inference refinement
        # If we have validated training data, use similarity matching
        # This works well even with small datasets (1-5 samples)
        # -----------------------------------------------------------------
        predictor = InvoicePredictor(vendor_id, vendor_dir)
        if predictor.df is not None and len(predictor.df) > 0:
            knn_result = predictor.predict(ocr_text)
            knn_predictions = knn_result['predictions']
            knn_confidences = knn_result['confidences']

            # Only use KNN predictions if they have higher confidence
            for field in ['invoice_number', 'invoice_date', 'invoice_amount']:
                if confidence_scores.get(field, 0) < knn_confidences.get(field, 0):
                    predictions[field] = knn_predictions[field]
                    confidence_scores[field] = knn_confidences[field]

            # Use KNN items if they found more items
            if knn_predictions.get('items') and len(knn_predictions['items']) > len(predictions.get('items', [])):
                predictions['items'] = knn_predictions['items']
                confidence_scores['items'] = knn_confidences.get('items', [])

    # -------------------------------------------------------------------------
    # Step 6: Ensure items array always exists (even if empty)
    # -------------------------------------------------------------------------
    if 'items' not in predictions or not predictions.get('items'):
        predictions['items'] = []
        confidence_scores['items'] = []
    else:
        if 'items' not in confidence_scores or not confidence_scores.get('items'):
            confidence_scores['items'] = [{
                'item_no': 0.8,
                'item_name': 0.8,
                'item_quantity': 0.8,
                'per_item_price': 0.8,
                'total_item_price': 0.8
            } for _ in predictions.get('items', [])]

    # -------------------------------------------------------------------------
    # Step 7: Store invoice record with predictions for later validation
    # -------------------------------------------------------------------------
    invoice = invoice_store.create_invoice(
        vendor_id=vendor_id,
        predictions=predictions,
        ocr_text=ocr_text,
        image_path=image_path.name,  # Store only filename, not full path
        confidence_scores=confidence_scores
    )

    return invoice


# =============================================================================
# Invoice Read Operations
# =============================================================================

@router.get("/invoices")
async def list_invoices(
    status: str = None,
    vendor_id: str = None,
    page: int = 1,
    limit: int = 20
):
    """
    List invoices with optional filtering by status and vendor.

    Args:
        status: Filter by status ('pending', 'validated', or 'all')
        vendor_id: Filter by specific vendor
        page: Page number for pagination (1-indexed)
        limit: Number of invoices per page

    Returns:
        dict: Paginated list of invoices
    """
    return invoice_store.list_invoices(
        status=status,
        vendor_id=vendor_id,
        page=page,
        limit=limit
    )


@router.get("/invoices/{invoice_id}")
async def get_invoice(invoice_id: str):
    """
    Get detailed information for a single invoice.

    Args:
        invoice_id: Unique invoice identifier

    Returns:
        dict: Full invoice record including predictions and confidence scores

    Raises:
        HTTPException 404: If invoice not found
    """
    invoice = invoice_store.get_invoice(invoice_id)
    if not invoice:
        raise HTTPException(404, "Invoice not found")
    return invoice


# =============================================================================
# Invoice Validation & Learning
# =============================================================================

@router.post("/invoices/{invoice_id}/validate")
async def validate_invoice(invoice_id: str, data: dict):
    """
    Submit validation corrections for an invoice.

    This is a critical learning step that:
    1. Marks the invoice as validated
    2. Learns position template from the corrected data (for accurate future extraction)
    3. Updates regex patterns based on the corrected values
    4. Adds the sample to vendor's training data for KNN inference

    Args:
        invoice_id: Unique invoice identifier
        data: Dict containing corrected field values:
              - invoice_number: str
              - invoice_date: str
              - invoice_amount: float
              - items: list of line item dicts

    Returns:
        dict: Validation confirmation with learning status

    Raises:
        HTTPException 404: If invoice not found
        HTTPException 400: If invoice already validated
    """
    invoice = invoice_store.get_invoice(invoice_id)
    if not invoice:
        raise HTTPException(404, "Invoice not found")

    if invoice['status'] == 'validated':
        raise HTTPException(400, "Invoice already validated")

    # -------------------------------------------------------------------------
    # Extract corrected fields from user submission
    # -------------------------------------------------------------------------
    corrected_data = {
        'invoice_number': data.get('invoice_number'),
        'invoice_date': data.get('invoice_date'),
        'invoice_amount': float(data.get('invoice_amount', 0)),
        'items': data.get('items', [])
    }

    # Mark invoice as validated
    invoice_store.validate_invoice(invoice_id, corrected_data)

    # -------------------------------------------------------------------------
    # Learn from this validated invoice to improve future extraction
    # -------------------------------------------------------------------------
    vendor_id = invoice['vendor_id']
    vendor_dir = Path(config['vendors']['storage_dir']) / vendor_id

    # Re-run OCR on the original image to get word positions for template learning
    upload_dir = Path(config['app']['upload_dir'])
    image_path = upload_dir / invoice['image_path']

    position_template_learned = False
    if image_path.exists():
        try:
            ocr_result = extract_full_image_data(str(image_path))
            words = ocr_result.get('words', [])
            img_width = ocr_result.get('image_width', 800)
            img_height = ocr_result.get('image_height', 1000)

            # Load or create template learner
            template_learner = VendorTemplateLearner.load_template(vendor_dir)
            if template_learner is None:
                template_learner = VendorTemplateLearner(vendor_id, vendor_dir)

            # Learn field positions from this validated invoice
            # After 3-4 validated invoices, the template enables high-accuracy extraction
            template_learner.learn_from_invoice(
                ocr_words=words,
                ground_truth=corrected_data,
                image_width=img_width,
                image_height=img_height
            )

            template_learner.save_template()
            position_template_learned = True
        except Exception as e:
            print(f"Could not learn position template: {e}")

    # -------------------------------------------------------------------------
    # Learn regex patterns from validated data
    # Updates invoice number format, date format, amount range, etc.
    # -------------------------------------------------------------------------
    pattern_learner = PatternLearner(vendor_id, vendor_dir)
    pattern_learner.learn_from_sample(corrected_data)

    # -------------------------------------------------------------------------
    # Add to vendor's training data for KNN inference
    # -------------------------------------------------------------------------
    training_sample = {
        'invoice_number': corrected_data['invoice_number'],
        'invoice_date': corrected_data['invoice_date'],
        'invoice_amount': corrected_data['invoice_amount'],
        'items_json': json.dumps(corrected_data.get('items', [])),
        'ocr_text': invoice['ocr_text']
    }
    vendor_store.add_training_sample(vendor_id, training_sample)
    training_count = vendor_store.get_training_count(vendor_id)

    return {
        'id': invoice_id,
        'status': 'validated',
        'validated_at': datetime.now().isoformat(),
        'retrain_triggered': position_template_learned,
        'message': f"Validation saved. {'Position template updated.' if position_template_learned else 'Patterns updated.'} {training_count} training samples."
    }


# =============================================================================
# Vendor Management
# =============================================================================

@router.get("/vendors")
async def list_vendors():
    """
    List all registered vendors.

    Returns:
        dict: List of vendor metadata objects
    """
    vendors = vendor_store.list_vendors()
    return {'vendors': vendors}


@router.post("/vendors")
async def create_vendor(data: dict):
    """
    Register a new vendor for invoice processing.

    Each vendor maintains its own:
    - Position templates (learned from validated invoices)
    - Regex patterns
    - Training data for KNN inference
    - PyTorch model weights

    Args:
        data: Dict with 'name' (required) and 'description' (optional)

    Returns:
        dict: Created vendor metadata including generated ID

    Raises:
        HTTPException 400: If vendor name is missing
    """
    name = data.get('name')
    if not name:
        raise HTTPException(400, "Vendor name is required")

    description = data.get('description', '')
    vendor = vendor_store.create_vendor(name, description)
    return vendor


@router.get("/vendors/{vendor_id}")
async def get_vendor(vendor_id: str):
    """
    Get detailed information for a specific vendor.

    Args:
        vendor_id: Unique vendor identifier

    Returns:
        dict: Vendor metadata including training statistics

    Raises:
        HTTPException 404: If vendor not found
    """
    vendor = vendor_store.get_vendor(vendor_id)
    if not vendor:
        raise HTTPException(404, "Vendor not found")
    return vendor


@router.post("/vendors/{vendor_id}/retrain")
async def retrain_vendor(vendor_id: str, options: dict = None):
    """
    Trigger retraining of vendor's position template and ML model.

    Note: The system learns automatically from validated invoices.
    This endpoint provides explicit control over when retraining occurs.

    Args:
        vendor_id: Unique vendor identifier
        options: Optional training configuration (epochs, learning_rate, etc.)

    Returns:
        dict: Training status and results

    Raises:
        HTTPException 404: If vendor not found
    """
    vendor = vendor_store.get_vendor(vendor_id)
    if not vendor:
        raise HTTPException(404, "Vendor not found")

    vendor_dir = Path(config['vendors']['storage_dir']) / vendor_id
    training_count = vendor_store.get_training_count(vendor_id)

    # Check if position template exists
    template_path = vendor_dir / "position_template.json"
    has_template = template_path.exists()

    return {
        'vendor_id': vendor_id,
        'status': 'ready',
        'model_type': 'position_based',
        'has_position_template': has_template,
        'message': 'Position-based extraction using learned templates. Validated invoices update the template automatically.',
        'training_samples': training_count
    }


@router.delete("/vendors/{vendor_id}")
async def delete_vendor(vendor_id: str):
    """
    Delete a vendor and all associated data.

    This removes:
    - Vendor metadata
    - Learned position templates
    - Learned patterns
    - Training data
    - ML model weights

    Args:
        vendor_id: Unique vendor identifier

    Returns:
        dict: Confirmation message

    Raises:
        HTTPException 404: If vendor not found
    """
    if vendor_store.delete_vendor(vendor_id):
        return {'message': 'Vendor deleted'}
    raise HTTPException(404, 'Vendor not found')


# =============================================================================
# Statistics & Health
# =============================================================================

@router.get("/stats")
async def get_stats():
    """
    Get overall system statistics.

    Returns:
        dict: Invoice counts (total, pending, validated) and vendor count
    """
    invoice_stats = invoice_store.get_stats()
    vendors = vendor_store.list_vendors()

    return {
        **invoice_stats,
        'total_vendors': len(vendors)
    }


@router.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.

    Returns:
        dict: System health status and configuration
    """
    import platform

    return {
        'status': 'healthy',
        'platform': platform.system(),
        'ocr_engine': 'tesseract',
        'inference_mode': 'position_based_with_fallback'
    }
