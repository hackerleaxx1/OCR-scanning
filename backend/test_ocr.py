"""
Test script to process sample invoices and compare OCR results with ground truth.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
import json
from ocr.tesseract import extract_full_image_data
from ocr.field_extractor import extract_fields_from_ocr
from ocr.position_extractor import VendorTemplateLearner, PositionBasedExtractor
from storage.vendor_store import VendorStore

# Paths
DATA_DIR = Path(__file__).parent.parent / "Data"
BACKEND_DIR = Path(__file__).parent
VENDOR_ID = "test_vendor"

def load_ground_truth(invoice_num):
    """Load ground truth JSON for an invoice."""
    json_path = DATA_DIR / f"{invoice_num}.json"
    with open(json_path, 'r') as f:
        return json.load(f)

def test_ocr_on_invoice(invoice_num):
    """Test OCR on a single invoice and compare with ground truth."""
    pdf_path = DATA_DIR / f"{invoice_num}.pdf"
    gt = load_ground_truth(invoice_num)

    print(f"\n{'='*60}")
    print(f"Testing Invoice #{invoice_num}")
    print(f"{'='*60}")
    print(f"PDF path: {pdf_path}")
    print(f"Ground truth: {json.dumps(gt, indent=2)}")

    # Run OCR
    print("\n[1] Running OCR...")
    try:
        ocr_result = extract_full_image_data(str(pdf_path))
        print(f"    OCR text length: {len(ocr_result['text'])} chars")
        print(f"    Words extracted: {len(ocr_result.get('words', []))}")
        print(f"    Image size: {ocr_result.get('image_width', '?')}x{ocr_result.get('image_height', '?')}")
    except Exception as e:
        print(f"    ERROR during OCR: {e}")
        return None

    # Run field extraction (generic, no template)
    print("\n[2] Running field extraction (generic)...")
    try:
        field_result = extract_fields_from_ocr(ocr_result)
        predictions = field_result['predictions']
        confidences = field_result['confidences']

        print(f"    Invoice Number: {predictions.get('invoice_number')} (conf: {confidences.get('invoice_number', 0):.2f})")
        print(f"    Invoice Date: {predictions.get('invoice_date')} (conf: {confidences.get('invoice_date', 0):.2f})")
        print(f"    Invoice Amount: {predictions.get('invoice_amount')} (conf: {confidences.get('invoice_amount', 0):.2f})")
        print(f"    Items: {len(predictions.get('items', []))} found")

        for i, item in enumerate(predictions.get('items', [])[:5]):
            print(f"      Item {i+1}: {item}")
    except Exception as e:
        print(f"    ERROR during field extraction: {e}")
        import traceback
        traceback.print_exc()
        predictions = {}
        confidences = {}

    # Compare with ground truth
    print("\n[3] Comparison with Ground Truth:")
    print(f"    GT Invoice Number: {gt.get('invoice_number')}")
    print(f"    OCR Invoice Number: {predictions.get('invoice_number')}")
    match_inv = predictions.get('invoice_number') == str(gt.get('invoice_number'))
    print(f"    Match: {'OK' if match_inv else 'FAIL'}")

    print(f"\n    GT Invoice Date: {gt.get('invoice_date')}")
    print(f"    OCR Invoice Date: {predictions.get('invoice_date')}")
    match_date = predictions.get('invoice_date') == gt.get('invoice_date')
    print(f"    Match: {'OK' if match_date else 'FAIL'}")

    print(f"\n    GT Invoice Amount: {gt.get('invoice_amount')}")
    print(f"    OCR Invoice Amount: {predictions.get('invoice_amount')}")
    match_amt = abs((predictions.get('invoice_amount') or 0) - gt.get('invoice_amount', 0)) < 1
    print(f"    Match: {'OK' if match_amt else 'FAIL'}")

    # Compare items
    gt_items = gt.get('items', [])
    pred_items = predictions.get('items', [])
    print(f"\n    GT Items: {len(gt_items)}")
    print(f"    OCR Items: {len(pred_items)}")

    for i, gt_item in enumerate(gt_items):
        if i < len(pred_items):
            pred_item = pred_items[i]
            print(f"\n    GT Item {i+1}: {gt_item.get('item_name')} - qty:{gt_item.get('item_quantity')} price:{gt_item.get('per_item_price')} total:{gt_item.get('total_item_price')}")
            print(f"    OCR Item {i+1}: {pred_item.get('item_name')} - qty:{pred_item.get('item_quantity')} price:{pred_item.get('per_item_price')} total:{pred_item.get('total_item_price')}")
        else:
            print(f"\n    GT Item {i+1}: {gt_item.get('item_name')} - MISSING in OCR")

    return {
        'invoice_num': invoice_num,
        'ocr_result': ocr_result,
        'predictions': predictions,
        'confidences': confidences,
        'ground_truth': gt
    }

def main():
    print("OCR Invoice Extraction Test")
    print("=" * 60)

    # Test each invoice
    results = {}
    for i in range(1, 5):
        results[i] = test_ocr_on_invoice(i)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for i, result in results.items():
        if result:
            gt = result['ground_truth']
            pred = result['predictions']

            inv_match = pred.get('invoice_number') == str(gt.get('invoice_number'))
            date_match = pred.get('invoice_date') == gt.get('invoice_date')
            amt_match = abs((pred.get('invoice_amount') or 0) - gt.get('invoice_amount', 0)) < 1

            print(f"\nInvoice {i}:")
            print(f"  Invoice #: {'OK' if inv_match else 'FAIL'} ({gt.get('invoice_number')} vs {pred.get('invoice_number')})")
            print(f"  Date: {'OK' if date_match else 'FAIL'} ({gt.get('invoice_date')} vs {pred.get('invoice_date')})")
            print(f"  Amount: {'OK' if amt_match else 'FAIL'} ({gt.get('invoice_amount')} vs {pred.get('invoice_amount')})")
            print(f"  Items: {len(pred.get('items', []))}/{len(gt.get('items', []))} extracted")

if __name__ == "__main__":
    main()