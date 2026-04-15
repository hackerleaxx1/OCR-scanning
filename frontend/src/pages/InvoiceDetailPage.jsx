import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { Card, CardBody } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Spinner } from '../components/ui/Spinner';
import { Alert } from '../components/ui/Alert';
import { useInvoiceStore } from '../stores/invoiceStore';
import { useVendorStore } from '../stores/vendorStore';

function ConfidenceBadge({ value }) {
  const color = value >= 0.8 ? 'green' : value >= 0.5 ? 'yellow' : 'red';
  return (
    <span className={`text-xs font-medium px-2 py-0.5 rounded bg-${color}-100 text-${color}-800`}>
      {(value * 100).toFixed(0)}%
    </span>
  );
}

export function InvoiceDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { currentInvoice, fetchInvoice, validateInvoice, isLoading } = useInvoiceStore();
  const { vendors } = useVendorStore();

  const [formData, setFormData] = useState({
    invoice_number: '',
    invoice_date: '',
    invoice_amount: '',
    items: [{ item_no: '', item_name: '', item_quantity: '', per_item_price: '', total_item_price: '' }],
  });

  const [validationResult, setValidationResult] = useState(null);

  useEffect(() => {
    if (id) {
      fetchInvoice(id);
    }
  }, [id]);

  useEffect(() => {
    if (currentInvoice && currentInvoice.predictions) {
      const p = currentInvoice.predictions;
      setFormData({
        invoice_number: p.invoice_number?.toString() || '',
        invoice_date: p.invoice_date?.toString() || '',
        invoice_amount: p.invoice_amount?.toString() || '',
        items: p.items?.length > 0 ? p.items : [{ item_no: '', item_name: '', item_quantity: '', per_item_price: '', total_item_price: '' }],
      });
    }
  }, [currentInvoice]);

  const handleChange = (field, value) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleItemChange = (index, field, value) => {
    setFormData((prev) => {
      const newItems = [...prev.items];
      newItems[index] = { ...newItems[index], [field]: value };
      return { ...prev, items: newItems };
    });
  };

  const addItem = () => {
    setFormData((prev) => ({
      ...prev,
      items: [...prev.items, { item_no: '', item_name: '', item_quantity: '', per_item_price: '', total_item_price: '' }],
    }));
  };

  const removeItem = (index) => {
    setFormData((prev) => ({
      ...prev,
      items: prev.items.filter((_, i) => i !== index),
    }));
  };

  const handleSubmit = async () => {
    try {
      const result = await validateInvoice(id, {
        invoice_number: formData.invoice_number,
        invoice_date: formData.invoice_date,
        invoice_amount: parseFloat(formData.invoice_amount) || 0,
        items: formData.items.map((item) => ({
          item_no: item.item_no || null,
          item_name: item.item_name,
          item_quantity: parseInt(item.item_quantity) || 0,
          per_item_price: parseFloat(item.per_item_price) || 0,
          total_item_price: parseFloat(item.total_item_price) || 0,
        })),
      });
      setValidationResult(result);
      toast.success('Validation submitted successfully!');
    } catch (error) {
      toast.error(error.message || 'Validation failed');
    }
  };

  if (!currentInvoice) {
    return (
      <div className="flex items-center justify-center h-64">
        <Spinner size="lg" />
      </div>
    );
  }

  const vendor = vendors.find((v) => v.id === currentInvoice.vendor_id);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Invoice Validation</h1>
          <p className="text-sm text-gray-500 mt-1">
            Vendor: {vendor?.name || currentInvoice.vendor_id}
          </p>
        </div>
        <div className="flex items-center gap-4">
          <span
            className={`px-3 py-1 rounded-full text-sm font-medium ${
              currentInvoice.status === 'validated'
                ? 'bg-green-100 text-green-800'
                : 'bg-yellow-100 text-yellow-800'
            }`}
          >
            {currentInvoice.status}
          </span>
          <Button variant="secondary" onClick={() => navigate('/invoices')}>
            Back to List
          </Button>
        </div>
      </div>

      {validationResult && (
        <Alert variant="success">{validationResult.message}</Alert>
      )}

      {currentInvoice.status === 'validated' ? (
        <Card>
          <CardBody>
            <Alert variant="info">
              This invoice has already been validated on{' '}
              {new Date(currentInvoice.validated_at).toLocaleString()}.
            </Alert>
            <div className="mt-4">
              <h3 className="font-semibold text-gray-800 mb-2">Invoice Details</h3>
              <div className="grid grid-cols-3 gap-4 mb-4">
                <div className="p-3 bg-gray-50 rounded">
                  <div className="text-xs font-medium text-gray-500 uppercase">Invoice Number</div>
                  <div className="text-gray-900">{currentInvoice.corrected_data?.invoice_number}</div>
                </div>
                <div className="p-3 bg-gray-50 rounded">
                  <div className="text-xs font-medium text-gray-500 uppercase">Invoice Date</div>
                  <div className="text-gray-900">{currentInvoice.corrected_data?.invoice_date}</div>
                </div>
                <div className="p-3 bg-gray-50 rounded">
                  <div className="text-xs font-medium text-gray-500 uppercase">Total Amount</div>
                  <div className="text-gray-900">${currentInvoice.corrected_data?.invoice_amount?.toFixed(2)}</div>
                </div>
              </div>

              <h3 className="font-semibold text-gray-800 mb-2">Line Items</h3>
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Item No.</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Item Name</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Quantity</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Per Unit Price</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Total Price</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {(currentInvoice.corrected_data?.items || []).map((item, idx) => (
                    <tr key={idx}>
                      <td className="px-4 py-2">{item.item_no || '-'}</td>
                      <td className="px-4 py-2">{item.item_name}</td>
                      <td className="px-4 py-2">{item.item_quantity}</td>
                      <td className="px-4 py-2">${item.per_item_price?.toFixed(2)}</td>
                      <td className="px-4 py-2">${item.total_item_price?.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardBody>
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Invoice Image */}
          <Card>
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="font-semibold text-gray-800">Invoice Image</h3>
            </div>
            <CardBody>
              {currentInvoice.image_path && (
                <iframe
                  src={`http://localhost:8000/uploads/${currentInvoice.image_path.split('/').pop()}`}
                  className="w-full h-96 rounded-lg shadow-md border-0"
                  title="Invoice PDF"
                />
              )}
              <div className="mt-4 p-3 bg-gray-50 rounded text-sm">
                <div className="font-medium text-gray-700 mb-1">OCR Raw Text:</div>
                <div className="text-gray-600 max-h-32 overflow-auto">
                  {currentInvoice.ocr_text || 'No text extracted'}
                </div>
              </div>
            </CardBody>
          </Card>

          {/* Validation Form */}
          <Card>
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="font-semibold text-gray-800">Invoice Details</h3>
              <p className="text-xs text-gray-500 mt-1">
                Correct any errors and submit to train the model
              </p>
            </div>
            <CardBody>
              <div className="space-y-4">
                {/* Header Fields */}
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <Input
                      label="Invoice Number"
                      value={formData.invoice_number}
                      onChange={(e) => handleChange('invoice_number', e.target.value)}
                    />
                    <div className="mt-1">
                      <ConfidenceBadge
                        value={currentInvoice.confidence_scores?.invoice_number || 0}
                      />
                    </div>
                  </div>
                  <div>
                    <Input
                      label="Invoice Date"
                      value={formData.invoice_date}
                      onChange={(e) => handleChange('invoice_date', e.target.value)}
                    />
                    <div className="mt-1">
                      <ConfidenceBadge
                        value={currentInvoice.confidence_scores?.invoice_date || 0}
                      />
                    </div>
                  </div>
                  <div>
                    <Input
                      label="Invoice Amount"
                      type="number"
                      step="0.01"
                      value={formData.invoice_amount}
                      onChange={(e) => handleChange('invoice_amount', e.target.value)}
                    />
                    <div className="mt-1">
                      <ConfidenceBadge
                        value={currentInvoice.confidence_scores?.invoice_amount || 0}
                      />
                    </div>
                  </div>
                </div>

                {/* Line Items */}
                <div className="mt-6">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-semibold text-gray-800">Line Items</h4>
                    <Button variant="secondary" size="sm" onClick={addItem}>
                      + Add Item
                    </Button>
                  </div>

                  <div className="space-y-3">
                    {formData.items.map((item, index) => {
                      const itemConf = currentInvoice.confidence_scores?.items?.[index] || {};
                      return (
                        <div key={index} className="p-4 bg-gray-50 rounded-lg">
                          <div className="flex items-start justify-between mb-2">
                            <span className="text-sm font-medium text-gray-700">Item {index + 1}</span>
                            {formData.items.length > 1 && (
                              <button
                                onClick={() => removeItem(index)}
                                className="text-red-500 hover:text-red-700 text-sm"
                              >
                                Remove
                              </button>
                            )}
                          </div>
                          <div className="grid grid-cols-5 gap-3">
                            <div>
                              <Input
                                label="Item No."
                                type="number"
                                value={item.item_no}
                                onChange={(e) => handleItemChange(index, 'item_no', e.target.value)}
                                placeholder="1"
                              />
                              <div className="mt-1">
                                <ConfidenceBadge value={itemConf.item_no || 0} />
                              </div>
                            </div>
                            <div>
                              <Input
                                label="Item Name"
                                value={item.item_name}
                                onChange={(e) => handleItemChange(index, 'item_name', e.target.value)}
                                placeholder="Product name"
                              />
                              <div className="mt-1">
                                <ConfidenceBadge value={itemConf.item_name || 0} />
                              </div>
                            </div>
                            <div>
                              <Input
                                label="Quantity"
                                type="number"
                                value={item.item_quantity}
                                onChange={(e) => handleItemChange(index, 'item_quantity', e.target.value)}
                                placeholder="0"
                              />
                              <div className="mt-1">
                                <ConfidenceBadge value={itemConf.item_quantity || 0} />
                              </div>
                            </div>
                            <div>
                              <Input
                                label="Per Unit Price"
                                type="number"
                                step="0.01"
                                value={item.per_item_price}
                                onChange={(e) => handleItemChange(index, 'per_item_price', e.target.value)}
                                placeholder="0.00"
                              />
                              <div className="mt-1">
                                <ConfidenceBadge value={itemConf.per_item_price || 0} />
                              </div>
                            </div>
                            <div>
                              <Input
                                label="Total Price"
                                type="number"
                                step="0.01"
                                value={item.total_item_price}
                                onChange={(e) => handleItemChange(index, 'total_item_price', e.target.value)}
                                placeholder="0.00"
                              />
                              <div className="mt-1">
                                <ConfidenceBadge value={itemConf.total_item_price || 0} />
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-6 border-t border-gray-200 flex justify-end gap-4">
                <Button variant="secondary" onClick={() => navigate('/invoices')}>
                  Cancel
                </Button>
                <Button onClick={handleSubmit} isLoading={isLoading}>
                  Submit Validation
                </Button>
              </div>
            </CardBody>
          </Card>
        </div>
      )}
    </div>
  );
}
