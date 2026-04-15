import { useEffect, useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { Card, CardBody } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Spinner } from '../components/ui/Spinner';
import { Alert } from '../components/ui/Alert';
import { useVendorStore } from '../stores/vendorStore';
import { useInvoiceStore } from '../stores/invoiceStore';

export function VendorDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { currentVendor, fetchVendor, isLoading } = useVendorStore();
  const { invoices, fetchInvoices } = useInvoiceStore();

  useEffect(() => {
    if (id) {
      fetchVendor(id);
      fetchInvoices({ vendor_id: id, limit: 100 });
    }
  }, [id]);

  if (!currentVendor && isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Spinner size="lg" />
      </div>
    );
  }

  if (!currentVendor) {
    return (
      <Alert variant="error">Vendor not found</Alert>
    );
  }

  const vendorInvoices = invoices.filter(i => i.vendor_id === id);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{currentVendor.name}</h1>
          <p className="text-sm text-gray-500 mt-1">{currentVendor.description}</p>
        </div>
        <Button variant="secondary" onClick={() => navigate('/vendors')}>
          Back to Vendors
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardBody className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {currentVendor.training_samples || 0}
            </div>
            <div className="text-sm text-gray-500 mt-1">Training Samples</div>
          </CardBody>
        </Card>
        <Card>
          <CardBody className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {currentVendor.training_samples >= 1 ? 'Active' : 'No Data'}
            </div>
            <div className="text-sm text-gray-500 mt-1">Status</div>
          </CardBody>
        </Card>
      </div>

      <Alert variant="info">
        <strong>KNN Model:</strong> Each validation instantly improves predictions. More validated invoices = more accurate predictions. No manual retraining needed.
      </Alert>

      <Card>
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="font-semibold text-gray-800">Recent Invoices</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Invoice #</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Amount</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Date</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {vendorInvoices.slice(0, 10).map((invoice) => (
                <tr key={invoice.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <Link to={`/invoices/${invoice.id}`} className="text-blue-600 hover:text-blue-800">
                      {invoice.predictions?.invoice_number || invoice.id}
                    </Link>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    ${invoice.predictions?.invoice_amount?.toFixed(2) || '0.00'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      invoice.status === 'validated'
                        ? 'bg-green-100 text-green-800'
                        : 'bg-yellow-100 text-yellow-800'
                    }`}>
                      {invoice.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-gray-500 text-sm">
                    {new Date(invoice.created_at).toLocaleDateString()}
                  </td>
                </tr>
              ))}
              {vendorInvoices.length === 0 && (
                <tr>
                  <td colSpan="4" className="px-6 py-8 text-center text-gray-500">
                    No invoices for this vendor yet
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
