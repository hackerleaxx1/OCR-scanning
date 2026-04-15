import { Link } from 'react-router-dom';
import { Card, CardBody } from '../components/ui/Card';
import { useInvoiceStore } from '../stores/invoiceStore';
import { useVendorStore } from '../stores/vendorStore';
import { useEffect } from 'react';

export function Dashboard() {
  const { invoices, fetchInvoices } = useInvoiceStore();
  const { vendors, fetchVendors } = useVendorStore();

  useEffect(() => {
    fetchInvoices({ limit: 100 });
    fetchVendors();
  }, []);

  const pendingCount = invoices.filter(i => i.status === 'pending').length;
  const validatedCount = invoices.filter(i => i.status === 'validated').length;

  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardBody className="text-center">
            <div className="text-3xl font-bold text-blue-600">{invoices.length}</div>
            <div className="text-sm text-gray-500 mt-1">Total Invoices</div>
          </CardBody>
        </Card>
        <Card>
          <CardBody className="text-center">
            <div className="text-3xl font-bold text-yellow-600">{pendingCount}</div>
            <div className="text-sm text-gray-500 mt-1">Pending Validation</div>
          </CardBody>
        </Card>
        <Card>
          <CardBody className="text-center">
            <div className="text-3xl font-bold text-green-600">{validatedCount}</div>
            <div className="text-sm text-gray-500 mt-1">Validated</div>
          </CardBody>
        </Card>
        <Card>
          <CardBody className="text-center">
            <div className="text-3xl font-bold text-purple-600">{vendors.length}</div>
            <div className="text-sm text-gray-500 mt-1">Vendors</div>
          </CardBody>
        </Card>
      </div>

      {/* Pending Validations Alert */}
      {pendingCount > 0 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-yellow-800">
                Invoices Pending Validation
              </h3>
              <p className="text-sm text-yellow-700 mt-1">
                {pendingCount} invoice(s) need validation. Train the ML model by correcting predictions.
              </p>
            </div>
            <div className="ml-auto">
              <Link
                to="/invoices?status=pending"
                className="text-sm font-medium text-yellow-800 hover:text-yellow-900"
              >
                View All →
              </Link>
            </div>
          </div>
        </div>
      )}

      {/* Recent Invoices */}
      <Card>
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-800">Recent Invoices</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Invoice #</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Vendor</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Amount</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Date</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {invoices.slice(0, 5).map((invoice) => (
                <tr key={invoice.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <Link to={`/invoices/${invoice.id}`} className="text-blue-600 hover:text-blue-800 font-medium">
                      {invoice.predictions?.invoice_number || invoice.id}
                    </Link>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-gray-600">
                    {vendors.find(v => v.id === invoice.vendor_id)?.name || invoice.vendor_id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-gray-600">
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
              {invoices.length === 0 && (
                <tr>
                  <td colSpan="5" className="px-6 py-8 text-center text-gray-500">
                    No invoices yet. <Link to="/upload" className="text-blue-600 hover:underline">Upload one</Link>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Link to="/upload">
          <Card className="hover:shadow-md transition-shadow cursor-pointer">
            <CardBody className="flex items-center">
              <div className="p-3 bg-blue-100 rounded-lg mr-4">
                <svg className="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <div>
                <h4 className="font-semibold text-gray-800">Upload Invoice</h4>
                <p className="text-sm text-gray-500">Upload a new invoice for OCR processing</p>
              </div>
            </CardBody>
          </Card>
        </Link>
        <Link to="/vendors">
          <Card className="hover:shadow-md transition-shadow cursor-pointer">
            <CardBody className="flex items-center">
              <div className="p-3 bg-purple-100 rounded-lg mr-4">
                <svg className="h-6 w-6 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                </svg>
              </div>
              <div>
                <h4 className="font-semibold text-gray-800">Manage Vendors</h4>
                <p className="text-sm text-gray-500">Add vendors and retrain ML models</p>
              </div>
            </CardBody>
          </Card>
        </Link>
      </div>
    </div>
  );
}
