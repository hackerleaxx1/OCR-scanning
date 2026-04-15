import { useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { Card } from '../components/ui/Card';
import { Select } from '../components/ui/Input';
import { Button } from '../components/ui/Button';
import { useInvoiceStore } from '../stores/invoiceStore';
import { useVendorStore } from '../stores/vendorStore';

export function InvoiceListPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const statusFilter = searchParams.get('status') || 'all';
  const vendorFilter = searchParams.get('vendor_id') || '';

  const { invoices, pagination, fetchInvoices, filters, setFilters } = useInvoiceStore();
  const { vendors, fetchVendors } = useVendorStore();

  useEffect(() => {
    fetchVendors();
  }, []);

  useEffect(() => {
    fetchInvoices({
      status: statusFilter,
      vendor_id: vendorFilter || null,
    });
  }, [statusFilter, vendorFilter]);

  const handleStatusChange = (e) => {
    setSearchParams({ status: e.target.value, vendor_id: vendorFilter });
  };

  const handleVendorChange = (e) => {
    setSearchParams({ status: statusFilter, vendor_id: e.target.value });
  };

  return (
    <div className="space-y-6">
      {/* Filters */}
      <Card>
        <div className="px-6 py-4 flex items-center gap-6">
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-gray-700">Status:</label>
            <Select
              value={statusFilter}
              onChange={handleStatusChange}
              options={[
                { value: 'all', label: 'All' },
                { value: 'pending', label: 'Pending' },
                { value: 'validated', label: 'Validated' },
              ]}
            />
          </div>

          {vendors.length > 0 && (
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium text-gray-700">Vendor:</label>
              <Select
                value={vendorFilter}
                onChange={handleVendorChange}
                options={[
                  { value: '', label: 'All Vendors' },
                  ...vendors.map((v) => ({ value: v.id, label: v.name })),
                ]}
              />
            </div>
          )}

          <div className="ml-auto text-sm text-gray-500">
            {pagination.total} invoice(s)
          </div>
        </div>
      </Card>

      {/* Invoice Table */}
      <Card>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Invoice #</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Vendor</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Amount</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Item</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Date</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {invoices.map((invoice) => (
                <tr key={invoice.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <Link
                      to={`/invoices/${invoice.id}`}
                      className="text-blue-600 hover:text-blue-800 font-medium"
                    >
                      {invoice.predictions?.invoice_number || invoice.id}
                    </Link>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-gray-600">
                    {vendors.find((v) => v.id === invoice.vendor_id)?.name || invoice.vendor_id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-gray-600">
                    ${invoice.predictions?.invoice_amount?.toFixed(2) || '0.00'}
                  </td>
                  <td className="px-6 py-4 text-gray-600 max-w-xs truncate">
                    {invoice.predictions?.items?.[0]?.item_name || '-'}
                    {invoice.predictions?.items?.length > 1 && ` (+${invoice.predictions.items.length - 1} more)`}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span
                      className={`px-2 py-1 rounded-full text-xs font-medium ${
                        invoice.status === 'validated'
                          ? 'bg-green-100 text-green-800'
                          : 'bg-yellow-100 text-yellow-800'
                      }`}
                    >
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
                  <td colSpan="6" className="px-6 py-12 text-center text-gray-500">
                    No invoices found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {pagination.pages > 1 && (
          <div className="px-6 py-4 border-t border-gray-200 flex items-center justify-between">
            <Button
              variant="secondary"
              size="sm"
              disabled={pagination.page <= 1}
              onClick={() =>
                fetchInvoices({ page: pagination.page - 1 })
              }
            >
              Previous
            </Button>
            <span className="text-sm text-gray-500">
              Page {pagination.page} of {pagination.pages}
            </span>
            <Button
              variant="secondary"
              size="sm"
              disabled={pagination.page >= pagination.pages}
              onClick={() =>
                fetchInvoices({ page: pagination.page + 1 })
              }
            >
              Next
            </Button>
          </div>
        )}
      </Card>
    </div>
  );
}
