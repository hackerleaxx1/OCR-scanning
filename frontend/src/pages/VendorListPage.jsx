import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import toast from 'react-hot-toast';
import { Card, CardBody } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Input, Select } from '../components/ui/Input';
import { Modal } from '../components/ui/Modal';
import { Spinner } from '../components/ui/Spinner';
import { useVendorStore } from '../stores/vendorStore';

export function VendorListPage() {
  const { vendors, fetchVendors, createVendor, deleteVendor, isLoading } = useVendorStore();
  const [showModal, setShowModal] = useState(false);
  const [newVendor, setNewVendor] = useState({ name: '', description: '' });

  useEffect(() => {
    fetchVendors();
  }, []);

  const handleCreate = async () => {
    if (!newVendor.name.trim()) {
      toast.error('Vendor name is required');
      return;
    }
    try {
      await createVendor(newVendor);
      setShowModal(false);
      setNewVendor({ name: '', description: '' });
      toast.success('Vendor created successfully');
    } catch (error) {
      toast.error(error.message || 'Failed to create vendor');
    }
  };

  const handleDelete = async (vendor) => {
    if (!confirm(`Delete vendor "${vendor.name}"? This will delete all training data.`)) return;
    try {
      await deleteVendor(vendor.id);
      toast.success('Vendor deleted');
    } catch (error) {
      toast.error(error.message || 'Failed to delete vendor');
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Vendors</h1>
        <Button onClick={() => setShowModal(true)}>+ Add Vendor</Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {vendors.map((vendor) => (
          <Card key={vendor.id}>
            <CardBody>
              <div>
                <h3 className="font-semibold text-gray-900">{vendor.name}</h3>
                <p className="text-sm text-gray-500 mt-1">{vendor.description || 'No description'}</p>
              </div>

              <div className="mt-4 pt-4 border-t border-gray-100 text-sm">
                <span className="text-gray-500">Training Samples:</span>
                <span className="ml-2 font-medium">{vendor.training_samples || 0}</span>
              </div>

              <div className="mt-4 flex gap-2">
                <Link to={`/vendors/${vendor.id}`} className="flex-1">
                  <Button variant="secondary" className="w-full" size="sm">
                    View Details
                  </Button>
                </Link>
                <Button variant="danger" size="sm" onClick={() => handleDelete(vendor)}>
                  Delete
                </Button>
              </div>
            </CardBody>
          </Card>
        ))}

        {vendors.length === 0 && !isLoading && (
          <div className="col-span-full text-center py-12 text-gray-500">
            No vendors yet. <Button variant="ghost" onClick={() => setShowModal(true)}>Create one</Button>
          </div>
        )}
      </div>

      <Modal isOpen={showModal} onClose={() => setShowModal(false)} title="Create Vendor">
        <div className="space-y-4">
          <Input
            label="Vendor Name"
            value={newVendor.name}
            onChange={(e) => setNewVendor((p) => ({ ...p, name: e.target.value }))}
            placeholder="e.g., Acme Corp"
          />
          <Input
            label="Description (optional)"
            value={newVendor.description}
            onChange={(e) => setNewVendor((p) => ({ ...p, description: e.target.value }))}
            placeholder="e.g., Monthly supply invoices"
          />
          <div className="flex justify-end gap-3 pt-4">
            <Button variant="secondary" onClick={() => setShowModal(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreate} isLoading={isLoading}>
              Create Vendor
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
