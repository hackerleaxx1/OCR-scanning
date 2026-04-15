import { useCallback, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import toast from 'react-hot-toast';
import { Card, CardBody } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Select } from '../components/ui/Input';
import { Spinner } from '../components/ui/Spinner';
import { useInvoiceStore } from '../stores/invoiceStore';
import { useVendorStore } from '../stores/vendorStore';

export function UploadPage() {
  const navigate = useNavigate();
  const { uploadInvoice, isLoading } = useInvoiceStore();
  const { vendors, fetchVendors } = useVendorStore();
  const [selectedVendor, setSelectedVendor] = useState('');
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);

  useState(() => {
    fetchVendors();
  }, []);

  const onDrop = useCallback((acceptedFiles) => {
    const f = acceptedFiles[0];
    setFile(f);
    if (f.type === 'application/pdf') {
      // For PDFs, create an embed preview
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target.result);
      reader.readAsDataURL(f);
    } else {
      // For images, use URL.createObjectURL for better preview
      const url = URL.createObjectURL(f);
      setPreview(url);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.pdf'] },
    maxFiles: 1,
  });

  const handleSubmit = async () => {
    if (!file) {
      toast.error('Please select a file');
      return;
    }
    if (!selectedVendor && vendors.length > 0) {
      toast.error('Please select a vendor');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    if (selectedVendor) {
      formData.append('vendor_id', selectedVendor);
    }

    try {
      const invoice = await uploadInvoice(formData);
      toast.success('Invoice uploaded successfully!');
      navigate(`/invoices/${invoice.id}`);
    } catch (error) {
      toast.error(error.message || 'Upload failed');
    }
  };

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <Card>
        <CardBody>
          <h2 className="text-lg font-semibold mb-4">Upload Invoice</h2>

          {/* Vendor Selection */}
          {vendors.length > 0 && (
            <div className="mb-6">
              <Select
                label="Select Vendor"
                value={selectedVendor}
                onChange={(e) => setSelectedVendor(e.target.value)}
                options={[
                  { value: '', label: '-- Select Vendor --' },
                  ...vendors.map((v) => ({ value: v.id, label: v.name })),
                ]}
              />
            </div>
          )}

          {/* Dropzone */}
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
              isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
            }`}
          >
            <input {...getInputProps()} />
            {preview ? (
              <div className="space-y-4">
                {file?.type === 'application/pdf' ? (
                  <embed src={preview} type="application/pdf" className="w-full h-64 rounded-lg shadow-md" />
                ) : (
                  <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded-lg shadow-md" />
                )}
                <p className="text-sm text-gray-500">{file?.name}</p>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="mx-auto w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center">
                  <svg className="h-8 w-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
                <div>
                  <p className="text-gray-700 font-medium">Drop invoice image here or click to select</p>
                  <p className="text-sm text-gray-500 mt-1">Supports JPG, PNG, PDF</p>
                </div>
              </div>
            )}
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-4 mt-6">
            <Button variant="secondary" onClick={() => navigate('/')}>
              Cancel
            </Button>
            <Button onClick={handleSubmit} disabled={!file || isLoading}>
              {isLoading ? <Spinner size="sm" /> : 'Upload & Process'}
            </Button>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
