import api from '../api/client';

export const invoiceApi = {
  upload: async (formData) => {
    const response = await api.post('/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  list: async (params = {}) => {
    const response = await api.get('/invoices', { params });
    return response.data;
  },

  get: async (id) => {
    const response = await api.get(`/invoices/${id}`);
    return response.data;
  },

  validate: async (id, data) => {
    const response = await api.post(`/invoices/${id}/validate`, data);
    return response.data;
  },
};
