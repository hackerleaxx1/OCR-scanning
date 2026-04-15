import api from '../api/client';

export const vendorApi = {
  list: async () => {
    const response = await api.get('/vendors');
    return response.data;
  },

  create: async (data) => {
    const response = await api.post('/vendors', data);
    return response.data;
  },

  get: async (id) => {
    const response = await api.get(`/vendors/${id}`);
    return response.data;
  },

  retrain: async (id, options = {}) => {
    const response = await api.post(`/vendors/${id}/retrain`, options);
    return response.data;
  },

  delete: async (id) => {
    const response = await api.delete(`/vendors/${id}`);
    return response.data;
  },
};
