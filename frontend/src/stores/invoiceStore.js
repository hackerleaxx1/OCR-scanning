import { create } from 'zustand';
import { invoiceApi } from '../api/invoices';

export const useInvoiceStore = create((set, get) => ({
  invoices: [],
  currentInvoice: null,
  pagination: { page: 1, limit: 20, total: 0, pages: 0 },
  filters: { status: 'all', vendor_id: null },
  isLoading: false,
  error: null,

  fetchInvoices: async (params = {}) => {
    set({ isLoading: true, error: null });
    try {
      const filters = get().filters;
      const mergedParams = { ...filters, ...params };
      const data = await invoiceApi.list(mergedParams);
      set({
        invoices: data.invoices,
        pagination: data.pagination,
        isLoading: false
      });
    } catch (error) {
      set({ error: error.message, isLoading: false });
    }
  },

  fetchInvoice: async (id) => {
    set({ isLoading: true, error: null });
    try {
      const invoice = await invoiceApi.get(id);
      set({ currentInvoice: invoice, isLoading: false });
      return invoice;
    } catch (error) {
      set({ error: error.message, isLoading: false });
      throw error;
    }
  },

  uploadInvoice: async (formData) => {
    set({ isLoading: true, error: null });
    try {
      const invoice = await invoiceApi.upload(formData);
      set({ isLoading: false });
      return invoice;
    } catch (error) {
      set({ error: error.message, isLoading: false });
      throw error;
    }
  },

  validateInvoice: async (id, data) => {
    set({ isLoading: true, error: null });
    try {
      const result = await invoiceApi.validate(id, data);
      set({ isLoading: false });
      return result;
    } catch (error) {
      set({ error: error.message, isLoading: false });
      throw error;
    }
  },

  setFilters: (filters) => {
    set((state) => ({ filters: { ...state.filters, ...filters } }));
  },

  setCurrentInvoice: (invoice) => set({ currentInvoice: invoice }),
}));
