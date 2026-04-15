import { create } from 'zustand';
import { vendorApi } from '../api/vendors';

export const useVendorStore = create((set, get) => ({
  vendors: [],
  currentVendor: null,
  isLoading: false,
  error: null,

  fetchVendors: async () => {
    set({ isLoading: true, error: null });
    try {
      const data = await vendorApi.list();
      set({ vendors: data.vendors || [], isLoading: false });
    } catch (error) {
      set({ error: error.message, isLoading: false });
    }
  },

  fetchVendor: async (id) => {
    set({ isLoading: true, error: null });
    try {
      const vendor = await vendorApi.get(id);
      set({ currentVendor: vendor, isLoading: false });
      return vendor;
    } catch (error) {
      set({ error: error.message, isLoading: false });
      throw error;
    }
  },

  createVendor: async (data) => {
    set({ isLoading: true, error: null });
    try {
      const vendor = await vendorApi.create(data);
      set((state) => ({
        vendors: [vendor, ...state.vendors],
        isLoading: false
      }));
      return vendor;
    } catch (error) {
      set({ error: error.message, isLoading: false });
      throw error;
    }
  },

  retrainVendor: async (id, options = {}) => {
    set({ isLoading: true, error: null });
    try {
      const result = await vendorApi.retrain(id, options);
      set({ isLoading: false });
      return result;
    } catch (error) {
      set({ error: error.message, isLoading: false });
      throw error;
    }
  },

  deleteVendor: async (id) => {
    set({ isLoading: true, error: null });
    try {
      await vendorApi.delete(id);
      set((state) => ({
        vendors: state.vendors.filter((v) => v.id !== id),
        isLoading: false
      }));
    } catch (error) {
      set({ error: error.message, isLoading: false });
      throw error;
    }
  },
}));
