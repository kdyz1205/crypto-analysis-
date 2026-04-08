// frontend/js/services/onchain.js
import { fetchJson } from '../util/fetch.js';

export const getHealth = () => fetchJson('/api/onchain/health');
export const getWallets = () => fetchJson('/api/onchain/wallets');
export const getSmartMoney = () => fetchJson('/api/onchain/wallets/smart-money');
export const getWallet = (address) => fetchJson(`/api/onchain/wallets/${address}`);
export const trackWallet = (address) =>
  fetchJson(`/api/onchain/wallets/track/${address}`, { method: 'POST' });
export const untrackWallet = (address) =>
  fetchJson(`/api/onchain/wallets/track/${address}`, { method: 'DELETE' });
export const getSignals = (limit = 50) => fetchJson(`/api/onchain/signals?limit=${limit}`);
export const getRecommendations = (limit = 20) =>
  fetchJson(`/api/onchain/signals/recommendations?limit=${limit}`);

export const tokenAnalyze = (address, network = 'solana', pool = null) => {
  const params = new URLSearchParams({ network });
  if (pool) params.set('pool', pool);
  return fetchJson(`/api/onchain/token/analyze/${address}?${params}`);
};
