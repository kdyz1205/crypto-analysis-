// frontend/js/state/ui.js
import { publish } from '../util/events.js';

export const uiState = {
  activeLayer: 'workbench',      // 'workbench' (default)
  chatDockOpen: false,
  onchainPanelOpen: false,
  researchDrawerOpen: false,
  researchActiveSubTab: 'backtest',  // 'backtest' | 'replay' | 'similar' | 'ribbon'
  opsDrawerOpen: false,
};

export function setActiveLayer(layer) {
  if (uiState.activeLayer === layer) return;
  uiState.activeLayer = layer;
  publish('ui.layer.changed', layer);
}

export function setChatDock(open) {
  if (uiState.chatDockOpen === open) return;
  uiState.chatDockOpen = open;
  publish('ui.chat.toggled', open);
}

export function setOnchainPanel(open) {
  if (uiState.onchainPanelOpen === open) return;
  uiState.onchainPanelOpen = open;
  publish('ui.onchain.toggled', open);
}

export function setResearchDrawer(open) {
  if (uiState.researchDrawerOpen === open) return;
  uiState.researchDrawerOpen = open;
  publish('ui.research.toggled', open);
}

export function setResearchSubTab(tab) {
  if (uiState.researchActiveSubTab === tab) return;
  uiState.researchActiveSubTab = tab;
  publish('ui.research.subtab', tab);
}

export function setOpsDrawer(open) {
  if (uiState.opsDrawerOpen === open) return;
  uiState.opsDrawerOpen = open;
  publish('ui.ops.toggled', open);
}
