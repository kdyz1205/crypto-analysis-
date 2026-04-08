// frontend/js/state/agent.js
import { publish } from '../util/events.js';

export const agentState = {
  panelOpen: false,
  activeSubTab: 'overview',  // 'overview' | 'execution' | 'risk' | 'ops'
  pollTimer: null,
  pollInFlight: false,
  lastStatus: null,
  lastOKXBalanceFetch: 0,
};

export function setPanelOpen(open) {
  if (agentState.panelOpen === open) return;
  agentState.panelOpen = open;
  publish('agent.panel.toggled', open);
}

export function setActiveSubTab(tab) {
  if (agentState.activeSubTab === tab) return;
  agentState.activeSubTab = tab;
  publish('agent.subtab.changed', tab);
}

export function setLastStatus(status) {
  agentState.lastStatus = status;
  publish('agent.status.updated', status);
}
