// frontend/js/command_palette/commands.js — command registry

import { marketState, setSymbol, setIntervalTF } from '../state/market.js';
import { uiState, setChatDock, setResearchDrawer } from '../state/ui.js';
import * as agentSvc from '../services/agent.js';
import { togglePanel as toggleExec } from '../execution/panel.js';

export function buildCommands() {
  const cmds = [];

  // Symbols (only show top 30 to keep palette fast)
  const topSymbols = (marketState.allSymbols || []).slice(0, 60);
  for (const sym of topSymbols) {
    cmds.push({
      id: `symbol:${sym}`,
      label: `Switch to ${sym}`,
      category: 'Symbol',
      action: () => setSymbol(sym),
    });
  }

  // Timeframes
  for (const tf of ['5m', '15m', '1h', '4h', '1d']) {
    cmds.push({
      id: `tf:${tf}`,
      label: `Timeframe ${tf}`,
      category: 'Timeframe',
      action: () => setIntervalTF(tf),
    });
  }

  // Panels
  cmds.push(
    { id: 'panel:execution', label: 'Open Execution Center', category: 'Panel', action: () => toggleExec() },
    { id: 'panel:research', label: 'Toggle Research Drawer', category: 'Panel', action: () => setResearchDrawer(!uiState.researchDrawerOpen) },
    { id: 'panel:chat', label: 'Toggle Chat Dock', category: 'Panel', action: () => setChatDock(!uiState.chatDockOpen) },
  );

  // Agent actions
  cmds.push(
    { id: 'agent:start', label: 'Start Agent', category: 'Agent', action: () => agentSvc.start() },
    { id: 'agent:stop', label: 'Stop Agent', category: 'Agent', action: () => agentSvc.stop() },
    { id: 'agent:revive', label: 'Revive Agent', category: 'Agent', action: () => agentSvc.revive() },
    { id: 'agent:scan', label: 'Scan Now', category: 'Agent', action: () => agentSvc.getSignals() },
    { id: 'agent:paper', label: 'Switch to Paper mode', category: 'Agent', action: () => agentSvc.setConfig({ mode: 'paper' }) },
    { id: 'agent:live', label: 'Switch to Live mode', category: 'Agent', action: () => agentSvc.setConfig({ mode: 'live' }) },
  );

  // Views
  cmds.push(
    { id: 'view:combat', label: 'Toggle Combat Mode (full screen)', category: 'View',
      action: () => document.body.classList.toggle('combat-mode') },
  );

  return cmds;
}
