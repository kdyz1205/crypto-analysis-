import { publish } from '../util/events.js';

export const strategyState = {
  config: null,
  snapshotEnvelope: null,
  replayEnvelope: null,
  error: null,
  layerVisibility: {
    primaryTrendlines: true,
    debugTrendlines: false,
    confirmingTouches: true,
    barTouches: false,
    projectedLine: true,
    signalMarkers: true,
    collapsedInvalidations: true,
    orderMarkers: false,
  },
};

export function setStrategyConfig(config) {
  strategyState.config = config;
  publish('strategy.config.updated', config);
}

export function setStrategySnapshot(snapshotEnvelope) {
  strategyState.snapshotEnvelope = snapshotEnvelope;
  publish('strategy.snapshot.updated', snapshotEnvelope);
}

export function clearStrategySnapshot() {
  strategyState.snapshotEnvelope = null;
  publish('strategy.snapshot.updated', null);
}

export function setStrategyReplay(replayEnvelope) {
  strategyState.replayEnvelope = replayEnvelope;
  publish('strategy.replay.updated', replayEnvelope);
}

export function clearStrategyReplay() {
  strategyState.replayEnvelope = null;
  publish('strategy.replay.updated', null);
}

export function setStrategyError(error) {
  strategyState.error = error ? String(error) : null;
  publish('strategy.error.updated', strategyState.error);
}

export function setStrategyLayerVisible(layer, visible) {
  if (!(layer in strategyState.layerVisibility)) return;
  const next = !!visible;
  if (strategyState.layerVisibility[layer] === next) return;
  strategyState.layerVisibility[layer] = next;
  publish('strategy.layers.changed', { ...strategyState.layerVisibility });
}

export function toggleStrategyLayer(layer) {
  if (!(layer in strategyState.layerVisibility)) return;
  setStrategyLayerVisible(layer, !strategyState.layerVisibility[layer]);
}

export function getCurrentStrategySnapshot() {
  return strategyState.snapshotEnvelope?.snapshot || null;
}
