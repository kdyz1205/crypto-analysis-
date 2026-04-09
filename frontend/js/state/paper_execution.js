export const paperExecutionState = {
  state: null,
  config: null,
  lastError: null,
  lastStepResult: null,
  loadingState: false,
  loadingConfig: false,
  stepping: false,
  resetting: false,
  killSwitchUpdating: false,
  autoRefreshEnabled: false,
};

export function setPaperExecutionState(state) {
  paperExecutionState.state = state;
}

export function setPaperExecutionConfig(config) {
  paperExecutionState.config = config;
}

export function setPaperExecutionError(error) {
  paperExecutionState.lastError = error;
}

export function setPaperExecutionLastStep(stepResult) {
  paperExecutionState.lastStepResult = stepResult;
}

export function clearPaperExecutionError() {
  paperExecutionState.lastError = null;
}

export function setPaperExecutionLoadingState(loading) {
  paperExecutionState.loadingState = !!loading;
}

export function setPaperExecutionLoadingConfig(loading) {
  paperExecutionState.loadingConfig = !!loading;
}

export function setPaperExecutionStepping(loading) {
  paperExecutionState.stepping = !!loading;
}

export function setPaperExecutionResetting(loading) {
  paperExecutionState.resetting = !!loading;
}

export function setPaperExecutionKillSwitchUpdating(loading) {
  paperExecutionState.killSwitchUpdating = !!loading;
}

export function isPaperExecutionBusy() {
  return !!(paperExecutionState.stepping || paperExecutionState.resetting || paperExecutionState.killSwitchUpdating);
}
