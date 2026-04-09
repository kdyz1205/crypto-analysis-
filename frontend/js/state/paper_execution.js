export const paperExecutionState = {
  state: null,
  config: null,
  lastError: null,
  lastStepResult: null,
  loadingState: false,
  loadingConfig: false,
  mutating: false,
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

export function setPaperExecutionMutating(loading) {
  paperExecutionState.mutating = !!loading;
}
