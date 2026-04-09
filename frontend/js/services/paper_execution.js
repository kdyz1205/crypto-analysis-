import { fetchJson } from '../util/fetch.js';

let activeStepController = null;

export async function getPaperExecutionState() {
  const response = await fetchJson('/api/paper-execution/state');
  return response.state;
}

export async function getPaperExecutionConfig() {
  const response = await fetchJson('/api/paper-execution/config');
  return response.config;
}

export async function setPaperExecutionConfig(payload) {
  const response = await fetchJson('/api/paper-execution/config', {
    method: 'POST',
    body: payload,
  });
  return response.config;
}

export async function resetPaperExecution(payload = {}) {
  const response = await fetchJson('/api/paper-execution/reset', {
    method: 'POST',
    body: payload,
  });
  return response.state;
}

export async function stepPaperExecution(payload) {
  activeStepController?.abort();
  activeStepController = new AbortController();
  try {
    return await fetchJson('/api/paper-execution/step', {
      method: 'POST',
      body: payload,
      timeout: 120000,
      signal: activeStepController.signal,
    });
  } finally {
    activeStepController = null;
  }
}

export async function setPaperKillSwitch(payload) {
  const response = await fetchJson('/api/paper-execution/kill-switch', {
    method: 'POST',
    body: payload,
  });
  return response.state;
}

export function abortPaperExecutionStep() {
  activeStepController?.abort();
  activeStepController = null;
}
