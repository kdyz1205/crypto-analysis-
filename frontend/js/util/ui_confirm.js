// frontend/js/util/ui_confirm.js
//
// Promise-based confirmation modal. Replacement for native confirm() on
// destructive paths where an unintended dismiss is dangerous.
//
// Why NOT native confirm():
//   - Native confirm() dismisses on ESC. That is safe for "cancel" but
//     our draw toolbar's ESC also exits draw-tool mode — a reflex ESC
//     after mis-clicking "清空" feels like an interaction we want
//     protected by an explicit button press, not a hardware key that
//     also means something else in the same panel.
//   - Native confirm() styling is out-of-place in the v2 dark UI and
//     blocks the main thread (sync).
//   - Native confirm() cannot say "type CLEAR to confirm" or colour the
//     button red — UX affordances we need for destructive actions.
//
// Contract:
//   - ESC does NOT dismiss. User must click a button.
//   - Backdrop click does NOT dismiss.
//   - Enter while confirm-button focused = confirm. Enter while cancel-
//     button focused = cancel. (Keyboard accessibility.)
//   - Modal stacks if called while another is open (Promises resolve
//     independently).
//
// Usage:
//   import { uiConfirm } from '../../util/ui_confirm.js';
//   const ok = await uiConfirm({
//     title: '清空所有画线?',
//     message: 'BTCUSDT 4h 上的所有手画线将被删除,无法恢复。',
//     confirmLabel: '清空',
//     destructive: true,
//   });
//   if (!ok) return;

let _stylesInjected = false;

function _injectStyles() {
  if (_stylesInjected) return;
  _stylesInjected = true;
  const s = document.createElement('style');
  s.setAttribute('data-ui-confirm-styles', '1');
  s.textContent = `
    .ui-confirm-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.6);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 10050;
      animation: ui-confirm-fade-in 120ms ease-out;
    }
    @keyframes ui-confirm-fade-in {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    .ui-confirm-box {
      background: var(--v2-panel, #1a1d24);
      border: 1px solid var(--v2-border, #2b2f36);
      border-radius: 8px;
      padding: 20px 24px 18px;
      min-width: 320px;
      max-width: 480px;
      box-shadow: 0 16px 48px rgba(0, 0, 0, 0.45);
      color: var(--v2-text, #e6e6e6);
      font-family: inherit;
    }
    .ui-confirm-title {
      font-size: 14px;
      font-weight: 600;
      margin: 0 0 8px 0;
    }
    .ui-confirm-message {
      font-size: 12px;
      color: var(--v2-muted, #9ca3af);
      margin: 0 0 18px 0;
      line-height: 1.5;
      white-space: pre-wrap;
      word-wrap: break-word;
    }
    .ui-confirm-actions {
      display: flex;
      gap: 8px;
      justify-content: flex-end;
    }
    .ui-confirm-btn {
      padding: 6px 16px;
      border-radius: 4px;
      border: 1px solid var(--v2-border, #2b2f36);
      background: var(--v2-bg, #0e1013);
      color: var(--v2-text, #e6e6e6);
      font-size: 12px;
      font-family: inherit;
      cursor: pointer;
      transition: border-color 120ms, background 120ms;
    }
    .ui-confirm-btn:hover {
      border-color: var(--v2-text, #e6e6e6);
    }
    .ui-confirm-btn:focus {
      outline: 2px solid rgba(96, 165, 250, 0.5);
      outline-offset: 1px;
    }
    .ui-confirm-btn-primary {
      background: var(--v2-green, #22c55e);
      border-color: var(--v2-green, #22c55e);
      color: #0a0a0a;
      font-weight: 600;
    }
    .ui-confirm-btn-primary:hover {
      filter: brightness(1.1);
    }
    .ui-confirm-btn-danger {
      background: var(--v2-red, #ef4444);
      border-color: var(--v2-red, #ef4444);
      color: #ffffff;
      font-weight: 600;
    }
    .ui-confirm-btn-danger:hover {
      filter: brightness(1.1);
    }
  `;
  document.head.appendChild(s);
}

/**
 * @param {object} opts
 * @param {string} [opts.title='确认'] - dialog title
 * @param {string} [opts.message=''] - body text (supports \n)
 * @param {string} [opts.confirmLabel='确认']
 * @param {string} [opts.cancelLabel='取消']
 * @param {boolean} [opts.destructive=false] - red confirm button
 * @returns {Promise<boolean>} true if confirmed, false if cancelled
 */
export function uiConfirm({
  title = '确认',
  message = '',
  confirmLabel = '确认',
  cancelLabel = '取消',
  destructive = false,
} = {}) {
  _injectStyles();

  return new Promise((resolve) => {
    const backdrop = document.createElement('div');
    backdrop.className = 'ui-confirm-backdrop';
    backdrop.setAttribute('data-testid', 'ui-confirm');

    const confirmClass = destructive
      ? 'ui-confirm-btn ui-confirm-btn-danger'
      : 'ui-confirm-btn ui-confirm-btn-primary';

    // Build DOM with textContent (no innerHTML with user strings) to
    // defang XSS — title/message may include symbol names coming from
    // state which, while normally clean, we do not want to trust.
    const box = document.createElement('div');
    box.className = 'ui-confirm-box';
    box.setAttribute('role', 'dialog');
    box.setAttribute('aria-modal', 'true');

    const titleEl = document.createElement('div');
    titleEl.className = 'ui-confirm-title';
    titleEl.textContent = title;

    const msgEl = document.createElement('div');
    msgEl.className = 'ui-confirm-message';
    msgEl.textContent = message;

    const actions = document.createElement('div');
    actions.className = 'ui-confirm-actions';

    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'ui-confirm-btn';
    cancelBtn.textContent = cancelLabel;
    cancelBtn.setAttribute('data-testid', 'ui-confirm-cancel');

    const confirmBtn = document.createElement('button');
    confirmBtn.className = confirmClass;
    confirmBtn.textContent = confirmLabel;
    confirmBtn.setAttribute('data-testid', 'ui-confirm-ok');

    actions.appendChild(cancelBtn);
    actions.appendChild(confirmBtn);
    box.appendChild(titleEl);
    if (message) box.appendChild(msgEl);
    box.appendChild(actions);
    backdrop.appendChild(box);

    document.body.appendChild(backdrop);

    let done = false;
    const cleanup = () => {
      if (done) return;
      done = true;
      document.removeEventListener('keydown', onKey, true);
      try { document.body.removeChild(backdrop); } catch {}
    };

    const onYes = () => { cleanup(); resolve(true); };
    const onNo = () => { cleanup(); resolve(false); };

    // Keyboard handling — capture phase so we consume ESC before any
    // other document-level keydown handler (e.g. chart_drawing.js Esc =
    // exit draw mode). Capture prevents the modal from being indirectly
    // auto-dismissed.
    const onKey = (e) => {
      if (e.key === 'Escape') {
        // ESC does NOT dismiss. Swallow it so draw-mode's ESC handler
        // doesn't fire either while a confirm is on screen.
        e.preventDefault();
        e.stopPropagation();
        return;
      }
      if (e.key === 'Enter') {
        const focused = document.activeElement;
        if (focused === cancelBtn) {
          e.preventDefault();
          e.stopPropagation();
          onNo();
        } else {
          // Default: Enter confirms. Safe because confirm-button is
          // auto-focused on open.
          e.preventDefault();
          e.stopPropagation();
          onYes();
        }
      }
    };

    cancelBtn.addEventListener('click', onNo);
    confirmBtn.addEventListener('click', onYes);

    // Backdrop click does NOT dismiss — explicit button press required.
    backdrop.addEventListener('click', (e) => {
      if (e.target === backdrop) {
        e.preventDefault();
        e.stopPropagation();
      }
    });

    document.addEventListener('keydown', onKey, true);

    // Focus confirm button so keyboard users can Enter-to-confirm.
    try { confirmBtn.focus(); } catch {}
  });
}
