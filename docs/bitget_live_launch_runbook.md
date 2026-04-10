# Bitget Live Launch Runbook

Updated: 2026-04-10

## 1. Purpose

This runbook is the final operator checklist for moving the current system from:

1. paper-first verified mode
2. demo / dry-run live bridge validation
3. live bridge preflight

to:

4. manually confirmed Bitget real-money submit

This document does not authorize automatic live looping.
It only defines the safe manual path.

## 2. Required Credentials

Bitget requires all three credentials:

1. `BITGET_API_KEY`
2. `BITGET_SECRET_KEY`
3. `BITGET_PASSPHRASE`

Without all three, the live bridge must remain blocked.

## 3. Required Environment Flags

### Demo / dry-run validation

```text
ENABLE_LIVE_TRADING=true
DRY_RUN=true
CONFIRM_LIVE_TRADING=false
```

### Real-money submit

```text
ENABLE_LIVE_TRADING=true
DRY_RUN=false
CONFIRM_LIVE_TRADING=true
```

If any one of those conditions is not satisfied, `mode=live` must remain blocked.

## 4. Validated Runtime Preset

Current validated preset:

1. symbols: `HYPEUSDT`, `RIVERUSDT`
2. timeframe: `1h`
3. trigger mode: `pre_limit`
4. strategy lookback: `80`
5. strategy window: `100`

Current live default envs are aligned to that preset:

```text
LIVE_ALLOWED_SYMBOLS=HYPEUSDT,RIVERUSDT
LIVE_ALLOWED_TIMEFRAMES=1h
LIVE_ALLOWED_TRIGGER_MODES=pre_limit
LIVE_MAX_POSITIONS=1
LIVE_MAX_NOTIONAL=100
LIVE_RECONCILIATION_MAX_AGE_SECONDS=300
```

Do not widen that whitelist before re-validating the expanded scope.

## 5. Mandatory Launch Sequence

For every session:

1. start the app
2. open `/v2`
3. open `Execution Center`
4. confirm paper-first path is healthy
5. confirm the desired paper intent exists
6. run `Live Preflight`
7. run `Reconcile`
8. run `Preview selected intent`
9. if testing demo: `Submit to demo`
10. if and only if all live preflight checks pass and you intend real money: `Submit to live`

Never skip preflight and reconcile.

## 6. What Must Be Green Before Real Money

The `Live Preflight` block must report:

1. `ENABLE_LIVE_TRADING` passed
2. `Bitget API keys` passed
3. `CONFIRM_LIVE_TRADING` passed
4. `DRY_RUN disabled` passed
5. `Fresh reconciliation` passed
6. `Reconciliation not stale` passed
7. `Reconciliation unblocked` passed
8. `Live-eligible intent selected` passed
9. `Intent passes live gating` passed

If any blocking check fails, do not submit live.

## 7. Reconciliation Rules

Reconciliation is required before submit.

It must be:

1. present
2. fresh within `LIVE_RECONCILIATION_MAX_AGE_SECONDS`
3. unblocked

If reconciliation is stale, rerun it before any submit.

## 8. Data Visibility Rules

Before trusting a signal, verify the history coverage shown in:

1. strategy snapshot
2. paper step
3. runtime last tick

Do not assume `full_history` means "full exchange universe forever".
Use the displayed values:

1. `dataSourceMode`
2. `dataSourceKind`
3. `loadedBarCount`
4. `analysisInputBarCount`
5. `analysisWasTrimmed`
6. `listingStartTimestamp`

## 9. Hard Stop Conditions

Do not submit live if any of the following are true:

1. preflight is blocked
2. reconciliation is blocked
3. selected intent falls outside the validated whitelist
4. strategy history coverage is more truncated than expected
5. paper step behavior differs from expected preset behavior
6. runtime kill switch is active
7. Bitget credentials are missing or unverified

## 10. Current Boundaries

The system is ready for:

1. manual preflight
2. manual reconcile
3. manual preview
4. manual Bitget demo submit
5. manual Bitget real-money submit after all checks pass

The system is not yet approved for:

1. automatic real-money loop
2. multi-user orchestration
3. database-backed recovery
4. portfolio allocator
5. widened market whitelist without re-validation
